"""
Before running, make sure that:
  - ONLY_LABELED_IMAGES = True  in config_v1.py
  - CONF_THRESHOLD_GOOD_AND_BAD_BOX = 0.01  in config_v1.py

Run inside Wheat3DGS-Yolo-SAM/ directory with: python metrics/metrics_yolo_v1.py

Per-image metrics are computed first (except AP which is pooled globally).
Aggregated mean ± std is printed at the end. JSON saved to metrics/results/metrics_yolo_v1.json

------------------------------------------------------------------------
TABLE OF CONTENTS  (in file order)
------------------------------------------------------------------------
  CONFIG ..................... output dirs, MATCHING_IOU_THRESHOLD
  FILE LOADING ............... load_gt_boxes, load_pred_boxes, load_pred_boxes_with_conf
  PRINT & JSON OUTPUT ........ print_single_result, print_aggregated_results, save_results_json

  [Part 1] GT MATCHING + PRECISION / RECALL / F1
           → compute_iou_matrix, match_boxes, compute_precision_recall_f1,
             compute_stats, compute_iou_stats

  [Part 2] MATCH VISUALIZATION      save_match_visualization
           → colored boxes image: blue=TP, orange=FP, red=FN

  [Part 3] TP IoU HISTOGRAMS        save_iou_histogram
           → histogram of TP/FP/FN IoU distributions

  [Part 4] BOX SIZE RATIO           compute_box_size_ratio
           → pred area / GT area for each TP match

  [Part 5] FP/FN SPATIAL HEATMAPS   _build_density_grid, _save_single_heatmap,
                                     save_fp_fn_heatmap, save_aggregated_heatmap
           → where in the image FP and FN errors cluster

  [Part 6] AP & PR CURVE            compute_ap, save_pr_curve
           → precision-recall curve + average precision (COCO 101-point)

  SINGLE-IMAGE EVAL .......... evaluate_single_image  (runs parts 1-4 for one image)
  AGGREGATED EVAL ............ evaluate_all_plots      (entry point, runs parts 5-6 too)
------------------------------------------------------------------------
"""


import os
import sys
import json
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # headless backend — prevents Qt/display warnings in WSL
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Add parent directory so that config_v1 can be import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_v1 import DATA_DIR, CONF_THRESHOLD_GOOD_BOX, CONF_THRESHOLD_GOOD_AND_BAD_BOX, IOU_THRESHOLD


# =====================================================================
# Config
# =====================================================================

# IoU threshold for matching predicted to the GT boxes. This is SEPARATE from IOU_THRESHOLD in config_v1.py
MATCHING_IOU_THRESHOLD = 0.35 # was 0.5 default

# Where to save JSON results
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Subfolder for match visualization images — cleared on every run
VIZ_DIR = os.path.join(RESULTS_DIR, "match_viz")

# Subfolder for TP IoU histograms — cleared on every run
HIST_DIR = os.path.join(RESULTS_DIR, "TP_IoU_histograms")

# Subfolders for FP and FN spatial heatmaps — cleared on every run
HEATMAP_FP_DIR = os.path.join(RESULTS_DIR, "heatmaps_FP")
HEATMAP_FN_DIR = os.path.join(RESULTS_DIR, "heatmaps_FN")

# Subfolder for PR curves — cleared on every run
PR_CURVE_DIR = os.path.join(RESULTS_DIR, "pr_curves")



# =====================================================================
# File Loading
# =====================================================================

def load_gt_boxes(label_path, img_w, img_h):
    """Load GT boxes from manual label txt file, then convert to absolute pixel positions."""
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = (float(x) for x in parts[:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def load_pred_boxes(bbox_path):
    """Load predicted boxes from a .pt file saved by yolo_v1.py."""
    tensor = torch.load(bbox_path, weights_only=True)
    if tensor.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return tensor[:, :4].numpy().astype(np.float32)


def load_pred_boxes_with_conf(bboxes_with_conf_path):
    """Load 5-col [x1,y1,x2,y2,conf] tensor saved by yolo_v1.py for AP evaluation.
    Returns None if the file doesn't exist (re-run YOLO to generate it).
    """
    if not os.path.exists(bboxes_with_conf_path):
        return None
    tensor = torch.load(bboxes_with_conf_path, weights_only=True)
    if tensor.numel() == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return tensor.numpy().astype(np.float32)



# =====================================================================
# Print & JSON Output
# =====================================================================

def print_single_result(result, iou_threshold):
    diff = result['pred_count'] - result['gt_count']
    print(f"  GT count:        {result['gt_count']}")
    print(f"  Pred count:      {result['pred_count']}  (diff vs GT: {diff:+d})")
    print(f"  TP / FP / FN:    {result['tp']} / {result['fp']} / {result['fn']}")
    print(f"  Precision:       {result['precision']:.4f}")
    print(f"  Recall:          {result['recall']:.4f}")
    print(f"  F1 Score:        {result['f1']:.4f}")
    if result['iou_stats']:
        s = result['iou_stats']
        print(f"  TP IoU:          mean={s['mean']:.3f}  median={s['median']:.3f}  "
              f"std={s['std']:.3f}  min={s['min']:.3f}  max={s['max']:.3f}")
    if result['size_ratio']:
        s = result['size_ratio']
        print(f"  Box size ratio   mean={s['mean']:.3f}  median={s['median']:.3f}  "
              f"std={s['std']:.3f}  (pred area / GT area, >1 means pred is bigger)")
    print()


def print_aggregated_results(per_plot_results):
    scalar_metrics = [
        ('pred_count', 'Pred count'),
        ('gt_count', 'GT count'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1','F1 Score'),
    ]
    for key, label in scalar_metrics:
        vals = [r[key] for r in per_plot_results]
        print(f"  {label:<22}  {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    iou_means = [r['iou_stats']['mean'] for r in per_plot_results if r['iou_stats']]
    if iou_means:
        print(f"  {'TP IoU mean':<22}  {np.mean(iou_means):.4f} ± {np.std(iou_means):.4f}")

    ratio_means = [r['size_ratio']['mean'] for r in per_plot_results if r['size_ratio']]
    if ratio_means:
        print(f"  {'Box size ratio mean':<22}  {np.mean(ratio_means):.4f} ± {np.std(ratio_means):.4f}")
    print()


def save_results_json(per_plot_results, iou_threshold, conf_threshold, ap=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    aggregated = {}
    for key in ['pred_count', 'gt_count', 'precision', 'recall', 'f1']:
        vals = [r[key] for r in per_plot_results]
        aggregated[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    iou_means = [r['iou_stats']['mean'] for r in per_plot_results if r['iou_stats']]
    if iou_means:
        aggregated['tp_iou_mean'] = {'mean': float(np.mean(iou_means)), 'std': float(np.std(iou_means))}

    ratio_means = [r['size_ratio']['mean'] for r in per_plot_results if r['size_ratio']]
    if ratio_means:
        aggregated['size_ratio_mean'] = {'mean': float(np.mean(ratio_means)), 'std': float(np.std(ratio_means))}

    if ap is not None:
        aggregated['ap'] = float(ap)

    output = {
        'config': {
            'matching_iou_threshold':       iou_threshold,
            'conf_threshold_good_box':      conf_threshold,
            'conf_threshold_nms_floor':     CONF_THRESHOLD_GOOD_AND_BAD_BOX,
            'yolo_nms_iou_threshold':       IOU_THRESHOLD,
        },
        'per_plot':   per_plot_results,
        'aggregated': aggregated,
    }

    out_path = os.path.join(RESULTS_DIR, 'metrics_yolo_v1.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {out_path}\n")




# =====================================================================
# [Part 1] GT Matching + Precision / Recall / F1
# =====================================================================

def compute_iou_matrix(pred_boxes, gt_boxes):
    """Vectorized pairwise IoU between all predicted and GT boxes and returns as np.ndarray shape (N_pred, N_gt)."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)

    p = pred_boxes[:, np.newaxis, :]  # (N, 1, 4)
    g = gt_boxes[np.newaxis, :, :]    # (1, M, 4)

    inter_x1 = np.maximum(p[:, :, 0], g[:, :, 0])
    inter_y1 = np.maximum(p[:, :, 1], g[:, :, 1])
    inter_x2 = np.minimum(p[:, :, 2], g[:, :, 2])
    inter_y2 = np.minimum(p[:, :, 3], g[:, :, 3])
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area   = (gt_boxes[:, 2]  - gt_boxes[:, 0])  * (gt_boxes[:, 3]  - gt_boxes[:, 1])
    union_area = pred_area[:, np.newaxis] + gt_area[np.newaxis, :] - inter_area

    return np.where(union_area > 0, inter_area / union_area, 0.0).astype(np.float32)


def match_boxes(iou_matrix, iou_threshold):
    """ Greedy matching: for each predicted box match it  with the best unmatched GT box with IoU >= iou_threshold,
    while each GT box matched max once. Then returns:
    tp_matches: list of (pred_idx, gt_idx, iou_value)
    fp_idxs: pred indices with no GT match (False Positives)
    fn_idxs: GT indices never matched (False Negatives)"""
    n_pred, n_gt = iou_matrix.shape
    if n_pred == 0:
        return [], [], list(range(n_gt))
    if n_gt == 0:
        return [], list(range(n_pred)), []

    matched_gt = set()
    tp_matches, fp_idxs = [], []

    for pred_idx in range(n_pred):
        row = iou_matrix[pred_idx].copy()
        if matched_gt:
            row[list(matched_gt)] = -1  # exclude already matched GTs
        best_gt_idx = int(np.argmax(row))
        if row[best_gt_idx] >= iou_threshold:
            tp_matches.append((pred_idx, best_gt_idx, float(row[best_gt_idx])))
            matched_gt.add(best_gt_idx)
        else:
            fp_idxs.append(pred_idx)

    fn_idxs = [i for i in range(n_gt) if i not in matched_gt]
    return tp_matches, fp_idxs, fn_idxs


def compute_precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_stats(values):
    """If not empty, then compute descriptive stats and return dict (floats)"""
    if not values:
        return None
    arr = np.array(values, dtype=np.float32)
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }


def compute_iou_stats(tp_matches):
    """IoU values of all TP matches."""
    return compute_stats([iou for _, _, iou in tp_matches])



# =====================================================================
# [Part 2] Match Visualization
# =====================================================================

def save_match_visualization(image_path, pred_boxes, gt_boxes, tp_matches, fp_idxs, fn_idxs, out_path, iou_matrix):
    """Draw matching results on the image and save it.
    Blue = matched pred boxes (TP), Yellow-orange = unmatched pred boxes (FP), Red = unmatched GT boxes (FN).
    Each box is labeled with its IoU value (TP: matched IoU, FP: best IoU it achieved, FN: best IoU any pred had).
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    line_width = 4
    font_size = 28

    # try to load a real TTF font for readable text; fall back to PIL default if not found
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    def draw_box_with_label(box, color, label):
        """Draw a rectangle and a label above it with white text and colored stroke outline."""
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        text_x, text_y = x1, max(0, y1 - font_size - 2)
        # white text with colored stroke — same two-pass trick as yolo_v1.py
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font,
                  stroke_width=2, stroke_fill=color)

    # 1. blue: matched pred boxes (TP) — show matched IoU
    for pred_idx, gt_idx, iou_val in tp_matches:
        draw_box_with_label(pred_boxes[pred_idx], (30, 120, 255), f"TP {iou_val:.2f}")

    # 2. yellow-orange: unmatched pred boxes (FP) — show best IoU it achieved against any GT
    for pred_idx in fp_idxs:
        best_iou = float(np.max(iou_matrix[pred_idx])) if iou_matrix.shape[1] > 0 else 0.0
        draw_box_with_label(pred_boxes[pred_idx], (255, 160, 0), f"FP {best_iou:.2f}")

    # 3. red: unmatched GT boxes (FN) — show best IoU any pred had against this GT
    for gt_idx in fn_idxs:
        best_iou = float(np.max(iou_matrix[:, gt_idx])) if iou_matrix.shape[0] > 0 else 0.0
        draw_box_with_label(gt_boxes[gt_idx], (220, 30, 30), f"FN {best_iou:.2f}")

    img.save(out_path, quality=92) # bit higher quality than 90
    print(f"  Viz saved: {out_path}")




# =====================================================================
# [Part 3] TP IoU Histograms
# =====================================================================

def save_iou_histogram(tp_ious, title, out_path, iou_threshold, show_total=False,
                       fp_best_ious=None, fn_best_ious=None):
    """Save a histogram of TP/FP/FN IoU values to a PNG file.
    TP=blue, FP=orange, FN=red. Smaller groups are plotted on top so nothing is fully hidden.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    hist_kwargs = dict(bins=20, range=(0.0, 1.0), edgecolor='white', alpha=0.75)

    # collect all three groups with their display properties
    groups = [
        (tp_ious,                    'steelblue', f'TP ({len(tp_ious)})'),
        (fp_best_ious or [],         '#FF9500',   f'FP ({len(fp_best_ious or [])})'),
        (fn_best_ious or [],         '#DC1E1E',   f'FN ({len(fn_best_ious or [])})'),
    ]

    # plot largest group first so smaller ones appear on top and aren't hidden
    for values, color, label in sorted(groups, key=lambda g: len(g[0]), reverse=True):
        if values:
            counts, _, patches = ax.hist(values, color=color, label=label, **hist_kwargs)
            # write count above each bar
            for count, patch in zip(counts, patches):
                if count > 0:
                    ax.text(patch.get_x() + patch.get_width() / 2, count + 0.1, str(int(count)),
                            ha='center', va='bottom', fontsize=7)

    # vertical line at the matching threshold
    ax.axvline(iou_threshold, color='black', linestyle='--', linewidth=1.5, label=f'threshold {iou_threshold}')

    # x-axis: tick every 0.05, label every 0.1
    tick_positions = [round(v * 0.05, 2) for v in range(21)]
    tick_labels = [f'{pos:.1f}' if v % 2 == 0 else '' for v, pos in enumerate(tick_positions)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel('IoU')
    ax.set_ylabel('Count')

    # add total TP count to title for individual images
    full_title = f"{title}  (TP: {len(tp_ious)})" if show_total else title
    ax.set_title(full_title)

    ax.set_xlim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Histogram saved: {out_path}")




# =====================================================================
# [Part 4] Box Size Ratio
# =====================================================================

def compute_box_size_ratio(pred_boxes, gt_boxes, tp_matches):
    """For each TP match: (predicted area) / (GT area) = ratio."""
    ratios = []
    for pred_idx, gt_idx, _ in tp_matches:
        pb, gb = pred_boxes[pred_idx], gt_boxes[gt_idx]
        pred_area = (pb[2] - pb[0]) * (pb[3] - pb[1])
        gt_area = (gb[2] - gb[0]) * (gb[3] - gb[1])
        if gt_area > 0:
            ratios.append(float(pred_area / gt_area))
    return compute_stats(ratios)




# =====================================================================
# [Part 5] FP/FN Spatial Heatmaps
# =====================================================================

def _build_density_grid(boxes, img_w, img_h, grid_size):
    """Count box centers into a grid and apply Gaussian blur."""
    from scipy.ndimage import gaussian_filter
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        col = int(cx / img_w * grid_size)
        row = int(cy / img_h * grid_size)
        grid[min(row, grid_size - 1), min(col, grid_size - 1)] += 1
    return gaussian_filter(grid, sigma=1.5)


def _save_single_heatmap(image_path, grid, img_w, img_h, cmap, label, title, out_path, vmax=None):
    """Overlay one density grid on the original image and save.
    vmax: shared scale across plots so colors are comparable — if None, uses this grid's own max.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.colors as mcolors

    # YlOrRd: position 0.0=pale yellow, 0.15=saturated yellow, 0.4=orange, 0.7=red, 1.0=dark red
    # start at 0.15 to get a punchy saturated yellow, not the washed-out pale one
    cmap_trunc = mcolors.LinearSegmentedColormap.from_list(
        'trunc', matplotlib.colormaps[cmap](np.linspace(0.15, 1.0, 256)))

    scale_max = vmax if vmax is not None else (grid.max() or 1)

    fig, ax = plt.subplots(figsize=(13, 8))
    img = Image.open(image_path).convert("RGB")
    ax.imshow(img)

    # mask below 10% of the global max to cut Gaussian blur halo
    threshold = scale_max * 0.1
    masked = np.ma.masked_where(grid < threshold, grid)
    ax.imshow(masked, cmap=cmap_trunc, alpha=0.80, extent=[0, img_w, img_h, 0],
              vmin=0, vmax=scale_max, aspect='auto')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=cmap_trunc, norm=plt.Normalize(0, scale_max))
    fig.colorbar(sm, cax=cax, label=label)

    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Heatmap saved: {out_path}")


def save_fp_fn_heatmap(image_path, fp_boxes, fn_boxes, img_w, img_h, out_path_fp, out_path_fn, grid_size=50):
    """Save two separate heatmap images: one for FP and one for FN, each overlaid on the original.
    Each image is normalized independently to its own max so its full color range is used.
    """
    fp_grid = _build_density_grid(fp_boxes, img_w, img_h, grid_size)
    fn_grid = _build_density_grid(fn_boxes, img_w, img_h, grid_size)

    _save_single_heatmap(image_path, fp_grid, img_w, img_h,
                         cmap='YlOrRd',
                         label=f'FP density ({len(fp_boxes)} total)',
                         title=f'FP spatial heatmap — {len(fp_boxes)} false positives',
                         out_path=out_path_fp, vmax=None)

    _save_single_heatmap(image_path, fn_grid, img_w, img_h,
                         cmap='YlOrRd',
                         label=f'FN density ({len(fn_boxes)} total)',
                         title=f'FN spatial heatmap — {len(fn_boxes)} false negatives',
                         out_path=out_path_fn, vmax=None)


def save_aggregated_heatmap(all_boxes_list, all_img_ws, all_img_hs, label, title, out_path, grid_size=50):
    """Aggregate box centers from all plots into one normalized heatmap (no background image).
    Positions are normalized to [0, 1] before binning so different image sizes are comparable.
    """
    from scipy.ndimage import gaussian_filter
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.colors as mcolors

    cmap_trunc = mcolors.LinearSegmentedColormap.from_list(
        'trunc', matplotlib.colormaps['YlOrRd'](np.linspace(0.15, 1.0, 256)))

    # build grid in normalized [0,1] space so all plots contribute equally regardless of resolution
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for boxes, img_w, img_h in zip(all_boxes_list, all_img_ws, all_img_hs):
        for x1, y1, x2, y2 in boxes:
            cx_norm = ((x1 + x2) / 2) / img_w
            cy_norm = ((y1 + y2) / 2) / img_h
            col = int(cx_norm * grid_size)
            row = int(cy_norm * grid_size)
            grid[min(row, grid_size - 1), min(col, grid_size - 1)] += 1

    grid = gaussian_filter(grid, sigma=1.5)

    fig, ax = plt.subplots(figsize=(8, 7))
    threshold = grid.max() * 0.1
    masked = np.ma.masked_where(grid < threshold, grid)
    im = ax.imshow(masked, cmap=cmap_trunc, origin='upper', vmin=0, aspect='auto',
                   extent=[0, 1, 1, 0])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    fig.colorbar(im, cax=cax, label=label)

    ax.set_xlabel('x (normalized)')
    ax.set_ylabel('y (normalized)')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Aggregated heatmap saved: {out_path}")





# =====================================================================
# [Part 6] AP & PR Curve
# =====================================================================

def compute_ap(all_pred_entries, all_gt_boxes_list, iou_threshold):
    """Compute AP by globally pooling all predictions across all images, sorted by confidence.
    all_pred_entries: list of (conf, x1, y1, x2, y2, img_idx)
    all_gt_boxes_list: list of np.ndarray [N_gt, 4] for each image (same order as img_idx)
    Returns: ap (float), precisions list, recalls list (sorted by confidence desc)
    """
    n_gt_total = sum(len(gt) for gt in all_gt_boxes_list)
    if n_gt_total == 0 or len(all_pred_entries) == 0:
        return 0.0, [], []

    # sort all predictions globally by confidence descending
    sorted_preds = sorted(all_pred_entries, key=lambda x: -x[0])

    # track which GT boxes have been matched per image
    matched_gt = [set() for _ in all_gt_boxes_list]
    tp_list, fp_list = [], []

    for _conf, x1, y1, x2, y2, img_idx in sorted_preds:
        gt_boxes = all_gt_boxes_list[img_idx]
        if len(gt_boxes) == 0:
            tp_list.append(0)
            fp_list.append(1)
            continue

        pred_box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        iou_row = compute_iou_matrix(pred_box, gt_boxes)[0]  # shape (N_gt,)

        # find best unmatched GT box
        best_gt_idx, best_iou = -1, -1.0
        for gt_idx in range(len(gt_boxes)):
            if gt_idx not in matched_gt[img_idx] and iou_row[gt_idx] > best_iou:
                best_iou = iou_row[gt_idx]
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            tp_list.append(1)
            fp_list.append(0)
            matched_gt[img_idx].add(best_gt_idx)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    recalls    = tp_cum / n_gt_total
    precisions = tp_cum / (tp_cum + fp_cum)

    # COCO-style 101-point interpolated AP
    ap = 0.0
    for r_thr in np.linspace(0, 1, 101):
        prec_at_r = precisions[recalls >= r_thr]
        ap += float(np.max(prec_at_r)) if len(prec_at_r) > 0 else 0.0
    ap /= 101

    # confidence for each point on the curve (sorted descending, one per pred)
    sorted_confs = [float(e[0]) for e in sorted_preds]

    return float(ap), precisions.tolist(), recalls.tolist(), sorted_confs


def save_pr_curve(precisions, recalls, confs, ap, iou_threshold, title, out_path,
                  total_preds=None, tp=None, fp=None, fn=None, mark_every=50):
    """Save a Precision-Recall curve image.
    mark_every: label every k-th prediction point on the curve with its confidence value.
    total_preds/tp/fp/fn: counts at the fixed CONF_THRESHOLD_GOOD_BOX for the stats box.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax_top = None
    ax.plot(recalls, precisions, color='steelblue', linewidth=2)
    ax.fill_between(recalls, precisions, alpha=0.15, color='steelblue')

    # major ticks every 0.1 with label, minor ticks every 0.01 without label
    major_ticks = np.linspace(0.0, 1.0, 11)
    minor_ticks = np.linspace(0.0, 1.0, 101)
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([f'{v:.1f}' for v in major_ticks])
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticklabels([f'{v:.1f}' for v in major_ticks])
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(True, which='major', alpha=0.4)
    ax.grid(True, which='minor', alpha=0.2)

    recalls_arr    = np.array(recalls)
    precisions_arr = np.array(precisions)

    # mark the exact max recall the curve reaches
    if recalls:
        max_recall = max(recalls)
        ax.axvline(max_recall, color='tomato', linestyle='--', linewidth=1.2)
        ax.text(max_recall + 0.005, 0.02, f'max recall\n{max_recall:.4f}',
                color='tomato', fontsize=8, va='bottom')

    # min precision horizontal line
    if len(precisions_arr) > 0:
        min_prec = float(precisions_arr.min())
        ax.axhline(min_prec, color='darkorange', linestyle=':', linewidth=1.2)
        ax.text(0.005, min_prec + 0.005, f'min prec {min_prec:.4f}',
                color='darkorange', fontsize=8, va='bottom')

    # vertical lines every 0.05 recall, precision values shown on a top x-axis
    r_marks = np.arange(0.05, 1.0, 0.05)
    prec_at_marks = []
    for r_mark in r_marks:
        idx = int(np.argmin(np.abs(recalls_arr - r_mark)))
        prec_at_marks.append(float(precisions_arr[idx]))
        ax.axvline(r_mark, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)

    # top x-axis: same range, ticks at r_marks, labeled with precision values
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(r_marks)
    ax_top.set_xticklabels([f'{p:.2f}' for p in prec_at_marks], fontsize=6.5, rotation=45, ha='left')
    ax_top.set_xlabel('Precision at recall', fontsize=8)

    # mark every k-th prediction on the curve with its confidence value (green markers)
    conf_color = '#2ca02c'  # green — not used by any other element in the plot
    if confs and mark_every > 0:
        indices = list(range(mark_every - 1, len(recalls), mark_every))
        for idx_num, i in enumerate(indices):
            label = f'conf value (every {mark_every}th pred)' if idx_num == 0 else None
            ax.plot(recalls[i], precisions[i], 'o', color=conf_color, markersize=3, label=label)
            ax.annotate(f'{confs[i]:.2f}', xy=(recalls[i], precisions[i]),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=7, color=conf_color)
        ax.legend()

    # stats box: total preds + TP/FP/FN at the fixed good-box threshold
    if total_preds is not None:
        stats_lines = [
            f'Total preds (all conf (>=0.01)): {total_preds}',
            f'above CONF_THRESHOLD_GOOD_BOX (>= {CONF_THRESHOLD_GOOD_BOX}):',
            f'  TP: {tp}   FP: {fp}   (TP+FP = {tp+fp} preds)',
            f'  FN: {fn}   (missed GT boxes, not preds)',
        ]
        ax.text(0.01, 0.01, '\n'.join(stats_lines), transform=ax.transAxes,
                fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    if ax_top is not None:
        ax_top.set_xlim(0.0, 1.0)  # must match after ax xlim is fixed
    ax.set_title(f"{title}\nAP@IoU{iou_threshold:.2f} = {ap:.4f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  PR curve saved: {out_path}")





# =====================================================================
# Single-Image Eval
# =====================================================================

def evaluate_single_image(pred_pt_path, gt_label_path, image_path, iou_threshold, viz_out_path=None):
    """Run all metrics for one (predicted, GT) image pair and return dict if not empty (otherwise None)."""
    if not os.path.exists(pred_pt_path):
        print(f"[SKIP] No bbox file found: {pred_pt_path}")
        print(f"Run main_v1.py first to generate YOLO predictions.")
        return None
    if not os.path.exists(image_path):
        print(f"[SKIP] Image not found: {image_path}")
        return None

    with Image.open(image_path) as img:
        img_w, img_h = img.size #(width, height)
    gt_boxes = load_gt_boxes(gt_label_path, img_w, img_h)
    pred_boxes = load_pred_boxes(pred_pt_path)

    iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)
    tp_matches, fp_idxs, fn_idxs = match_boxes(iou_mat, iou_threshold)

    tp, fp, fn = len(tp_matches), len(fp_idxs), len(fn_idxs)
    precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)

    if viz_out_path is not None:
        save_match_visualization(image_path, pred_boxes, gt_boxes, tp_matches, fp_idxs, fn_idxs, viz_out_path, iou_mat)

    return {
        'pred_count': len(pred_boxes),
        'gt_count': len(gt_boxes),
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou_stats': compute_iou_stats(tp_matches),
        'size_ratio': compute_box_size_ratio(pred_boxes, gt_boxes, tp_matches),
        'tp_ious': [iou for _, _, iou in tp_matches],  # raw values for histogram
        # best IoU each FP pred achieved vs any GT (all below threshold)
        'fp_best_ious': [float(np.max(iou_mat[i])) if iou_mat.shape[1] > 0 else 0.0 for i in fp_idxs],
        # best IoU any pred had vs each FN GT box (all below threshold)
        'fn_best_ious': [float(np.max(iou_mat[:, j])) if iou_mat.shape[0] > 0 else 0.0 for j in fn_idxs],
        # raw boxes for spatial heatmap
        'fp_boxes': pred_boxes[fp_idxs].tolist() if len(fp_idxs) > 0 else [],
        'fn_boxes': gt_boxes[fn_idxs].tolist() if len(fn_idxs) > 0 else [],
        'img_w': img_w,
        'img_h': img_h,
    }





# =====================================================================
# Aggregated Eval
# =====================================================================

def find_labeled_plots(data_dir):
    """Collect all GT files and return as list (plot_dir, gt_label_path, image_stem)."""
    labeled = []
    for plot_name in sorted(os.listdir(data_dir)):
        plot_dir = os.path.join(data_dir, plot_name)
        label_dir = os.path.join(plot_dir, 'manual_label')
        if not os.path.isdir(label_dir):
            continue
        for fname in sorted(os.listdir(label_dir)):
            if fname.endswith('.txt'):
                stem = os.path.splitext(fname)[0]
                labeled.append((plot_dir, os.path.join(label_dir, fname), stem))
    return labeled


def evaluate_all_plots(data_dir=None, iou_threshold=None):
    """Evaluate all plots, then prints per-plot results and mean ± std aggregate, then saves full results as JSON."""
    if data_dir is None:
        data_dir = DATA_DIR
    if iou_threshold is None:
        iou_threshold = MATCHING_IOU_THRESHOLD

    print(f"\n{'='*58}")
    print(f" YOLO EVALUATION vs MANUAL LABELS")
    print(f"{'='*58}")
    print(f" Data dir:              {data_dir}")
    print(f" Conf threshold (good): {CONF_THRESHOLD_GOOD_BOX}  (CONF_THRESHOLD_GOOD_BOX — used for precision/recall/F1)")
    print(f" Conf threshold (NMS floor): {CONF_THRESHOLD_GOOD_AND_BAD_BOX}  (CONF_THRESHOLD_GOOD_AND_BAD_BOX — floor for AP curve)")
    print(f" Matching IoU thr:      {iou_threshold}  (MATCHING_IOU_THRESHOLD)")
    print(f" YOLO NMS IoU thr:      {IOU_THRESHOLD}  (IOU_THRESHOLD — used during YOLO inference, not here)")
    print(f"{'='*58}\n")

    # wipe and recreate output folders so they only contain images from this run
    for folder in [VIZ_DIR, HIST_DIR, HEATMAP_FP_DIR, HEATMAP_FN_DIR, PR_CURVE_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    labeled_plots = find_labeled_plots(data_dir)
    if not labeled_plots:
        print("No labeled plots found. Expected: data/<plot>/manual_label/<name>.txt")
        return

    print(f"Found {len(labeled_plots)} labeled image(s).\n")

    per_plot_results = []
    # for AP: collect all pred entries globally and GT boxes per image
    all_pred_entries = []   # list of (conf, x1, y1, x2, y2, img_idx)
    all_gt_boxes_for_ap = []  # np.ndarray per image, same order as img_idx
    ap_data_available = True

    for plot_dir, gt_label_path, stem in labeled_plots:
        plot_name = os.path.basename(plot_dir)
        print(f"--- Plot: {plot_name}  |  Image: {stem} ---")

        image_path = os.path.join(plot_dir, 'images', stem + '.png')
        pred_pt_path = os.path.join(plot_dir, 'bboxes', stem + '.pt')
        viz_out_path = os.path.join(VIZ_DIR, f"{plot_name}_{stem}_matches.jpg")

        result = evaluate_single_image(pred_pt_path, gt_label_path, image_path, iou_threshold, viz_out_path)
        if result is None:
            continue

        result['plot_name'] = plot_name
        result['image_stem'] = stem
        result['_plot_dir'] = plot_dir   # needed for AP data collection
        per_plot_results.append(result)
        print_single_result(result, iou_threshold)

        # store image path for heatmap rendering after global max is known
        result['_image_path'] = image_path

        # per-image TP IoU histogram
        hist_path = os.path.join(HIST_DIR, f"{plot_name}_{stem}_iou_hist.png")
        save_iou_histogram(result['tp_ious'], f"TP IoU — {plot_name} / {stem}", hist_path, iou_threshold,
                           show_total=True, fp_best_ious=result['fp_best_ious'], fn_best_ious=result['fn_best_ious'])
        print()

    if not per_plot_results:
        print("No results computed. Make sure YOLO has been run (bboxes/ folder must exist).")
        return

    # per-image heatmaps (each normalized independently)
    print("Saving heatmaps...")
    for r in per_plot_results:
        plot_name, stem = r['plot_name'], r['image_stem']
        heatmap_fp_path = os.path.join(HEATMAP_FP_DIR, f"{plot_name}_{stem}_heatmap_FP.jpg")
        heatmap_fn_path = os.path.join(HEATMAP_FN_DIR, f"{plot_name}_{stem}_heatmap_FN.jpg")
        save_fp_fn_heatmap(r['_image_path'], r['fp_boxes'], r['fn_boxes'],
                           r['img_w'], r['img_h'], heatmap_fp_path, heatmap_fn_path)

    # aggregated heatmaps across all plots (normalized position, no background)
    save_aggregated_heatmap(
        [r['fp_boxes'] for r in per_plot_results],
        [r['img_w'] for r in per_plot_results],
        [r['img_h'] for r in per_plot_results],
        label='FP density (all plots)',
        title=f'FP aggregated heatmap — {sum(r["fp"] for r in per_plot_results)} total false positives',
        out_path=os.path.join(HEATMAP_FP_DIR, 'aggregated_FP_heatmap.jpg'))

    save_aggregated_heatmap(
        [r['fn_boxes'] for r in per_plot_results],
        [r['img_w'] for r in per_plot_results],
        [r['img_h'] for r in per_plot_results],
        label='FN density (all plots)',
        title=f'FN aggregated heatmap — {sum(r["fn"] for r in per_plot_results)} total false negatives',
        out_path=os.path.join(HEATMAP_FN_DIR, 'aggregated_FN_heatmap.jpg'))
    print()

    print(f"\n{'='*58}")
    print(f"AGGREGATED RESULTS (mean ± std across {len(per_plot_results)} plot(s))")
    print(f"{'='*58}")
    print_aggregated_results(per_plot_results)

    # aggregated TP IoU histogram across all plots
    all_tp_ious = [iou for r in per_plot_results for iou in r['tp_ious']]
    all_fp_ious = [iou for r in per_plot_results for iou in r['fp_best_ious']]
    all_fn_ious = [iou for r in per_plot_results for iou in r['fn_best_ious']]
    save_iou_histogram(all_tp_ious, f"IoU distribution — all plots aggregated",
                       os.path.join(HIST_DIR, "aggregated_iou_hist.png"), iou_threshold,
                       fp_best_ious=all_fp_ious, fn_best_ious=all_fn_ious)

    # AP computation — requires bboxes_with_conf/ folder from yolo_v1.py
    print(f"\n{'='*58}")
    print(f"AVERAGE PRECISION (AP)")
    print(f"{'='*58}")
    for idx, r in enumerate(per_plot_results):
        bboxes_with_conf_path = os.path.join(r['_plot_dir'], 'bboxes_with_conf', r['image_stem'] + '.pt')
        with_conf = load_pred_boxes_with_conf(bboxes_with_conf_path)
        if with_conf is None:
            ap_data_available = False
            print(f"[AP] bboxes_with_conf not found for {r['plot_name']}/{r['image_stem']}")
            print(f"     → Re-run YOLO (main_v1.py) to generate bboxes_with_conf/ folder.")
            break
        for row in with_conf:
            x1, y1, x2, y2, conf = row
            all_pred_entries.append((float(conf), float(x1), float(y1), float(x2), float(y2), idx))
        # reload GT boxes (small overhead — needed here separately from evaluate_single_image)
        all_gt_boxes_for_ap.append(load_gt_boxes(
            os.path.join(r['_plot_dir'], 'manual_label', r['image_stem'] + '.txt'),
            r['img_w'], r['img_h']))

    ap = None
    if ap_data_available and len(all_gt_boxes_for_ap) > 0:
        ap, precisions, recalls, confs = compute_ap(all_pred_entries, all_gt_boxes_for_ap, iou_threshold)
        total_tp = sum(r['tp'] for r in per_plot_results)
        total_fp = sum(r['fp'] for r in per_plot_results)
        total_fn = sum(r['fn'] for r in per_plot_results)
        print(f"  AP@IoU{iou_threshold:.2f} = {ap:.4f}  "
              f"(NMS floor: {CONF_THRESHOLD_GOOD_AND_BAD_BOX}, {len(all_pred_entries)} total preds, "
              f"{sum(len(g) for g in all_gt_boxes_for_ap)} GT boxes)")
        pr_out = os.path.join(PR_CURVE_DIR, 'pr_curve_aggregated.png')
        save_pr_curve(precisions, recalls, confs, ap, iou_threshold,
                      f'PR Curve — all plots aggregated ({len(per_plot_results)} image(s))', pr_out,
                      total_preds=len(all_pred_entries), tp=total_tp, fp=total_fp, fn=total_fn)
    print()

    save_results_json(per_plot_results, iou_threshold, CONF_THRESHOLD_GOOD_BOX, ap=ap)

    return per_plot_results


# =====================================================================
# Entry Point
# =====================================================================

if __name__ == "__main__":
    evaluate_all_plots()

