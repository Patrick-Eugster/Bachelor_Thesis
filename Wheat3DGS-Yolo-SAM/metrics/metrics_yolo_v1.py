"""
Run inside Wheat3DGS-Yolo-SAM/ directory with: python metrics/metrics_yolo_v1.py
    
First it computes some metrics for each manual labeled image individually (except AP).

Then two aggregating approaches, one with mean + std and the other one for AP should be pooled.

JSON saved to metrics/results/metrics_yolo_v1.json
"""


import os
import sys
import json
import numpy as np
import torch
from PIL import Image

# Add parent directory so that config_v1 can be import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_v1 import DATA_DIR, CONF_THRESHOLD_GOOD_BOX, IOU_THRESHOLD


# =====================================================================
# Metrics Config
# =====================================================================

# IoU threshold for matching predicted to the GT boxes. This is SEPARATE from IOU_THRESHOLD in config_v1.py
MATCHING_IOU_THRESHOLD = 0.5

# Where to save JSON results
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")



# =====================================================================
# Helper Functions
# =====================================================================

# Loading Helper Functions
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



# Printing Helper Functions
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




# Saving (JSON) Hepler Functions
def save_results_json(per_plot_results, iou_threshold, conf_threshold):
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

    output = {
        'config': {
            'matching_iou_threshold': iou_threshold,
            'conf_threshold':         conf_threshold,
            'yolo_nms_iou_threshold': IOU_THRESHOLD,
        },
        'per_plot':   per_plot_results,
        'aggregated': aggregated,
    }

    out_path = os.path.join(RESULTS_DIR, 'metrics_yolo_v1.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {out_path}\n")




# =====================================================================
# Main Computation Functions
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


# TODO: Compute AP (Average Precision) by sweeping confidence threshold.
# Requires per-box confidence scores saved to a separate bboxes_eval/*.pt file
# (5 columns: x1, y1, x2, y2, conf). Currently yolo_v1.py only saves 4 columns.
def compute_ap(pred_boxes, pred_confs, gt_boxes, iou_threshold):
    pass


# TODO: spatial heatmap of FP/FN locations
def compute_fp_fn_heatmap(pred_boxes, gt_boxes, fp_idxs, fn_idxs, img_w, img_h):
    pass



# =====================================================================
# Single-Image Eval
# =====================================================================

def evaluate_single_image(pred_pt_path, gt_label_path, image_path, iou_threshold):
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

    return {
        'pred_count': len(pred_boxes),
        'gt_count': len(gt_boxes),
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou_stats': compute_iou_stats(tp_matches),
        'size_ratio': compute_box_size_ratio(pred_boxes, gt_boxes, tp_matches),
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
    print(f" Conf threshold:        {CONF_THRESHOLD_GOOD_BOX}  (CONF_THRESHOLD_GOOD_BOX from config_v1.py)")
    print(f" Matching IoU thr:      {iou_threshold}  (MATCHING_IOU_THRESHOLD)")
    print(f" YOLO NMS IoU thr:      {IOU_THRESHOLD}  (IOU_THRESHOLD — used during YOLO inference, not here)")
    print(f"{'='*58}\n")


    labeled_plots = find_labeled_plots(data_dir)
    if not labeled_plots:
        print("No labeled plots found. Expected: data/<plot>/manual_label/<name>.txt")
        return

    print(f"Found {len(labeled_plots)} labeled image(s).\n")

    per_plot_results = []

    for plot_dir, gt_label_path, stem in labeled_plots:
        plot_name = os.path.basename(plot_dir)
        print(f"--- Plot: {plot_name}  |  Image: {stem} ---")

        image_path = os.path.join(plot_dir, 'images', stem + '.png')
        pred_pt_path = os.path.join(plot_dir, 'bboxes', stem + '.pt')

        result = evaluate_single_image(pred_pt_path, gt_label_path, image_path, iou_threshold)
        if result is None:
            continue

        result['plot_name'] = plot_name
        result['image_stem'] = stem
        per_plot_results.append(result)
        print_single_result(result, iou_threshold)

    if not per_plot_results:
        print("No results computed. Make sure YOLO has been run (bboxes/ folder must exist).")
        return

    print(f"\n{'='*58}")
    print(f"AGGREGATED RESULTS (mean ± std across {len(per_plot_results)} plot(s))")
    print(f"{'='*58}")
    print_aggregated_results(per_plot_results)

    save_results_json(per_plot_results, iou_threshold, CONF_THRESHOLD_GOOD_BOX)
    return per_plot_results



# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    evaluate_all_plots()




