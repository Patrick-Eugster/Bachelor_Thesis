"""
eval_compare_3way.py — SAHI vs YOLO vs GT, head by head (FIP, needs manual_label/).

The plain per-method-vs-GT eval (eval_yolo_boxes.py) matches each method to GT
independently, so it can't show what SAHI ADDS vs BREAKS relative to plain YOLO.
This tool fills that gap using the 3-set Venn of {GT, YOLO, SAHI} boxes (7 regions):

  recall side (GT heads):   1 both found    2 YOLO-only (SAHI regression)
                            3 SAHI rescued  4 neither (hard miss)
  precision side (FP):      5 shared FP     6 YOLO-unique FP    7 SAHI-unique FP

Outputs (under results/mask_generation/{dataset}/evaluation/compare/{eval_experiment}/):
  - compare.json        : 2×2 coverage per size bucket (BOTH tertiles + COCO), split/merge,
                          FP breakdown, per-method count-error.
  - overlay_coverage/   : mixed overlay of regions 1-4 (recall story).
  - overlay_fp/         : mixed overlay of regions 5-7 (precision story).
  - regions/<name>/     : single-region images (only if overlay_mode = singles/both).
  - config.yaml         : the resolved settings of this run.

Run:  python src/mask_generation/evaluation/eval_compare_3way.py
      ... sahi_experiment=sahi_metrics_v1 overlay_mode=both fp_singles=true
"""

import glob
import os
import json
import shutil
import datetime
import yaml
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

# matching primitives + loaders + the eval-experiment name logic — single source of truth,
# imported from the GT eval (no edits to it). compare_common adds the method-vs-method pieces.
from eval_yolo_boxes import (compute_iou_matrix, match_boxes,
                             load_gt_boxes, load_pred_boxes, get_eval_experiment)
import compare_common as cc


# region keys, display labels, and overlay colors (BGR-free plain RGB)
RECALL_REGIONS = ['both', 'yolo_only', 'sahi_rescued', 'neither']   # regions 1,2,3,4
FP_REGIONS     = ['shared', 'yolo_unique', 'sahi_unique']           # regions 5,6,7
LABELS = {
    'both':         'both found',
    'yolo_only':    'YOLO-only (SAHI regression)',
    'sahi_rescued': 'SAHI rescued',
    'neither':      'neither (hard miss)',
    'shared':       'shared FP',
    'yolo_unique':  'YOLO-unique FP',
    'sahi_unique':  'SAHI-unique FP',
}
COLORS = {
    'both':         (40, 200, 40),    # green  — found by both
    'yolo_only':    (255, 160, 0),    # orange — SAHI lost a head YOLO had
    'sahi_rescued': (40, 120, 255),   # blue   — the SAHI value
    'neither':      (220, 30, 30),    # red    — nobody found it
    'shared':       (220, 40, 200),   # magenta— both hallucinated same spot
    'yolo_unique':  (0, 200, 200),    # cyan   — YOLO-only false box
    'sahi_unique':  (240, 220, 0),    # yellow — SAHI-only false box (usually seam dupe)
}
# regions worth isolating as single-color images; 6 & 7 only when fp_singles=true (see config)
SINGLE_DEFAULT   = ['yolo_only', 'sahi_rescued', 'neither', 'shared']   # regions 2,3,4,5
FP_SINGLE_EXTRA  = ['yolo_unique', 'sahi_unique']                       # regions 6,7


# =====================================================================
# Finding labeled images + box paths
# =====================================================================

def find_labeled_images(cfg):
    """Collect every GT-labeled image as (input_plot_dir, plot_name, gt_label_path, stem).
    Uses dataset.plot_glob so it works for FIP (plot_461/) — phone has no manual_label so
    this tool is FIP-only by construction (nogt covers phone)."""
    out = []
    plot_dirs = sorted(glob.glob(os.path.join(cfg.dataset.input_dir, cfg.dataset.plot_glob)))
    for input_plot_dir in plot_dirs:
        plot_name = os.path.relpath(input_plot_dir, cfg.dataset.input_dir)
        label_dir = os.path.join(input_plot_dir, 'manual_label')
        if not os.path.isdir(label_dir):
            continue
        for fname in sorted(os.listdir(label_dir)):
            if fname.endswith('.txt'):
                stem = os.path.splitext(fname)[0]
                out.append((input_plot_dir, plot_name, os.path.join(label_dir, fname), stem))
    return out


def bbox_path(cfg, method, experiment, plot_name, stem):
    """Path to a method's 4-col boxes (.pt) for one image — these are the final good boxes
    that actually go to SAM, i.e. what we want to compare."""
    return os.path.join(cfg.dataset.result_dir_masks, plot_name, method, experiment, 'bboxes', stem + '.pt')


# =====================================================================
# Core categorization (the 7-region Venn for one image)
# =====================================================================

def split_merge_counts(preds, gt, iou_threshold):
    """Count split/merge events between a method's boxes and GT (NOT greedy — looks at the
    full IoU matrix). split = one GT head covered by >=2 pred boxes (seam cut into halves);
    merge = one pred box covering >=2 GT heads (two heads fused). Returns (n_split, n_merge)."""
    iou = compute_iou_matrix(np.asarray(preds).reshape(-1, 4), np.asarray(gt).reshape(-1, 4))
    if iou.size == 0:
        return 0, 0
    over = iou >= iou_threshold              # boolean (Npred, Ngt)
    n_split = int((over.sum(axis=0) >= 2).sum())   # GT columns hit by >=2 preds
    n_merge = int((over.sum(axis=1) >= 2).sum())   # pred rows hitting >=2 GTs
    return n_split, n_merge


def categorize_3way(gt, yolo, sahi, iou_threshold):
    """Split one image's heads/boxes into the 7 Venn regions. Returns recall-side GT-index
    lists (regions 1-4, for the coverage table) and precision-side FP box arrays (regions 5-7),
    plus split/merge and raw counts. GT indices (not boxes) on the recall side so the caller
    can bucket them by size with the dataset-global cutoffs."""
    gt   = np.asarray(gt,   dtype=np.float32).reshape(-1, 4)
    yolo = np.asarray(yolo, dtype=np.float32).reshape(-1, 4)
    sahi = np.asarray(sahi, dtype=np.float32).reshape(-1, 4)

    # match each method to GT independently (greedy IoU>=thr, same as the GT eval)
    y_tp, y_fp_idx, _ = match_boxes(compute_iou_matrix(yolo, gt), iou_threshold)
    s_tp, s_fp_idx, _ = match_boxes(compute_iou_matrix(sahi, gt), iou_threshold)
    yolo_hit = {g for _, g, _ in y_tp}    # GT indices YOLO found
    sahi_hit = {g for _, g, _ in s_tp}    # GT indices SAHI found

    # recall side: bucket each GT head by (yolo_hit, sahi_hit)
    recall_idx = {'both': [], 'yolo_only': [], 'sahi_rescued': [], 'neither': []}
    for g in range(len(gt)):
        in_y, in_s = g in yolo_hit, g in sahi_hit
        if in_y and in_s:       recall_idx['both'].append(g)
        elif in_y and not in_s: recall_idx['yolo_only'].append(g)       # SAHI regression
        elif in_s and not in_y: recall_idx['sahi_rescued'].append(g)    # the SAHI value
        else:                   recall_idx['neither'].append(g)         # hard miss

    # precision side: cross-match the two FP sets → shared (both) vs method-unique
    yolo_fp = yolo[y_fp_idx] if y_fp_idx else np.zeros((0, 4), np.float32)
    sahi_fp = sahi[s_fp_idx] if s_fp_idx else np.zeros((0, 4), np.float32)
    cross = cc.categorize_two_sets(yolo_fp, sahi_fp, iou_threshold)
    fp_boxes = {
        'shared':      yolo_fp[[a for a, _, _ in cross['mutual']]] if cross['mutual'] else np.zeros((0, 4), np.float32),
        'yolo_unique': yolo_fp[cross['a_only']] if cross['a_only'] else np.zeros((0, 4), np.float32),
        'sahi_unique': sahi_fp[cross['b_only']] if cross['b_only'] else np.zeros((0, 4), np.float32),
    }

    return {
        'recall_idx': recall_idx,
        'fp_boxes': fp_boxes,
        'split_merge': {
            'yolo': split_merge_counts(yolo, gt, iou_threshold),
            'sahi': split_merge_counts(sahi, gt, iou_threshold),
        },
        'fp_counts': {
            'yolo_total': len(y_fp_idx), 'sahi_total': len(s_fp_idx),
            'shared': len(fp_boxes['shared']),
            'yolo_unique': len(fp_boxes['yolo_unique']),
            'sahi_unique': len(fp_boxes['sahi_unique']),
        },
        'counts': {'gt': len(gt), 'yolo': len(yolo), 'sahi': len(sahi)},
    }


# =====================================================================
# Size buckets (BOTH tertiles + COCO)
# =====================================================================

def box_areas(boxes):
    """Pixel area of each box [x1,y1,x2,y2]."""
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) if len(b) else np.zeros((0,), np.float32)


def tertile_cutoffs(all_areas):
    """33rd/66th percentile of the pooled GT areas → 3 dataset-relative buckets. Falls back to
    (0,0) if there are no areas (degenerate)."""
    if len(all_areas) == 0:
        return 0.0, 0.0
    return tuple(float(x) for x in np.percentile(all_areas, [100/3, 200/3]))


def bucket_of_tertile(area, cutoffs):
    """small if <= 33rd pct, medium if <= 66th pct, else large."""
    q33, q66 = cutoffs
    return 'small' if area <= q33 else ('medium' if area <= q66 else 'large')


def bucket_of_coco(area):
    """COCO fixed thresholds: small <32²=1024 px², medium <96²=9216, large otherwise."""
    return 'small' if area < 1024 else ('medium' if area < 9216 else 'large')


def empty_coverage():
    """Fresh {bucket: {region: 0}} accumulator for a 2×2 coverage table."""
    return {b: {r: 0 for r in RECALL_REGIONS} for b in ['small', 'medium', 'large']}


# =====================================================================
# Drawing overlays
# =====================================================================

def draw_image_overlays(image_path, gt, cat, dirs, name_tag, overlay_mode, fp_singles):
    """Write the Coverage + FP mixed overlays and/or the per-region single images for one image."""
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, 4)
    recall_boxes = {n: gt[cat['recall_idx'][n]] if cat['recall_idx'][n] else np.zeros((0, 4), np.float32)
                    for n in RECALL_REGIONS}
    fp_boxes = cat['fp_boxes']

    if overlay_mode in ('themed', 'both'):
        # bulk region first so the rare/interesting ones draw on top
        cov_layers = {LABELS[n]: (COLORS[n], recall_boxes[n]) for n in RECALL_REGIONS}
        cc.draw_overlay(image_path, cov_layers, os.path.join(dirs['coverage'], name_tag + '.jpg'))
        fp_layers = {LABELS[n]: (COLORS[n], fp_boxes[n]) for n in FP_REGIONS}
        cc.draw_overlay(image_path, fp_layers, os.path.join(dirs['fp'], name_tag + '.jpg'))

    if overlay_mode in ('singles', 'both'):
        singles = list(SINGLE_DEFAULT) + (FP_SINGLE_EXTRA if fp_singles else [])
        for n in singles:
            boxes = recall_boxes[n] if n in RECALL_REGIONS else fp_boxes[n]
            region_dir = os.path.join(dirs['regions'], n)
            os.makedirs(region_dir, exist_ok=True)
            cc.draw_overlay(image_path, {LABELS[n]: (COLORS[n], boxes)},
                            os.path.join(region_dir, name_tag + '.jpg'))


# =====================================================================
# Printing
# =====================================================================

def print_coverage_table(title, coverage):
    """Print one 2×2-per-bucket coverage table with per-method recall so the small-bucket
    SAHI gain is visible at a glance."""
    print(f"\n{title}")
    print(f"  {'bucket':<8} {'GT':>5} {'both':>6} {'sahi_resc':>10} {'yolo_only':>10} {'neither':>8}"
          f"   | {'R_yolo':>7} {'R_sahi':>7}")
    for b in ['small', 'medium', 'large']:
        c = coverage[b]
        gt = c['both'] + c['sahi_rescued'] + c['yolo_only'] + c['neither']
        r_yolo = (c['both'] + c['yolo_only']) / gt if gt else 0.0   # heads YOLO found / all
        r_sahi = (c['both'] + c['sahi_rescued']) / gt if gt else 0.0
        print(f"  {b:<8} {gt:>5} {c['both']:>6} {c['sahi_rescued']:>10} {c['yolo_only']:>10} "
              f"{c['neither']:>8}   | {r_yolo:>7.3f} {r_sahi:>7.3f}")


# =====================================================================
# Main eval
# =====================================================================

def evaluate(cfg):
    """Run the full 3-way comparison over all labeled FIP images, draw overlays, print the
    coverage tables, and write compare.json + config.yaml."""
    iou_thr = cfg.matching_iou_threshold
    eval_dir = os.path.join(cfg.dataset.result_dir_masks, "evaluation", "compare", get_eval_experiment(cfg))
    dirs = {
        'coverage': os.path.join(eval_dir, 'overlay_coverage'),
        'fp':       os.path.join(eval_dir, 'overlay_fp'),
        'regions':  os.path.join(eval_dir, 'regions'),
    }

    labeled = find_labeled_images(cfg)
    if not labeled:
        print("No labeled images found (expected input_plots/<cam>/<plot>/manual_label/<name>.txt).")
        return

    # wipe + recreate output folders so they only hold this run's images
    for d in [eval_dir, dirs['coverage'], dirs['fp'], dirs['regions']]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    print(f"\n{'=' * 64}")
    print(" SAHI vs YOLO vs GT — 3-way comparison")
    print(f"{'=' * 64}")
    print(f" YOLO boxes:  {cfg.yolo_method}/{cfg.yolo_experiment}")
    print(f" SAHI boxes:  {cfg.sahi_method}/{cfg.sahi_experiment}")
    print(f" Match IoU:   {iou_thr}   overlay_mode: {cfg.overlay_mode}")
    print(f"{'=' * 64}")

    # ---- pass 1: load + categorize every image (store for the second, bucketed pass) ----
    per_image = []
    all_gt_areas = []
    for input_plot_dir, plot_name, gt_label, stem in labeled:
        plot_safe = plot_name.replace(os.sep, '_')
        image_path = os.path.join(input_plot_dir, 'images', stem + '.png')
        y_path = bbox_path(cfg, cfg.yolo_method, cfg.yolo_experiment, plot_name, stem)
        s_path = bbox_path(cfg, cfg.sahi_method, cfg.sahi_experiment, plot_name, stem)
        if not (os.path.exists(y_path) and os.path.exists(s_path) and os.path.exists(image_path)):
            print(f"[SKIP] {plot_name}/{stem}: missing YOLO/SAHI boxes or image.")
            continue

        from PIL import Image
        with Image.open(image_path) as im:
            img_w, img_h = im.size
        gt   = load_gt_boxes(gt_label, img_w, img_h)
        yolo = load_pred_boxes(y_path)
        sahi = load_pred_boxes(s_path)

        cat = categorize_3way(gt, yolo, sahi, iou_thr)
        gt_areas = box_areas(gt)
        all_gt_areas.extend(gt_areas.tolist())

        name_tag = f"{plot_safe}_{stem}"
        draw_image_overlays(image_path, gt, cat, dirs, name_tag, cfg.overlay_mode, cfg.fp_singles)

        per_image.append({'plot': plot_safe, 'stem': stem, 'gt_areas': gt_areas, 'cat': cat})
        print(f"  {plot_name}/{stem}: GT {cat['counts']['gt']}  YOLO {cat['counts']['yolo']}  "
              f"SAHI {cat['counts']['sahi']}  | both {len(cat['recall_idx']['both'])} "
              f"rescued {len(cat['recall_idx']['sahi_rescued'])} yolo_only {len(cat['recall_idx']['yolo_only'])} "
              f"neither {len(cat['recall_idx']['neither'])}")

    if not per_image:
        print("No images had both YOLO and SAHI boxes — nothing to compare.")
        return

    # ---- pass 2: aggregate coverage per size bucket, both tertiles and COCO ----
    all_gt_areas = np.array(all_gt_areas, dtype=np.float32)
    cutoffs = tertile_cutoffs(all_gt_areas)
    cov_tertile, cov_coco = empty_coverage(), empty_coverage()
    for img in per_image:
        areas = img['gt_areas']
        for region in RECALL_REGIONS:
            for g in img['cat']['recall_idx'][region]:
                cov_tertile[bucket_of_tertile(areas[g], cutoffs)][region] += 1
                cov_coco[bucket_of_coco(areas[g])][region] += 1

    # ---- aggregate split/merge, FP, count-error (pooled across images) ----
    agg = {'yolo_split': 0, 'yolo_merge': 0, 'sahi_split': 0, 'sahi_merge': 0}
    fp = {'yolo_total': 0, 'sahi_total': 0, 'shared': 0, 'yolo_unique': 0, 'sahi_unique': 0}
    tot_gt = tot_yolo = tot_sahi = 0
    for img in per_image:
        c = img['cat']
        agg['yolo_split'] += c['split_merge']['yolo'][0]; agg['yolo_merge'] += c['split_merge']['yolo'][1]
        agg['sahi_split'] += c['split_merge']['sahi'][0]; agg['sahi_merge'] += c['split_merge']['sahi'][1]
        for k in fp:
            fp[k] += c['fp_counts'][k]
        tot_gt += c['counts']['gt']; tot_yolo += c['counts']['yolo']; tot_sahi += c['counts']['sahi']

    cer_yolo = (tot_yolo - tot_gt) / tot_gt if tot_gt else None   # >0 = over-counting
    cer_sahi = (tot_sahi - tot_gt) / tot_gt if tot_gt else None

    # ---- print ----
    print(f"\n GT area distribution (px²): min {all_gt_areas.min():.0f}  "
          f"median {np.median(all_gt_areas):.0f}  max {all_gt_areas.max():.0f}  "
          f"| tertile cutoffs {cutoffs[0]:.0f} / {cutoffs[1]:.0f}")
    print_coverage_table("COVERAGE by size — TERTILES (dataset-relative; read tuning off this)", cov_tertile)
    print_coverage_table("COVERAGE by size — COCO (fixed 1024 / 9216 px²; thesis reference)", cov_coco)

    print(f"\n SPLIT / MERGE (vs GT):   YOLO split {agg['yolo_split']}  merge {agg['yolo_merge']}"
          f"   |   SAHI split {agg['sahi_split']}  merge {agg['sahi_merge']}")
    print(f" FALSE POSITIVES:         YOLO {fp['yolo_total']}  SAHI {fp['sahi_total']}"
          f"   | shared {fp['shared']}  YOLO-unique {fp['yolo_unique']}  SAHI-unique {fp['sahi_unique']}")
    print(f" COUNT ERROR (pred-gt)/gt:  YOLO {cer_yolo:+.3f}   SAHI {cer_sahi:+.3f}"
          f"   (GT {tot_gt}  YOLO {tot_yolo}  SAHI {tot_sahi})")

    # ---- write JSON + config ----
    out = {
        'matching_iou_threshold': iou_thr,
        'methods': {'yolo': f"{cfg.yolo_method}/{cfg.yolo_experiment}",
                    'sahi': f"{cfg.sahi_method}/{cfg.sahi_experiment}"},
        'n_images': len(per_image),
        'coverage_by_size': {
            'tertiles': {'cutoffs': list(cutoffs), **cov_tertile},
            'coco': {'cutoffs': [1024, 9216], **cov_coco},
        },
        'split_merge': {'yolo': {'split': agg['yolo_split'], 'merge': agg['yolo_merge']},
                        'sahi': {'split': agg['sahi_split'], 'merge': agg['sahi_merge']}},
        'false_positives': fp,
        'count_error_ratio': {'yolo': cer_yolo, 'sahi': cer_sahi,
                              'totals': {'gt': tot_gt, 'yolo': tot_yolo, 'sahi': tot_sahi}},
        'per_image': [{'plot': i['plot'], 'stem': i['stem'],
                       'counts': i['cat']['counts'],
                       'recall': {r: len(i['cat']['recall_idx'][r]) for r in RECALL_REGIONS},
                       'fp_counts': i['cat']['fp_counts'],
                       'split_merge': i['cat']['split_merge']} for i in per_image],
    }
    with open(os.path.join(eval_dir, 'compare.json'), 'w') as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(eval_dir, 'config.yaml'), 'w') as f:
        yaml.dump({'experiment': get_eval_experiment(cfg),
                   'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                   'dataset': cfg.dataset.name,
                   'yolo': f"{cfg.yolo_method}/{cfg.yolo_experiment}",
                   'sahi': f"{cfg.sahi_method}/{cfg.sahi_experiment}",
                   'matching_iou_threshold': iou_thr,
                   'overlay_mode': cfg.overlay_mode, 'fp_singles': cfg.fp_singles},
                  f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved → {os.path.join(eval_dir, 'compare.json')}\n")
    return out


@hydra.main(version_base=None, config_path="../../../configs/mask_generation", config_name="eval_compare")
def main(cfg: DictConfig):
    evaluate(cfg)


if __name__ == "__main__":
    main()
