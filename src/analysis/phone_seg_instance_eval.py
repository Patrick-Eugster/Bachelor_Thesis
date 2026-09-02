#!/usr/bin/env python
"""phone_seg_instance_eval.py — INSTANCE / per-head 2D eval for the phone 3D segmentation.

Companion to phone_seg_cpu_eval.py, which is the PIXEL (binary-foreground) version. Both score the same
runs; this one is the count-based instance metric. Each predicted 3D-seg head (one distinct nonzero id in
the 2DSeg label map) is matched one-to-one against the manual per-head GT instance map
(manual_label/{stem}_sets/setN_instances.png, read via the manifest — the AUTHORITATIVE GT, not the stale
top-level gt_cache one), using the SAME Hungarian matching and merge/split definitions as
src/mask_generation/evaluation/eval_masks_instance.py, so the seg numbers are comparable to the mask-gen
instance tables. Reports head-count precision / recall / F1, matched-mask IoU, and merge/split counts at
IoU>=0.5 (0.25 and 0.75 too). The agisoft frame has NO per-head instance GT (only a warped union mask), so
agisoft runs come out as "no_instance_gt".

READ-ONLY + NON-DESTRUCTIVE. Reads results/ and input_plots/ only. Writes ONLY to
docs/analysis_results/phone_seg_instance_eval.json (a NEW file). It NEVER touches phone_seg_cpu_eval.json
or anything under results/ or input_plots/. Run with --dry-run to compute and print the table while
writing nothing at all.

    python src/analysis/phone_seg_instance_eval.py --dry-run   # validate, write nothing
    python src/analysis/phone_seg_instance_eval.py             # also write the json
"""
import os
import sys
import json
import argparse

import numpy as np
import cv2
import torch
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import find_objects

# reuse the run list + base paths from the pixel scorer (single source of truth; it has a __main__ guard,
# so importing it does not run anything).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phone_seg_cpu_eval import RUNS, BASE, S  # noqa: E402

OUT = "docs/analysis_results/phone_seg_instance_eval.json"
VARIANT_SUBDIR = {"pinhole": "", "opencv": "opencv", "agisoft": "agisoft"}


# ============================================================================================
# Verbatim copies from src/mask_generation/evaluation/eval_masks_instance.py (keep in sync).
# Copied rather than imported so this analysis script stays hydra-free and self-contained.
# ============================================================================================
def load_gt_instances(label_dir, stem):
    """Load one image's GT instance map through the sets manifest (the authoritative path).
    Returns (instance_map int32 HxW, gt_ids, areas_by_id) or (None, None, None) if there's no GT."""
    sets_dir = os.path.join(label_dir, f"{stem}_sets")
    man_path = os.path.join(sets_dir, "manifest.json")
    if not os.path.exists(man_path):
        return None, None, None
    man = json.load(open(man_path))
    entry = next((e for e in man.get("sets", []) if e["name"] == man.get("active")), None)
    if entry is None:                                   # active name missing -> fall back to the first set
        if not man.get("sets"):
            return None, None, None
        entry = man["sets"][0]
        print(f"    WARNING: manifest 'active' set not found for {stem}; using {entry['name']}")
    png = os.path.join(sets_dir, f"{entry['file']}_instances.png")
    m = cv2.imread(png, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None, None, None
    if m.ndim == 3:                                     # be tolerant if it ever got saved as 3-channel
        m = m[..., 0]
    m = m.astype(np.int32)
    areas = np.bincount(m.ravel())
    ids = np.nonzero(areas)[0]
    ids = ids[ids != 0]                                 # drop background
    return m, ids, areas


def compute_iou_matrix(gt_map, gt_ids, gt_areas, preds):
    """IoU of every prediction against every GT head — the instance-map + bincount trick.
    Returns (iou, inter), both (n_pred, n_gt) and aligned to gt_ids."""
    n_pred, n_gt = len(preds), len(gt_ids)
    inter = np.zeros((n_pred, n_gt), np.int64)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), np.float64), inter
    minlen = int(gt_map.max()) + 1
    for i, p in enumerate(preds):
        x0, y0, x1, y1 = p["bbox"]
        ids_under = gt_map[y0:y1, x0:x1][p["sub"]]      # the GT id at each pixel of this prediction
        cnt = np.bincount(ids_under, minlength=minlen)  # cnt[id] = overlap area with that head
        inter[i] = cnt[gt_ids]
    pred_areas = np.array([p["area"] for p in preds], np.int64)
    gt_a = gt_areas[gt_ids].astype(np.int64)
    union = pred_areas[:, None] + gt_a[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return iou, inter


def match_instances(iou, thr):
    """One-to-one matching of predictions to GT heads (Hungarian, maximising total IoU), keeping only
    pairs at/above the threshold."""
    if iou.size == 0:
        return []
    rows, cols = linear_sum_assignment(-iou)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if iou[i, j] >= thr]


def metrics_at(iou, thr, n_pred, n_gt):
    """precision / recall / F1 / mean matched IoU / panoptic quality at one IoU threshold."""
    pairs = match_instances(iou, thr)
    tp = len(pairs)
    fp, fn = n_pred - tp, n_gt - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    ious = [float(iou[i, j]) for i, j in pairs]
    sq = float(np.mean(ious)) if ious else 0.0          # segmentation quality = mean IoU of the matches
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = float(sum(ious) / denom) if denom > 0 else 0.0  # panoptic quality
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
            "mean_iou_matched": sq, "pq": pq}, pairs


def count_merges_splits(inter, preds, gt_areas, gt_ids, merge_frac, split_frac):
    """merge = ONE prediction swallowing >=2 GT heads; split = ONE GT head broken across >=2 predictions."""
    if inter.size == 0:
        return {"merge_preds": 0, "gt_heads_merged": 0, "split_gts": 0}
    pred_areas = np.array([p["area"] for p in preds], np.int64)
    gt_a = gt_areas[gt_ids].astype(np.int64)
    frac_of_gt = inter / np.maximum(gt_a[None, :], 1)          # how much of each GT head is in each pred
    frac_of_pred = inter / np.maximum(pred_areas[:, None], 1)  # how much of each pred is in each GT head
    merge_hits = frac_of_gt >= merge_frac
    merge_idx = np.nonzero(merge_hits.sum(axis=1) >= 2)[0]
    gt_merged = int(merge_hits[merge_idx].any(axis=0).sum()) if len(merge_idx) else 0
    split_hits = frac_of_pred >= split_frac
    split_idx = np.nonzero(split_hits.sum(axis=0) >= 2)[0]
    return {"merge_preds": int(len(merge_idx)), "gt_heads_merged": gt_merged,
            "split_gts": int(len(split_idx))}
# ============================================================================================
# end verbatim block
# ============================================================================================


def nn_resize_labels(lab, out_hw):
    """Nearest-neighbour resize of an integer label map (preserves ids; cv2.resize rejects int32)."""
    H, W = lab.shape
    oh, ow = out_hw
    ys = np.clip((np.arange(oh) * H / oh).astype(int), 0, H - 1)
    xs = np.clip((np.arange(ow) * W / ow).astype(int), 0, W - 1)
    return lab[ys][:, xs]


def preds_from_labelmap(lab):
    """Turn a 2DSeg integer label map (one nonzero id per predicted head) into the sparse
    [{bbox, sub, area}] form compute_iou_matrix expects. find_objects gives every id's bbox in one pass."""
    preds = []
    for i, sl in enumerate(find_objects(lab)):
        lid = i + 1
        if sl is None:
            continue
        sub = lab[sl] == lid
        if not sub.any():
            continue
        y0, x0 = sl[0].start, sl[1].start
        y1, x1 = sl[0].stop, sl[1].stop
        preds.append({"bbox": (x0, y0, x1, y1),
                      "sub": np.ascontiguousarray(sub),
                      "area": int(sub.sum())})
    return preds


def score_run(mp, exp, stem, sfm, sess=S):
    """Instance metrics for one seg run, or a status string if its pred / GT is unavailable.
    sess = the run's session (defaults to the A/0715 anchor; generalization runs pass their own)."""
    base = f"results/reconstruction/phone/{sess}"
    pt = f"{base}/{mp}/segmentation_3d/{exp}/2DSeg/{stem}.pt"
    if not os.path.exists(pt):
        return {"status": "missing_pred"}
    sub = VARIANT_SUBDIR.get(sfm, "")
    label_dir = os.path.join("input_plots", "phone", sess, sub, "manual_label") if sub \
        else os.path.join("input_plots", "phone", sess, "manual_label")
    gt_map, gt_ids, gt_areas = load_gt_instances(label_dir, stem)
    if gt_map is None:
        return {"status": "no_instance_gt"}     # e.g. agisoft frame — only a warped union mask exists
    lab = torch.load(pt, weights_only=True)
    lab = lab.numpy() if hasattr(lab, "numpy") else np.array(lab)
    lab = lab.astype(np.int32)
    resized = lab.shape != gt_map.shape
    if resized:
        lab = nn_resize_labels(lab, gt_map.shape)
    preds = preds_from_labelmap(lab)
    n_pred, n_gt = len(preds), len(gt_ids)
    iou, inter = compute_iou_matrix(gt_map, gt_ids, gt_areas, preds)
    m50, _ = metrics_at(iou, 0.5, n_pred, n_gt)
    m25, _ = metrics_at(iou, 0.25, n_pred, n_gt)
    m75, _ = metrics_at(iou, 0.75, n_pred, n_gt)
    ms = count_merges_splits(inter, preds, gt_areas, gt_ids, 0.5, 0.5)
    return {"status": "ok", "n_pred": n_pred, "n_gt": n_gt, "resized_pred": bool(resized),
            "precision": round(m50["precision"], 4), "recall": round(m50["recall"], 4),
            "f1": round(m50["f1"], 4), "iou_matched": round(m50["mean_iou_matched"], 4),
            "f1_iou25": round(m25["f1"], 4), "f1_iou75": round(m75["f1"], 4),
            "merge_preds": ms["merge_preds"], "split_gts": ms["split_gts"],
            "gt_heads_merged": ms["gt_heads_merged"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print the table but DO NOT write the json (writes nothing at all)")
    args = ap.parse_args()

    rows = []
    for run in RUNS:
        name, mp, exp, gtp, stem, tag, sfm, iters = run[:8]
        sess = run[8] if len(run) > 8 else S      # optional per-run session override
        r = score_run(mp, exp, stem, sfm, sess)
        r.update({"name": name, "sfm": sfm, "iters": iters, "mask": tag})
        rows.append(r)

    # print table
    print(f"{'run':<24}{'sfm':<9}{'iters':>6}  {'P':>6}{'R':>7}{'F1':>7}{'IoUm':>7}"
          f"{'F1@25':>7}{'F1@75':>7}{'mrg':>5}{'spl':>5}{'nPred':>7}{'nGT':>6}   mask")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['name']:<24}{r['sfm']:<9}       -- {r['status']}")
            continue
        print(f"{r['name']:<24}{r['sfm']:<9}{r['iters']:>6}  "
              f"{r['precision']:>6.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}{r['iou_matched']:>7.3f}"
              f"{r['f1_iou25']:>7.3f}{r['f1_iou75']:>7.3f}{r['merge_preds']:>5}{r['split_gts']:>5}"
              f"{r['n_pred']:>7}{r['n_gt']:>6}   {r['mask']}")

    if args.dry_run:
        print(f"\n[DRY RUN] validated end-to-end; NOTHING was written. A real run would write: {OUT}")
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"session": S, "metric": "instance / per-head (Hungarian, IoU>=0.5)", "runs": rows},
              open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}   (NEW file; phone_seg_cpu_eval.json, results/ and input_plots/ untouched)")


if __name__ == "__main__":
    main()
