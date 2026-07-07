#!/usr/bin/env python3
"""Seed the ground-truth labeling with model boxes so you correct instead of draw from scratch.

For each selected GT image we take the UNION of the YOLO and SAHI predicted boxes, dedup the
heavy overlap, restrict to the plot ROI, and write them as a normalized YOLO <stem>.txt next to
the greyed image in gt_labeling/. Load that folder into the annotation tool and just fix things:
delete false boxes, tighten loose ones, add anything both models missed.

UNION (not SAHI-alone) on purpose: a head SAHI missed but YOLO caught is already seeded, so
neither model's recall is favored by the seed origin — keeps the YOLO-vs-SAHI eval fair. This is
model-assisted GT; disclose it as such.

Usage:
    python src/analysis/make_gt_box_seeds.py
    python src/analysis/make_gt_box_seeds.py --methods sahi_yolo_sam --iou_dedup 0.5
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mask_generation"))
import roi_mask  # noqa: E402

PHONE_ROOT = "/workspace/input_plots/phone"
RESULTS_ROOT = "/workspace/results/mask_generation/phone"
DEFAULT_SELECTION = "/workspace/input_plots/phone/gt_selection.json"  # 36 GT images (12 sessions × 3)


def iou_vec(a, boxes):
    """IoU of one xyxy box against an (N,4) array -> (N,)."""
    x1 = np.maximum(a[0], boxes[:, 0]); y1 = np.maximum(a[1], boxes[:, 1])
    x2 = np.minimum(a[2], boxes[:, 2]); y2 = np.minimum(a[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = aa + ab - inter
    return np.where(union > 0, inter / union, 0.0)


def dedup(boxes, thr):
    """Greedy: keep boxes largest-first, drop any that overlap a kept box by > thr."""
    if len(boxes) == 0:
        return boxes
    order = np.argsort(-((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
    kept = []
    for i in order:
        if not kept or iou_vec(boxes[i], boxes[kept]).max() <= thr:
            kept.append(i)
    return boxes[kept]


def load_boxes(method, session, stem, mask_gen_exp):
    """Load xyxy pixel boxes from one method's bboxes/<stem>.pt, or empty."""
    p = os.path.join(RESULTS_ROOT, session, method, mask_gen_exp, "bboxes", stem + ".pt")
    if not os.path.isfile(p):
        return np.zeros((0, 4), dtype=np.float32)
    t = torch.load(p, weights_only=True)
    if t.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return t[:, :4].numpy().astype(np.float32)


def image_wh(session, stem):
    """(w,h) of the undistorted image, and its path."""
    hits = glob.glob(os.path.join(PHONE_ROOT, session, "images", stem + ".*"))
    if not hits:
        return None, None, None
    import cv2
    img = cv2.imread(hits[0])
    if img is None:
        return None, None, None
    h, w = img.shape[:2]
    return w, h, hits[0]


def roi_cfg():
    """Same ROI block as the greyed images — used to drop seed boxes outside the plot polygon."""
    return {"roi": {"enabled": True, "source": "markers", "min_markers": 3,
                    "buffer_frac": 0.05, "buffer_px": 0, "fallback": "none",
                    "filter_boxes": True, "filter_mode": "overlap", "filter_tol_px": 0.0}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=DEFAULT_SELECTION)
    ap.add_argument("--methods", nargs="+", default=["yolo_sam_v1", "sahi_yolo_sam"])
    ap.add_argument("--mask_gen_experiment", default="initial")
    ap.add_argument("--iou_dedup", type=float, default=0.6)
    ap.add_argument("--out_subdir", default="gt_labeling")
    args = ap.parse_args()

    with open(args.selection) as f:
        selection = json.load(f)
    cfg = roi_cfg()
    total = 0

    for session, picks in selection.items():
        out_dir = os.path.join(PHONE_ROOT, session, args.out_subdir)
        os.makedirs(out_dir, exist_ok=True)
        # yolo-mark wants a class-name list; single class = wheat head
        with open(os.path.join(out_dir, "classes.txt"), "w") as cf:
            cf.write("wheat_head\n")
        print(f"\n=== {session} ===")
        for p in picks:
            stem = p["stem"]
            w, h, img_path = image_wh(session, stem)
            if w is None:
                print(f"   MISSING image  {stem}")
                continue
            per = {m: load_boxes(m, session, stem, args.mask_gen_experiment) for m in args.methods}
            union = np.concatenate([per[m] for m in args.methods], axis=0) if per else np.zeros((0, 4))
            merged = dedup(union, args.iou_dedup)
            # keep only boxes overlapping the plot polygon (consistent with the greyed image)
            if len(merged):
                keep = roi_mask.roi_keep_mask(merged, img_path, cfg, img_w=w, img_h=h)
                merged = merged[keep]
            # write normalized YOLO txt
            lines = []
            for x1, y1, x2, y2 in merged:
                cx = (x1 + x2) / 2 / w; cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w;      bh = (y2 - y1) / h
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            with open(os.path.join(out_dir, stem + ".txt"), "w") as tf:
                tf.write("\n".join(lines) + ("\n" if lines else ""))
            counts = "  ".join(f"{m.split('_')[0]}={len(per[m])}" for m in args.methods)
            print(f"   {stem}:  {counts}  union={len(union)} -> dedup+ROI={len(merged)} seed boxes")
            total += len(merged)

    print(f"\nWrote seed .txt for all GT images ({total} seed boxes total) into each '{args.out_subdir}/'.")
    print("Load that folder in the tool, then CORRECT: delete false boxes, tighten, add any misses.")


if __name__ == "__main__":
    main()
