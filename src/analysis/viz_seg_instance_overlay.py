#!/usr/bin/env python
"""viz_seg_instance_overlay.py — sanity-check the phone 3D-seg instance eval by EYE.

The per-head instance F1 came out far below the pixel foreground score, and we want to see whether the
predicted 3D-seg head regions actually line up with the GT heads or are systematically offset / the wrong
size (which the union pixel-IoU would hide). This draws, for one seg run on the labeled frame:
  - LEFT / full frame : GT foreground tinted RED and predicted foreground tinted BLUE over the dimmed
    photo. Where they agree the head shows MAGENTA; a red-only or blue-only fringe = a mismatch/offset.
  - a ZOOM crop with per-instance OUTLINES (GT red, predicted cyan) so per-head alignment is visible.

READ-ONLY: reads results/ + input_plots/ only; writes ONLY PNG(s) under docs/analysis_results/.

    python src/analysis/viz_seg_instance_overlay.py                       # default run ocv15k_sam1_conf070
    python src/analysis/viz_seg_instance_overlay.py --run ocv15k_conf070
"""
import os
import sys
import argparse

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phone_seg_cpu_eval import RUNS, BASE, S  # noqa: E402
from phone_seg_instance_eval import load_gt_instances, VARIANT_SUBDIR, nn_resize_labels  # noqa: E402

OUTDIR = "docs/analysis_results/seg_instance_overlay"


def boundaries(m):
    """1-px instance outlines: pixels where the label differs from a neighbour and a head is involved."""
    e = np.zeros(m.shape, bool)
    e[:, :-1] |= m[:, :-1] != m[:, 1:]
    e[:-1, :] |= m[:-1, :] != m[1:, :]
    fg = m > 0
    near_fg = fg.copy()
    near_fg[:, :-1] |= fg[:, 1:]
    near_fg[:-1, :] |= fg[1:, :]
    return e & near_fg


def find_run(name):
    for r in RUNS:
        if r[0] == name:
            return r
    raise SystemExit(f"run '{name}' not in RUNS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ocv15k_sam1_conf070")
    ap.add_argument("--zoom", default="1500,1100,900", help="cx,cy,size of the zoom crop in GT pixels")
    args = ap.parse_args()

    name, mp, exp, gtp, stem, tag, sfm, iters = find_run(args.run)

    # --- load GT instance map (authoritative, via manifest) ---
    sub = VARIANT_SUBDIR.get(sfm, "")
    label_dir = os.path.join("input_plots", "phone", S, sub, "manual_label") if sub \
        else os.path.join("input_plots", "phone", S, "manual_label")
    gt_map, gt_ids, _ = load_gt_instances(label_dir, stem)
    if gt_map is None:
        raise SystemExit(f"no per-head GT for {name} ({sfm})")

    # --- load predicted 2DSeg label map ---
    import torch
    pt = f"{BASE}/{mp}/segmentation_3d/{exp}/2DSeg/{stem}.pt"
    lab = torch.load(pt, weights_only=True)
    lab = (lab.numpy() if hasattr(lab, "numpy") else np.array(lab)).astype(np.int32)
    if lab.shape != gt_map.shape:
        lab = nn_resize_labels(lab, gt_map.shape)

    # --- background photo (dimmed), matched to the GT frame ---
    img_path = os.path.join("input_plots", "phone", S, sub, "images", f"{stem}.jpg") if sub \
        else os.path.join("input_plots", "phone", S, "images", f"{stem}.jpg")
    if os.path.exists(img_path):
        bg = cv2.imread(img_path)
        if bg.shape[:2] != gt_map.shape:
            bg = cv2.resize(bg, (gt_map.shape[1], gt_map.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        bg = np.full((*gt_map.shape, 3), 60, np.uint8)
    base = (bg.astype(np.float32) * 0.45).astype(np.uint8)

    gt_fg = gt_map > 0
    pred_fg = lab > 0

    # --- full-frame filled overlay: GT red, pred blue, agreement magenta ---
    full = base.copy()
    tint = full.astype(np.float32)
    tint[gt_fg]   = 0.45 * tint[gt_fg]   + 0.55 * np.array([40, 40, 220], np.float32)   # red   (BGR)
    tint[pred_fg] = 0.55 * tint[pred_fg] + 0.45 * np.array([220, 80, 40], np.float32)    # blue  (BGR)
    full = tint.astype(np.uint8)
    cv2.putText(full, f"{name}: GT=red  pred=blue  overlap=magenta  (GT {len(gt_ids)} heads, pred {int(lab.max())} ids)",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    os.makedirs(OUTDIR, exist_ok=True)
    scale = 1600 / full.shape[1]
    full_s = cv2.resize(full, (1600, int(full.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    p_full = os.path.join(OUTDIR, f"{name}_overlay_full.png")
    cv2.imwrite(p_full, full_s)

    # --- zoom crop with per-instance OUTLINES ---
    cx, cy, sz = (int(v) for v in args.zoom.split(","))
    H, W = gt_map.shape
    x0, y0 = max(0, cx - sz // 2), max(0, cy - sz // 2)
    x1, y1 = min(W, x0 + sz), min(H, y0 + sz)
    crop = base[y0:y1, x0:x1].copy()
    gt_b = boundaries(gt_map[y0:y1, x0:x1])
    pr_b = boundaries(lab[y0:y1, x0:x1])
    gt_b = cv2.dilate(gt_b.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
    pr_b = cv2.dilate(pr_b.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
    crop[gt_b] = (40, 40, 240)     # red   GT outline
    crop[pr_b] = (240, 240, 40)    # cyan  pred outline
    cv2.putText(crop, "GT=red  pred=cyan", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    p_zoom = os.path.join(OUTDIR, f"{name}_overlay_zoom.png")
    cv2.imwrite(p_zoom, crop)

    print(f"GT heads: {len(gt_ids)}   pred ids: {int(lab.max())}   frame: {gt_map.shape}")
    print(f"wrote {p_full}")
    print(f"wrote {p_zoom}   (zoom {x0}:{x1},{y0}:{y1})")
    print("(read-only: nothing under results/ or input_plots/ was modified)")


if __name__ == "__main__":
    main()
