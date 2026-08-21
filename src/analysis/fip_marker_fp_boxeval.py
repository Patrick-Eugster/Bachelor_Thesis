"""Tests whether the coded calibration markers explain part of YOLO11's box-level edge over YOLOv5 on
FIP. On each plot's single labeled GT image, GT boxes never mark a marker, so any kept detection sitting
on a marker plate is a pure false positive. Using the reliable per-camera marker centers from the
reprocessed marker_projections.csv (verified to land on the plates in our undistorted images), we count,
per detector, how many kept boxes (conf >= 0.35) fall within a plate radius of a marker, and how many GT
boxes do (a sanity check that real heads rarely sit there). Run from repo root."""
import csv
import glob
import os
import numpy as np
import torch

PLOTS = [f"plot_{n}" for n in range(461, 468)]
MARKER_CSV = "demoanlage2025_v0/FIP_single_row_exp_2024_reprocessed/{plot}/marker_projections.csv"
BOX = {  # detector -> bboxes_with_conf dir (xyxy+conf), the same runs behind the FIP box eval
    "YOLOv5-1280": "results/mask_generation/fip/{plot}/yolo_sam_v1/fip_imgsz_1280/bboxes_with_conf",
    "YOLO11":      "results/mask_generation/fip/{plot}/yolo11_sam/yolo11_eval/bboxes_with_conf",
}
KEEP = 0.35
R = 200          # a box/GT center within this many px of a marker center counts as "on the plate"


def gt_stem_and_boxes(plot):
    """Returns the plot's single GT-labeled stem and its GT box centers (pixel xy)."""
    lbl = glob.glob(f"input_plots/fip/{plot}/manual_label/*.txt")[0]
    stem = os.path.splitext(os.path.basename(lbl))[0]
    from PIL import Image
    w, h = Image.open(f"input_plots/fip/{plot}/images/{stem}.png").size
    cx = []
    for line in open(lbl):
        p = line.split()
        if len(p) >= 5:
            cx.append(((float(p[1])) * w, (float(p[2])) * h))
    return stem, np.array(cx) if cx else np.zeros((0, 2))


def marker_centers(plot, stem):
    """Marker centers for this camera from the reprocessed projections CSV."""
    pts = []
    for r in csv.DictReader(open(MARKER_CSV.format(plot=plot))):
        if r["Camera"] == stem:
            pts.append((float(r["X"]), float(r["Y"])))
    return np.array(pts) if pts else np.zeros((0, 2))


def near_marker(centers, markers):
    """Count how many of `centers` fall within R px of any marker center."""
    if len(centers) == 0 or len(markers) == 0:
        return 0
    d = np.sqrt(((centers[:, None, :] - markers[None, :, :]) ** 2).sum(-1))
    return int((d.min(1) <= R).sum())


def main():
    print(f"marker-proximal boxes (center within {R}px of a marker), kept at conf>={KEEP}\n")
    print(f"{'plot':<10} {'#mk':>3} {'gt_on_mk':>8} {'YOLOv5_FP':>10} {'YOLO11_FP':>10}")
    tot = {"YOLOv5-1280": 0, "YOLO11": 0}
    tot_gt = 0
    for plot in PLOTS:
        stem, gtc = gt_stem_and_boxes(plot)
        mk = marker_centers(plot, stem)
        gt_on = near_marker(gtc, mk)
        tot_gt += gt_on
        row = {}
        for det, d in BOX.items():
            f = f"{d.format(plot=plot)}/{stem}.pt"
            if not os.path.exists(f):
                row[det] = -1; continue
            t = torch.load(f, weights_only=True).numpy()
            kept = t[t[:, 4] >= KEEP]
            cen = np.stack([(kept[:, 0] + kept[:, 2]) / 2, (kept[:, 1] + kept[:, 3]) / 2], 1) if len(kept) else np.zeros((0, 2))
            row[det] = near_marker(cen, mk)
            if row[det] >= 0:
                tot[det] += row[det]
        print(f"{plot:<10} {len(mk):>3} {gt_on:>8} {row['YOLOv5-1280']:>10} {row['YOLO11']:>10}")
    print(f"\n{'TOTAL':<10} {'':>3} {tot_gt:>8} {tot['YOLOv5-1280']:>10} {tot['YOLO11']:>10}")


if __name__ == "__main__":
    main()
