"""Recomputes the plot_466-style 2D seg precision with the calibration-marker plates excluded, fairly and
uniformly across all 21 FIP bridge runs (7 plots x 3 detector mask sets). Markers are not wheat and the GT
never labels them, so a 3D head projected onto a plate is a pure false positive that says nothing about
wheat segmentation. For each run we rebuild the binary prediction from the per-variant 2DSeg label map
(pred = any head id > 0), rebuild binary GT from the manual mask, and locate the plates as filled
near-white blobs; we then report precision/IoU with and without the plate pixels. As a self-check we first
confirm the no-exclusion precision matches the stored metrics_2d.json. Run from repo root."""
import csv
import glob
import json
import os
import numpy as np
import cv2
import torch
from PIL import Image

PLOTS = [f"plot_{n}" for n in range(461, 468)]
DETS = ["seg_yv5_640", "seg_yv5_1280", "seg_yolo11"]
SEG = "results/reconstruction/fip/{p}/vanilla_3dgs/fipseg15k_pp/segmentation_3d/{d}"
WHITE_MIN = 175
PLATE_MIN_AREA = 40000


def gt_stem(plot):
    return os.path.splitext(os.path.basename(glob.glob(f"input_plots/fip/{plot}/manual_label/*.txt")[0]))[0]


def plate_mask(img):
    """Filled marker-plate mask (large near-white blobs, inner rings filled)."""
    white = np.all(img >= WHITE_MIN, axis=2).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    m = np.zeros(white.shape, np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= PLATE_MIN_AREA:
            comp = (lbl == i).astype(np.uint8)
            c, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(m, c, -1, 1, cv2.FILLED)
    return m.astype(bool)


def prec_iou(gt, pred, valid):
    """Precision and IoU over the valid (non-excluded) pixels."""
    g, p = gt & valid, pred & valid
    tp = int((g & p).sum()); fp = int((~g & p).sum()); fn = int((g & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return prec, iou


def main():
    print(f"{'plot':<8}{'detector':<14}{'prec_stored':>11}{'prec_recomp':>12}{'prec_noMark':>12}{'iou_recomp':>11}{'iou_noMark':>11}")
    for plot in PLOTS:
        stem = gt_stem(plot)
        raw = np.array(Image.open(f"input_plots/fip/{plot}/images/{stem}.png").convert("RGB"))
        gt = np.array(Image.open(f"input_plots/fip/{plot}/manual_label/{stem}_gt_mask.png").convert("L")) >= 128
        plates = plate_mask(raw)
        valid_all = np.ones(gt.shape, bool)
        valid_nomark = ~plates
        for d in DETS:
            lb = torch.load(f"{SEG.format(p=plot, d=d)}/2DSeg/{stem}.pt", weights_only=True)
            lb = lb.numpy() if hasattr(lb, "numpy") else np.array(lb)
            if lb.shape != gt.shape:
                lb = cv2.resize(lb.astype(np.int32), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred = lb > 0
            stored = json.load(open(f"{SEG.format(p=plot, d=d)}/eval_2d/metrics_2d.json"))[0]["precision"]
            p_re, i_re = prec_iou(gt, pred, valid_all)
            p_nm, i_nm = prec_iou(gt, pred, valid_nomark)
            print(f"{plot:<8}{d:<14}{stored:>11.3f}{p_re:>12.3f}{p_nm:>12.3f}{i_re:>11.3f}{i_nm:>11.3f}")


if __name__ == "__main__":
    main()
