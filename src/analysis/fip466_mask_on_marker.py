"""Checks, for plot_466 cam_12, which SAM instance masks actually cover the calibration-marker plates,
comparing the YOLOv5-1280 run (paints the markers as FP in 3D) against the YOLO11 run (leaves them
clean). The lift to 3D uses SAM masks, not raw YOLO boxes, so a mask from a box next to the plate can
still bleed onto the bright plate. We locate the plates as filled near-white blobs, then for every
per-instance mask PNG report how much of the plate it covers. Run from repo root."""
import glob
import os
import numpy as np
import cv2
from PIL import Image

STEM = "FPWW036_SR0466_FIP2_cam_12"
IMG = f"input_plots/fip/plot_466/images/{STEM}.png"
RUNS = {
    "yv5_1280": "results/mask_generation/fip/plot_466/yolo_sam_v1/fipseg_yv5_1280_c35/masks",
    "yolo11":   "results/mask_generation/fip/plot_466/yolo11_sam/fipseg_yolo11_c35/masks",
}
WHITE_MIN = 175
PLATE_MIN_AREA = 40000
COVER_MIN = 0.05     # report a mask if it covers at least this fraction of the plate


def plate_mask(img):
    """Filled marker-plate mask (large near-white blobs, inner rings filled)."""
    white = np.all(img >= WHITE_MIN, axis=2).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    mask = np.zeros(white.shape, np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= PLATE_MIN_AREA:
            comp = (lbl == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask, cnts, -1, 1, thickness=cv2.FILLED)
    return mask


def main():
    img = np.array(Image.open(IMG).convert("RGB"))
    plate = plate_mask(img).astype(bool)
    plate_px = int(plate.sum())
    print(f"plate pixels in {STEM}: {plate_px}\n")
    for run, mdir in RUNS.items():
        files = sorted(glob.glob(f"{mdir}/{STEM}_*.png"))
        cover_hits = []
        union = np.zeros(plate.shape, bool)
        for f in files:
            m = np.array(Image.open(f).convert("L")) > 127
            if m.shape != plate.shape:
                m = cv2.resize(m.astype(np.uint8), (plate.shape[1], plate.shape[0])) > 0
            inter = np.logical_and(m, plate).sum()
            frac = inter / plate_px if plate_px else 0
            if frac >= COVER_MIN:
                cover_hits.append((os.path.basename(f), frac))
            union |= np.logical_and(m, plate)
        cover_hits.sort(key=lambda x: -x[1])
        print(f"=== {run}: {len(files)} masks total, {len(cover_hits)} cover >= {COVER_MIN:.0%} of plate ===")
        for name, frac in cover_hits[:10]:
            print(f"   {name:<40} covers {frac:6.1%} of plate")
        print(f"   TOTAL plate coverage by all masks (union): {union.sum()/plate_px:6.1%}\n")


if __name__ == "__main__":
    main()
