"""Finds every YOLOv5-1280 detection on plot_466 that lands on a coded calibration marker, so we can
say exactly which images the marker false-positives come from and at what confidence. Rather than
testing what is inside each box (the detector often fires on the marker's dark navy ring, not the
white plate), we first locate each marker plate as a large near-white blob, then flag any detection
whose center falls inside that plate's region. Confidence comes from the parallel fip_imgsz_1280 run's
bboxes_with_conf (same detector + input size, so the raw detections match the seg-feeding
fipseg_yv5_1280_c35 run). Marks which hits cross the 0.35 keep threshold the seg run used. Run from
repo root."""
import glob
import os
import numpy as np
import torch
import cv2
from PIL import Image

CONF_DIR = "results/mask_generation/fip/plot_466/yolo_sam_v1/fip_imgsz_1280/bboxes_with_conf"
IMG_DIR = "input_plots/fip/plot_466/images"
KEEP_THR = 0.35          # the confidence the seg run (c35) kept boxes at
WHITE_MIN = 175          # a pixel counts as plate-white when R,G,B are all >= this
PLATE_MIN_AREA = 40000   # a marker plate is a large white blob; ignore small white specks
PAD = 10                 # small padding around the plate bbox when testing box centers


def plate_mask(img):
    """Returns a filled plate mask: large near-white blobs with their dark inner rings filled in, so a
    detection sitting on a ring still counts as on-plate while grass in the rotated bbox corners does not."""
    white = np.all(img >= WHITE_MIN, axis=2).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    mask = np.zeros(white.shape, np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= PLATE_MIN_AREA:
            comp = (lbl == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask, cnts, -1, 1, thickness=cv2.FILLED)  # fill so inner rings count
    return mask


def main():
    hits = []
    for f in sorted(glob.glob(f"{CONF_DIR}/*.pt")):
        stem = os.path.splitext(os.path.basename(f))[0]
        img = np.array(Image.open(f"{IMG_DIR}/{stem}.png").convert("RGB"))
        mask = plate_mask(img)
        if mask.sum() == 0:
            continue
        h, w = mask.shape
        boxes = torch.load(f, weights_only=True).numpy()
        for x1, y1, x2, y2, conf in boxes:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if 0 <= cy < h and 0 <= cx < w and mask[cy, cx]:
                hits.append((stem, float(conf), (int(x1), int(y1), int(x2), int(y2))))

    hits.sort(key=lambda h: (h[0], -h[1]))
    kept = [h for h in hits if h[1] >= KEEP_THR]
    print(f"detections whose center is on a marker plate (plate area >= {PLATE_MIN_AREA}px):")
    print(f"{'image':<34} {'conf':>6}  kept?")
    for stem, conf, _ in hits:
        print(f"{stem:<34} {conf:>6.3f}  {'KEPT' if conf >= KEEP_THR else 'dropped'}")
    imgs_kept = sorted({h[0] for h in kept})
    print(f"\ntotal marker-region detections: {len(hits)}  |  KEPT at conf>={KEEP_THR}: {len(kept)}")
    print(f"images with a KEPT marker detection ({len(imgs_kept)}): {imgs_kept}")


if __name__ == "__main__":
    main()
