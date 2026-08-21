"""Draws the detected marker-plate exclusion mask on each of the 7 FIP GT images, so the plate detection
used by the marker-masked seg evaluation can be checked by eye. The mask is the same plate_mask used there:
near-white pixels (all channels >= 175), morphologically closed, kept as connected blobs >= 40000 px, and
filled so the inner rings count. The masked (excluded) pixels are tinted red over the image. Output goes to
docs/analysis_results/plate_mask_overlay/. Run from repo root."""
import glob
import os
import numpy as np
import cv2
from PIL import Image

PLOTS = [f"plot_{n}" for n in range(461, 468)]
WHITE_MIN, PLATE_MIN_AREA, CLOSE_K = 175, 40000, 25
OUTDIR = "docs/analysis_results/plate_mask_overlay"


def gt_stem(plot):
    return os.path.splitext(os.path.basename(glob.glob(f"input_plots/fip/{plot}/manual_label/*.txt")[0]))[0]


def plate_mask(img):
    """Filled marker-plate mask: large near-white blobs with inner rings filled in."""
    white = np.all(img >= WHITE_MIN, axis=2).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((CLOSE_K, CLOSE_K), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    m = np.zeros(white.shape, np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= PLATE_MIN_AREA:
            c, _ = cv2.findContours((lbl == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(m, c, -1, 1, cv2.FILLED)
    return m.astype(bool)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for plot in PLOTS:
        stem = gt_stem(plot)
        img = np.array(Image.open(f"input_plots/fip/{plot}/images/{stem}.png").convert("RGB"))
        mask = plate_mask(img)
        over = img.copy()
        over[mask] = (0.5 * over[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)  # red tint
        out = f"{OUTDIR}/{plot}_{stem}_platemask.jpg"
        cv2.imwrite(out, cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
        print(f"{plot}: masked {mask.sum():>8} px ({mask.mean()*100:4.1f}% of frame) -> {out}")
    print(f"\nred = excluded plate pixels | full path: {os.path.abspath(OUTDIR)}")


if __name__ == "__main__":
    main()
