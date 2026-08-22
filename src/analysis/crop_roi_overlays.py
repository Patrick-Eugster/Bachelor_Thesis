"""Crop the plain SAM overlays for the six phone GT sessions to their ROI content, so the big grey
margin outside the marker hull is removed and the heads fill the frame on the thesis page. Each source
overlay (YOLOv5 @4032, per-tile, SAM2) has the area outside the plot ROI painted flat grey (114,114,114);
we take the bounding box of everything that is not that grey, pad it a little, and save the crop. Reads
the copies under docs/analysis_results/ and writes the crops into thesis/figures/, touching nothing in
results/.

    python src/analysis/crop_roi_overlays.py
"""
import os
import glob

import cv2
import numpy as np

SRC = "docs/analysis_results/maskgen_phone_pertile_sam2_allsessions"
DST = "thesis/figures/maskgen_phone_pertile_allsessions"
GREY = 114        # the ROI-outside fill value (B=G=R=114)
TOL = 15          # a pixel counts as content if any channel is more than TOL off the grey
PAD = 25          # padding in pixels around the content bounding box
MAXW = 1500       # cap the crop width for the thesis (a half-text-width panel needs no more); keeps the PDF small
JPEG_Q = 88       # jpeg quality for the saved crop


def crop_one(path):
    """Return the image cropped to its non-grey (ROI) content (padded), then downscaled so its width is
    at most MAXW --- a half-page panel does not need the full 3000-pixel frame, and the cap keeps the
    figure from bloating the PDF."""
    im = cv2.imread(path)
    content = (np.abs(im.astype(np.int16) - GREY) > TOL).any(axis=2)
    ys, xs = np.where(content)
    y0, y1 = max(0, ys.min() - PAD), min(im.shape[0], ys.max() + PAD)
    x0, x1 = max(0, xs.min() - PAD), min(im.shape[1], xs.max() + PAD)
    crop = im[y0:y1, x0:x1]
    if crop.shape[1] > MAXW:
        h = round(crop.shape[0] * MAXW / crop.shape[1])
        crop = cv2.resize(crop, (MAXW, h), interpolation=cv2.INTER_AREA)
    return crop


def main():
    os.makedirs(DST, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(SRC, "*.jpg"))):
        # source name is like A_20250715_IMG_20250715_153912.jpg -> keep the A_<date> lead as the crop name
        base = os.path.basename(path)
        fld_date = "_".join(base.split("_")[:2])   # e.g. A_20250715
        out = os.path.join(DST, f"{fld_date}.jpg")
        crop = crop_one(path)
        cv2.imwrite(out, crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        print(f"{base}  ->  {out}   {crop.shape[1]}x{crop.shape[0]}")


if __name__ == "__main__":
    main()
