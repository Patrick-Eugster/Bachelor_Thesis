#!/usr/bin/env python3
"""Offline correctness check for the segmentation_3d crop-cache speedup.

Proves that the cropped IoU (calculate_seg_iou_gpu_crop, used by find_match after the
crop-cache change) returns EXACTLY the same value as the original full-frame IoU
(calculate_seg_iou_gpu) on real SAM masks — without running the (very slow) 3D
segmentation loop, without a GPU, and without a trained model. Runs on CPU in seconds.

If this passes, the crop cache is byte-identical to the old behaviour and only the
runtime changes. See docs/segmentation_3d/SEGMENTATION_3D_RUNTIME.md.

Usage:
    python src/analysis/verify_crop_iou.py                 # default masks dir
    python src/analysis/verify_crop_iou.py --masks_dir <dir> --n 60
"""
import os
import sys
import glob
import random
import argparse

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gaussians.utils.wheatgs_utils import (
    build_mask_crop,
    calculate_seg_iou_gpu,
    calculate_seg_iou_gpu_crop,
)

DEFAULT_MASKS = "results/mask_generation/phone/field_A/20250715/sahi_yolo_sam/initial/masks"


def load_mask(path):
    """Decode one mask PNG to a full-frame CPU bool tensor (H, W)."""
    m = np.array(Image.open(path).convert("L")) > 127
    return torch.from_numpy(m)  # bool tensor on CPU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks_dir", default=DEFAULT_MASKS)
    ap.add_argument("--n", type=int, default=50, help="how many masks to load")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fs = sorted(glob.glob(os.path.join(args.masks_dir, "*.png")))
    if not fs:
        print(f"No masks found in {args.masks_dir}")
        sys.exit(1)
    random.seed(args.seed)
    sample = random.sample(fs, min(args.n, len(fs)))

    full = [load_mask(p) for p in sample]          # full-frame bool masks (candidates)
    crops = [build_mask_crop(m) for m in full]     # tight-bbox crop entries (or None if empty)

    # Use every mask in turn as the "rendered blob" (pred_seg) and IoU it against all
    # candidates both ways. Real masks make realistic partial-overlap / disjoint cases.
    n_cmp = 0
    max_diff = 0.0
    mismatches = 0
    for pred in full:
        pred_area = pred.sum()
        for cand_full, entry in zip(full, crops):
            iou_full = calculate_seg_iou_gpu(cand_full, pred)
            if entry is None:
                # empty candidate: original IoU must be 0 so find_match's "skip empty" is equivalent
                iou_crop = 0.0
            else:
                iou_crop = calculate_seg_iou_gpu_crop(pred, pred_area, entry)
            n_cmp += 1
            d = abs(iou_full - iou_crop)
            max_diff = max(max_diff, d)
            if iou_full != iou_crop:
                mismatches += 1

    print(f"masks loaded:     {len(full)}  (from {args.masks_dir})")
    print(f"empty masks:      {sum(1 for e in crops if e is None)}")
    print(f"comparisons:      {n_cmp}")
    print(f"max |diff|:       {max_diff}")
    print(f"exact mismatches: {mismatches}")
    if mismatches == 0:
        print("\nPASS — cropped IoU is bit-identical to full-frame IoU on every pair.")
        sys.exit(0)
    else:
        print("\nFAIL — cropped IoU differs from full-frame IoU. Do NOT trust the crop cache.")
        sys.exit(1)


if __name__ == "__main__":
    main()
