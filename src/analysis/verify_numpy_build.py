#!/usr/bin/env python3
"""Offline regression check for the segmentation_3d crop-cache BUILD fix.

The cache build (`_decode_crop` in run_3d_seg.py) was rewritten from a torch path
(PILtoTorch -> full-frame torch bool -> mask[y0:y1,x0:x1].cpu().numpy().copy()) to a
PURE-NUMPY path, because the torch intermediate is a numpy VIEW whose `.base` pinned the
whole 12 MB frame -> a ~6.5 MB/mask RAM leak that OOM'd the Euler FIP build (see
docs/segmentation_3d/CROP_CACHE_OOM_AND_IOU_DEBUG.md and the reference note on the
torch-slice-.numpy() view leak).

This proves the new numpy build is:
  (1) BIT-IDENTICAL to the old torch path — same bbox, same area, same crop pixels; and
  (2) memory-FLAT — RSS does not grow per mask (no full-frame retention).
Runs on CPU in seconds, no GPU / no model / no 3D seg loop.

Usage:
    python src/analysis/verify_numpy_build.py                       # default FIP plot_461 masks
    python src/analysis/verify_numpy_build.py --masks_dir <dir> --n 3000
"""
import os
import sys
import glob
import resource
import argparse

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gaussians.utils.wheatgs_utils import PILtoTorch, binarize_mask, build_mask_crop

# FIP plot_461 is where the build leak actually OOM'd, so it's the natural default target.
DEFAULT_MASKS = "results/mask_generation/fip/plot_461/yolo_sam_v1/initial/masks"


def old_entry(path, resolution):
    """The ORIGINAL torch build path (via PILtoTorch + build_mask_crop). Kept only as the
    reference to diff against — this is the path that leaked on Euler."""
    with Image.open(path) as temp:
        m = binarize_mask(PILtoTorch(temp.copy(), resolution)).squeeze() > 0
    e = build_mask_crop(m)
    if e is None:
        return None
    y0, y1, x0, x1, crop, area = e
    return (y0, y1, x0, x1, crop, int(area))


def new_entry(path, resolution):
    """The NEW pure-numpy build path — an exact copy of _decode_crop's body in run_3d_seg.py.
    Never creates a full-frame torch tensor, so no parent buffer can be pinned."""
    with Image.open(path) as temp:
        arr = np.asarray(temp.resize(resolution))
    m_np = (arr > 0) if arr.ndim == 2 else (arr > 0).any(axis=2)
    ys, xs = np.nonzero(m_np)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop_np = m_np[y0:y1, x0:x1].copy()
    return (y0, y1, x0, x1, torch.from_numpy(crop_np), int(crop_np.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks_dir", default=DEFAULT_MASKS)
    ap.add_argument("--n", type=int, default=3000, help="how many masks to check")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.masks_dir, "*.png")))
    if not paths:
        sys.exit(f"no masks found in {args.masks_dir}")
    print(f"found {len(paths)} masks in {args.masks_dir}")

    # native size = resolution for the resolution=1 case (what the seg uses at full res)
    with Image.open(paths[0]) as im0:
        res = im0.size  # (W, H) — the same tuple PILtoTorch resizes to
    print(f"mask size {res[0]}x{res[1]}, resolution={res} (resolution=1 / no resize)")

    n = min(args.n, len(paths))

    # (1) bit-identical: old torch path vs new numpy path
    mism = 0
    for p in paths[:n]:
        o, nw = old_entry(p, res), new_entry(p, res)
        if (o is None) != (nw is None):
            mism += 1; print("None-mismatch", p); continue
        if o is None:
            continue
        if o[:4] != nw[:4] or o[5] != nw[5] or not torch.equal(o[4], nw[4]):
            mism += 1
            print(f"MISMATCH {os.path.basename(p)}: old {o[:4]},{o[5]} new {nw[:4]},{nw[5]} "
                  f"crop_eq={torch.equal(o[4], nw[4])}")
    print(f"\n(1) BIT-IDENTICAL: {mism} mismatches over {n} masks — "
          f"{'PASS ✅' if mism == 0 else 'FAIL ❌'}")

    # (2) memory-flat: build the cache with the NEW path only, watch RSS
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    cache = {}
    for i, p in enumerate(paths[:n]):
        e = new_entry(p, res)
        if e is not None:
            cache[p] = e
        if (i + 1) % 1000 == 0:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            cmb = sum(e[4].numel() for e in cache.values()) / 1e6
            print(f"    build {i+1}/{n}: peak RSS {rss:.2f} GB (Δ {rss - rss0:+.2f}), cache {cmb:.0f} MB")
    print("(2) MEMORY-FLAT: Δ should stay ~0 while cache grows only tens of MB (no full-frame retention)")


if __name__ == "__main__":
    main()
