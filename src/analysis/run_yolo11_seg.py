#!/usr/bin/env python3
"""Standalone runner for the 3DKD-wheat YOLO11-seg model (yolo-medium-segment.pt) — instance
segmentation, i.e. it outputs a mask PER wheat head directly, no SAM.

NOT wired into the pipeline; this is an exploratory tool to see whether instance-seg handles
occluding phone heads better than box-dedup. Saves, per plot, into
results/mask_generation/{dataset}/{plot}/yolo11_seg/{exp}/ :
    bboxes/<stem>.pt      the seg model's boxes (xyxy) — for a fair count vs the other methods
    masks/<stem>_NNN.png  one binary mask per instance (full-res)
    overlay/<stem>.jpg    the instance masks drawn in random colors (like sam_vis) to eyeball

OOM handling: the seg head upsamples every mask to imgsz^2 on the GPU, which OOMs 16 GB on dense
images at 3008. So we run per-image and, on CUDA OOM, retry that image at a smaller imgsz.

Usage:
    python src/analysis/run_yolo11_seg.py --dataset fip   --plot plot_461
    python src/analysis/run_yolo11_seg.py --dataset phone --plot field_A/20250715
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np
import torch
from ultralytics import YOLO

WEIGHTS = "/workspace/src/mask_generation/weights/yolo-medium-segment.pt"
INPUT_ROOT = "/workspace/input_plots"
RESULT_ROOT = "/workspace/results/mask_generation"
# seg-head mask upsample needs N×imgsz² on the GPU. On DENSE images (phone ~700+ heads) even 2048
# OOMs 16 GB, and a driver/allocator OOM CORRUPTS the CUDA context → the next image hard-crashes
# (not a catchable OOM). So the safe move is to pick a start imgsz that never OOMs for the density,
# not to rely on stepping down mid-process. --imgsz sets the top; we still keep a couple of fallbacks.
ALL_SIZES = [2048, 1600, 1280, 1024]


def reset(d):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)


def predict_with_fallback(model, img_path, ladder, conf, iou, max_det):
    """Run seg predict, stepping imgsz down on CUDA OOM. Returns (result, imgsz_used) or (None, None)."""
    for imgsz in ladder:
        try:
            r = model.predict(img_path, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                              verbose=False, device=0 if torch.cuda.is_available() else "cpu")[0]
            return r, imgsz
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # catch the clean OOM, the driver "CUDA driver error: out of memory", AND the allocator
            # "INTERNAL ASSERT" that follows a corrupting OOM — all mean "too big, try smaller".
            msg = str(e).lower()
            if not any(k in msg for k in ("out of memory", "internal assert", "cuda error")):
                raise
            torch.cuda.empty_cache()
            continue
    return None, None


def colored_overlay(img_bgr, masks):
    """Draw each instance mask as a translucent random color over the image."""
    out = img_bgr.astype(np.float32)
    rng = np.random.default_rng(0)
    for m in masks:
        c = rng.integers(60, 255, 3).astype(np.float32)
        out[m] = 0.45 * out[m] + 0.55 * c
    return out.astype(np.uint8)


def poly_overlay(img_bgr, polygons):
    """FAST overlay: fill each instance polygon (masks.xy is already in ORIGINAL image px) with a
    random translucent color. Avoids the per-mask full-res resize — visual inspection only."""
    layer = img_bgr.copy()
    rng = np.random.default_rng(0)
    for poly in polygons:
        if poly is None or len(poly) < 3:
            continue
        c = tuple(int(x) for x in rng.integers(60, 255, 3))
        cv2.fillPoly(layer, [poly.astype(np.int32)], c)
    out = img_bgr.copy()
    cv2.addWeighted(layer, 0.5, out, 0.5, 0, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fip", "phone"])
    ap.add_argument("--plot", required=True, help="fip: plot_461 ; phone: field_A/20250715")
    ap.add_argument("--exp", default="initial")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max_det", type=int, default=1000)
    ap.add_argument("--imgsz", type=int, default=2048,
                    help="top imgsz (FIP ~270 heads: 2048 ok; DENSE phone ~700 heads: use 1280 to avoid OOM-crash)")
    ap.add_argument("--save_masks", action="store_true", default=True)
    ap.add_argument("--no_masks", action="store_true",
                    help="skip per-instance PNG masks + use the fast polygon overlay (visual only, no slow I/O)")
    args = ap.parse_args()
    ladder = [s for s in ALL_SIZES if s <= args.imgsz] or [args.imgsz]

    img_dir = os.path.join(INPUT_ROOT, args.dataset, args.plot, "images")
    images = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
    if not images:
        raise SystemExit(f"no images in {img_dir}")

    out = os.path.join(RESULT_ROOT, args.dataset, args.plot, "yolo11_seg", args.exp)
    bbox_dir, mask_dir, ov_dir = [os.path.join(out, k) for k in ("bboxes", "masks", "overlay")]
    for d in (bbox_dir, mask_dir, ov_dir):
        reset(d)

    model = YOLO(WEIGHTS)
    print(f"YOLO11-seg loaded (task={model.task}) → {len(images)} images from {img_dir}")

    total = 0
    for i, p in enumerate(images):
        stem = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p)
        h, w = img.shape[:2]
        r, used = predict_with_fallback(model, p, ladder, args.conf, args.iou, args.max_det)
        if r is None:
            print(f"  {stem}: OOM at all imgsz — skipped")
            torch.save(torch.tensor([]), os.path.join(bbox_dir, stem + ".pt"))
            continue

        n = 0
        if r.masks is not None and len(r.masks) > 0:
            n = len(r.masks)
            if args.no_masks:
                # FAST: draw the polygons directly (already original px) — no per-mask resize / PNG writes
                overlay = poly_overlay(img, r.masks.xy)
            else:
                md = r.masks.data.cpu().numpy()  # (N, mh, mw) at input res
                masks_full = [cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5 for m in md]
                overlay = colored_overlay(img, masks_full)
                for j, m in enumerate(masks_full):  # per-instance binary masks (the slow part)
                    cv2.imwrite(os.path.join(mask_dir, f"{stem}_{j:03d}.png"), (m.astype(np.uint8) * 255))
        else:
            overlay = img
        # boxes (the seg model also outputs boxes)
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            torch.save(torch.tensor(xyxy), os.path.join(bbox_dir, stem + ".pt"))
        else:
            torch.save(torch.tensor([]), os.path.join(bbox_dir, stem + ".pt"))
        cv2.imwrite(os.path.join(ov_dir, stem + ".jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])

        total += n
        del r                     # free the mask tensors so VRAM doesn't creep across frames
        torch.cuda.empty_cache()
        tag = "" if used == ladder[0] else f"  (imgsz {used} after OOM)"
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(images)}] {stem}: {n} instances{tag}")

    print(f"\nDONE: {total} instance masks across {len(images)} images ({total/max(1,len(images)):.0f}/img)")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
