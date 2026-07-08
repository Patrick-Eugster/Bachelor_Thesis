#!/usr/bin/env python3
"""Standalone runner for the 3DKD-wheat YOLO11-seg model (yolo-medium-segment.pt) — instance
segmentation, i.e. it outputs a mask PER wheat head directly, no SAM.

NOT wired into the pipeline; this is an exploratory tool to see whether instance-seg handles
occluding phone heads better than box-dedup. Saves, per plot, into
results/mask_generation/{dataset}/{plot}/yolo11_seg/{exp}/ :
    bboxes/<stem>.pt      the seg model's boxes (xyxy) — for a fair count vs the other methods
    masks/<stem>_NNN.png  one binary mask per instance (full-res)
    overlay/<stem>.jpg    the instance masks drawn in random colors (like sam_vis) to eyeball
    box_vis/<stem>.jpg    the seg model's boxes drawn as rectangles (like yolo_vis, but from the SEG
                          model — deliberately named box_vis, NOT yolo_vis, since these boxes are not
                          1:1 with the detector's yolo_vis: different model / conf / imgsz)

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
import time

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from mask_generation.roi_mask import apply_roi, roi_crop_box, roi_keep_mask

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


def fmt_dur(seconds):
    """Format a duration as '1h 02m 03s' / '2m 05s' / '9.4s' (h only when needed)."""
    s = float(seconds)
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m}m {sec:02d}s"


def load_bgr(path):
    """Load an image as a BGR numpy array via PIL (byte-identical to cv2.imread but silent).
    cv2.imread prints 'Invalid SOS parameters for sequential JPEG' on the Samsung-firmware phone
    JPEGs; PIL decodes them without the warning. Convert RGB->BGR so ultralytics + cv2 stay happy."""
    return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)


def build_roi_cfg(args):
    """Minimal dict cfg the roi_mask helpers understand (they only ever call .get()).
    --roi off → enabled False → apply_roi/roi_crop_box are no-ops (full-frame behaviour)."""
    return {"roi": {
        "enabled": bool(args.roi), "source": "markers", "fallback": "none", "min_markers": 3,
        "buffer_frac": args.buffer_frac, "buffer_px": 120, "fill": [114, 114, 114],
        "crop_before_inference": bool(args.roi),   # --roi turns on BOTH grey-out and crop-zoom
        "crop_top_margin_frac": args.crop_top, "crop_pad_frac": 0.0,
        "filter_boxes": True, "filter_mode": "overlap", "filter_tol_px": 0.0,
    }}


def predict_with_fallback(model, source, ladder, conf, iou, max_det):
    """Run seg predict, stepping imgsz down on CUDA OOM. Returns (result, imgsz_used) or (None, None).
    `source` is a numpy image (BGR) so we can hand it the ROI-greyed/cropped array, not a path."""
    for imgsz in ladder:
        try:
            r = model.predict(source, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
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


def box_overlay(img_bgr, boxes_xyxy):
    """Draw the seg model's boxes as rectangles on a copy — the box_vis output. Like the detector's
    yolo_vis but from the SEG model, so it's kept under box_vis/ (separate name) to avoid confusion."""
    out = img_bgr.copy()
    for x1, y1, x2, y2 in np.asarray(boxes_xyxy, dtype=np.int32).reshape(-1, 4):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # blue in BGR
    return out


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
    ap.add_argument("--roi", action="store_true",
                    help="phone ROI-zoom: grey outside the plot + CROP to the plot region before "
                         "inference so heads fill imgsz (higher effective res + less VRAM). Needs markers.")
    ap.add_argument("--buffer_frac", type=float, default=0.05, help="ROI soft-border (frac of short side)")
    ap.add_argument("--crop_top", type=float, default=0.10,
                    help="extra UPWARD crop margin (frac of short side) for tall side-view heads above the markers")
    ap.add_argument("--box_vis_only", action="store_true",
                    help="skip inference: just redraw box_vis/ from the existing bboxes/*.pt of --exp "
                         "(no model, no GPU). Use after an --no_masks run to add the box overlays.")
    args = ap.parse_args()
    ladder = [s for s in ALL_SIZES if s <= args.imgsz] or [args.imgsz]
    roi_cfg = build_roi_cfg(args)

    img_dir = os.path.join(INPUT_ROOT, args.dataset, args.plot, "images")
    images = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
    if not images:
        raise SystemExit(f"no images in {img_dir}")

    out = os.path.join(RESULT_ROOT, args.dataset, args.plot, "yolo11_seg", args.exp)
    bbox_dir, mask_dir, ov_dir, box_dir = [os.path.join(out, k)
                                           for k in ("bboxes", "masks", "overlay", "box_vis")]

    # --box_vis_only: redraw box_vis/ from the boxes already saved by a previous run — no inference.
    if args.box_vis_only:
        bfiles = sorted(glob.glob(os.path.join(bbox_dir, "*.pt")))
        if not bfiles:
            raise SystemExit(f"no bboxes/*.pt in {bbox_dir} — run the seg first, this only redraws")
        reset(box_dir)  # wipe ONLY box_vis; keep bboxes/masks/overlay from the original run
        img_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in images}
        nb = 0
        for bf in bfiles:
            stem = os.path.splitext(os.path.basename(bf))[0]
            ip = img_by_stem.get(stem)
            if ip is None:
                print(f"  {stem}: image not found in {img_dir}, skipping")
                continue
            boxes = torch.load(bf)
            boxes = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.asarray(boxes)
            cv2.imwrite(os.path.join(box_dir, stem + ".jpg"), box_overlay(load_bgr(ip), boxes),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            nb += int(np.asarray(boxes).size // 4)
        print(f"box_vis redrawn for {len(bfiles)} images ({nb} boxes) → {box_dir}")
        return

    for d in (bbox_dir, mask_dir, ov_dir, box_dir):
        reset(d)

    model = YOLO(WEIGHTS)
    print(f"YOLO11-seg loaded (task={model.task}) → {len(images)} images from {img_dir}")

    total = 0
    t0 = time.perf_counter()
    for i, p in enumerate(images):
        stem = os.path.splitext(os.path.basename(p))[0]
        img = load_bgr(p)  # PIL-based BGR load = silent on the Samsung-firmware phone JPEGs
        h, w = img.shape[:2]

        # ROI grey-out + crop-zoom (both no-ops unless --roi). We infer on the crop, then map
        # boxes/masks back to full-image px via (cx0,cy0) so the outputs stay in the original frame.
        img_roi = apply_roi(img, p, roi_cfg)
        crop = roi_crop_box(p, roi_cfg, w, h)
        cx0, cy0, cx1, cy1 = crop if crop is not None else (0, 0, w, h)
        infer_img = img_roi[cy0:cy1, cx0:cx1]

        r, used = predict_with_fallback(model, infer_img, ladder, args.conf, args.iou, args.max_det)
        if r is None:
            print(f"  {stem}: OOM at all imgsz — skipped")
            torch.save(torch.tensor([]), os.path.join(bbox_dir, stem + ".pt"))
            continue

        # boxes → full-image coords, then ROI box-filter (drop instances outside the plot polygon)
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            xyxy[:, [0, 2]] += cx0
            xyxy[:, [1, 3]] += cy0
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
        keep = roi_keep_mask(xyxy, p, roi_cfg, w, h) if len(xyxy) else np.zeros(0, dtype=bool)

        n = 0
        cw, ch = cx1 - cx0, cy1 - cy0
        if r.masks is not None and len(r.masks) > 0:
            # keep is index-aligned with r.boxes/r.masks; fall back to all if lengths ever mismatch
            idx = np.where(keep)[0] if len(keep) == len(r.masks) else np.arange(len(r.masks))
            n = len(idx)
            if args.no_masks:
                # FAST: polygons are in CROP px → offset back to full image, then fill (no PNG writes)
                polys = [r.masks.xy[k] + np.array([cx0, cy0], dtype=np.float32) for k in idx]
                overlay = poly_overlay(img, polys)
            else:
                md = r.masks.data.cpu().numpy()  # (N, mh, mw) at crop-input res
                masks_full = []
                for k in idx:  # resize each mask to the crop size, then paste into the full frame
                    mr = cv2.resize(md[k], (cw, ch), interpolation=cv2.INTER_NEAREST) > 0.5
                    full = np.zeros((h, w), dtype=bool)
                    full[cy0:cy1, cx0:cx1] = mr
                    masks_full.append(full)
                overlay = colored_overlay(img, masks_full)
                for j, m in enumerate(masks_full):  # per-instance binary masks (the slow part)
                    cv2.imwrite(os.path.join(mask_dir, f"{stem}_{j:03d}.png"), (m.astype(np.uint8) * 255))
        else:
            overlay = img

        kept_xyxy = xyxy[keep] if len(keep) == len(xyxy) else xyxy
        torch.save(torch.tensor(kept_xyxy) if len(kept_xyxy) else torch.tensor([]),
                   os.path.join(bbox_dir, stem + ".pt"))
        cv2.imwrite(os.path.join(ov_dir, stem + ".jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        # box_vis: the seg model's boxes as rectangles (separate from the detector's yolo_vis)
        cv2.imwrite(os.path.join(box_dir, stem + ".jpg"), box_overlay(img, kept_xyxy),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

        total += n
        del r                     # free the mask tensors so VRAM doesn't creep across frames
        torch.cuda.empty_cache()
        tag = "" if used == ladder[0] else f"  (imgsz {used} after OOM)"
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(images)}] {stem}: {n} instances{tag}")

    wall = time.perf_counter() - t0
    print(f"\nDONE: {total} instance masks across {len(images)} images ({total/max(1,len(images)):.0f}/img)")
    print(f"     runtime {fmt_dur(wall)} total, {wall/max(1,len(images)):.2f}s/image  (imgsz top={ladder[0]}, roi={args.roi})")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
