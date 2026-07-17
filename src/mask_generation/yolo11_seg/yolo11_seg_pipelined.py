"""
yolo11_seg_pipelined.py — YOLO11-seg as a first-class mask-generation method.

Unlike the other methods (a detector that produces boxes → shared SAM phase turns boxes into masks),
YOLO11-seg is an INSTANCE-SEGMENTATION model: one forward pass emits a mask PER wheat head directly,
no SAM. So its method config sets `only_yolo: true` and the orchestrator skips the SAM phase — this
function writes the final masks/ itself.

Ported from the standalone src/analysis/run_yolo11_seg.py so it can run through run_mask_generation.py
(`method=yolo11_seg`) with the same dataset / ROI / only_labeled_images / result-path plumbing as the
other methods. ROI comes from cfg.roi (grey-out; crop-zoom only if crop_before_inference is set, which
the production config leaves off — matching sahi/yolo). Weight from cfg.method.yolo11_seg_model.

Writes per plot, into results/mask_generation/{dataset}/{plot}/yolo11_seg/{experiment}/:
    bboxes/<stem>.pt   the seg model's boxes (xyxy) — for a fair count vs the other methods
    masks/<stem>_NNN.png  one full-res binary mask per head (what eval_masks_instance.py reads)
    overlay/<stem>.jpg    quick colour overlay for eyeballing
"""

import os
import glob
import time

import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import YOLO

from mask_generation.roi_mask import apply_roi, roi_crop_box, roi_keep_mask
from mask_generation.gt_labels import gt_labeled_stems
from mask_generation.yolo_sam_v1.yolo_v1_pipelined import reset_folder
from wheat_utils.path_utils import get_mask_generation_result_path

# seg-head mask upsample needs N×imgsz² on the GPU; on dense phone images (~700 heads) 2048 OOMs 16 GB
# (and a corrupting OOM hard-crashes the next image). imgsz sets the top; we step down on OOM.
ALL_SIZES = [2048, 1600, 1280, 1024]


def _load_bgr(path):
    """PIL-based BGR load — byte-identical to cv2.imread but silent on the Samsung-firmware phone JPEGs
    (cv2 prints 'Invalid SOS parameters for sequential JPEG')."""
    return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)


def _predict_with_fallback(model, source, ladder, conf, iou, max_det, device):
    """Run seg predict, stepping imgsz down on CUDA OOM. Returns (result, imgsz_used) or (None, None)."""
    for imgsz in ladder:
        try:
            r = model.predict(source, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                              verbose=False, device=device)[0]
            return r, imgsz
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            msg = str(e).lower()
            if not any(k in msg for k in ("out of memory", "internal assert", "cuda error")):
                raise
            torch.cuda.empty_cache()
            continue
    return None, None


def _colored_overlay(img_bgr, masks):
    """Translucent random colour per instance mask over the image — visual check only."""
    out = img_bgr.astype(np.float32)
    rng = np.random.default_rng(0)
    for m in masks:
        c = rng.integers(60, 255, 3).astype(np.float32)
        out[m] = 0.45 * out[m] + 0.55 * c
    return out.astype(np.uint8)


def _gather_images(folder, base_plot_path, cfg):
    """Same filtering rules as the other methods: labeled-only for metrics (now recognises mask-GT via
    gt_labeled_stems), else capped at limit_images, else all images of the plot."""
    image_files = sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")))
    if cfg.only_labeled_images:
        stems = gt_labeled_stems(os.path.join(base_plot_path, "manual_label"))
        image_files = [f for f in image_files if os.path.splitext(os.path.basename(f))[0] in stems]
        print(f"---ONLY_LABELED_IMAGES: filtered to {len(image_files)} labeled images")
    elif cfg.limit_images > 0:
        image_files = image_files[:cfg.limit_images]
    return image_files


def run_yolo11_seg_phase(image_folders, cfg):
    """Run YOLO11-seg on every plot's images and write masks/ + bboxes/ + overlay/. Returns the total
    number of instance masks written (used by the orchestrator's report as the box count)."""
    device = 0 if torch.cuda.is_available() else "cpu"
    weights_dir = os.path.join(os.path.dirname(__file__), "..", "weights")
    model_path = os.path.join(weights_dir, cfg.method.yolo11_seg_model)
    model = YOLO(model_path)
    print(f"Loaded YOLO11-seg: {cfg.method.yolo11_seg_model}  (task={model.task}, device={device})")

    top = int(cfg.method.imgsz)
    ladder = [s for s in ALL_SIZES if s <= top] or [top]
    conf = float(cfg.method.conf_threshold)
    iou = float(cfg.method.iou_threshold_nms)
    max_det = int(cfg.method.max_det)

    grand_total = 0
    for folder in image_folders:
        base_plot_path = os.path.dirname(folder)
        plot_name = os.path.relpath(base_plot_path, cfg.dataset.input_dir)
        result_path = get_mask_generation_result_path(cfg, plot_name)
        bbox_dir = os.path.join(result_path, "bboxes")
        mask_dir = os.path.join(result_path, "masks")
        ov_dir = os.path.join(result_path, "overlay")
        for d in (bbox_dir, mask_dir, ov_dir):
            reset_folder(d)

        images = _gather_images(folder, base_plot_path, cfg)
        print(f"\n[{plot_name}] YOLO11-seg on {len(images)} images (imgsz top={ladder[0]})")
        t0 = time.perf_counter()
        plot_total = 0
        for i, p in enumerate(images):
            stem = os.path.splitext(os.path.basename(p))[0]
            img = _load_bgr(p)
            h, w = img.shape[:2]

            # ROI: grey-out (+ crop-zoom only if crop_before_inference is set). Infer on the (possibly
            # cropped) frame, then map boxes/masks back to full-image px.
            img_roi = apply_roi(img, p, cfg)
            crop = roi_crop_box(p, cfg, w, h)
            cx0, cy0, cx1, cy1 = crop if crop is not None else (0, 0, w, h)
            infer_img = img_roi[cy0:cy1, cx0:cx1]

            r, used = _predict_with_fallback(model, infer_img, ladder, conf, iou, max_det, device)
            if r is None:
                print(f"  {stem}: OOM at all imgsz — skipped")
                torch.save(torch.tensor([]), os.path.join(bbox_dir, stem + ".pt"))
                continue

            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                xyxy[:, [0, 2]] += cx0
                xyxy[:, [1, 3]] += cy0
            else:
                xyxy = np.zeros((0, 4), dtype=np.float32)
            keep = roi_keep_mask(xyxy, p, cfg, w, h) if len(xyxy) else np.zeros(0, dtype=bool)

            n = 0
            cw, ch = cx1 - cx0, cy1 - cy0
            if r.masks is not None and len(r.masks) > 0:
                idx = np.where(keep)[0] if len(keep) == len(r.masks) else np.arange(len(r.masks))
                n = len(idx)
                md = r.masks.data.cpu().numpy()                 # (N, mh, mw) at crop-input res
                masks_full = []
                for k in idx:                                   # resize each mask to the crop, paste to full frame
                    mr = cv2.resize(md[k], (cw, ch), interpolation=cv2.INTER_NEAREST) > 0.5
                    full = np.zeros((h, w), dtype=bool)
                    full[cy0:cy1, cx0:cx1] = mr
                    masks_full.append(full)
                for j, m in enumerate(masks_full):
                    cv2.imwrite(os.path.join(mask_dir, f"{stem}_{j:03d}.png"), (m.astype(np.uint8) * 255))
                if cfg.method.get("save_overlay", True):
                    cv2.imwrite(os.path.join(ov_dir, stem + ".jpg"), _colored_overlay(img, masks_full),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])

            kept_xyxy = xyxy[keep] if len(keep) == len(xyxy) else xyxy
            torch.save(torch.tensor(kept_xyxy) if len(kept_xyxy) else torch.tensor([]),
                       os.path.join(bbox_dir, stem + ".pt"))
            plot_total += n
            del r
            torch.cuda.empty_cache()
            tag = "" if used == ladder[0] else f"  (imgsz {used} after OOM)"
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(images)}] {stem}: {n} instances{tag}")

        wall = time.perf_counter() - t0
        per = wall / max(1, len(images))
        print(f"[{plot_name}] {plot_total} masks across {len(images)} images "
              f"({plot_total/max(1,len(images)):.0f}/img, {per:.1f}s/img) → {result_path}")
        grand_total += plot_total
    return grand_total
