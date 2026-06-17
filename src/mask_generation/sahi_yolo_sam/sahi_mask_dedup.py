"""
sahi_mask_dedup.py — EXPERIMENTAL: SAHI with MASK-based dedup (does NOT touch the normal pipeline).

The normal SAHI path merges the overlapping-tile boxes with box overlap (IOS-NMM) BEFORE SAM. That
wrongly absorbs a small head that sits inside a big *diagonal* head's axis-aligned box (the box has
empty corners; box-IOS can't tell "two different heads" from "one head detected twice").

This file does it the other way round:
  1. SAHI tiles -> YOLO -> all pre-merge boxes  (NO box merge)            [reuses sahi_yolo_pipelined]
  2. light IoU pre-collapse of near-identical boxes (so we don't SAM the same head many times)
  3. SAM one CLEAN mask per box: positive point at the box center + negative points at the centers of
     overlapping neighbour boxes ("this head, not those")  -> each detected head gets its own mask
  4. dedup on MASK overlap: two masks that overlap a lot = same head -> merge; little = distinct -> keep
  5. write masks/ + bboxes/ (box = bounding box of each kept mask) + a colored viz

Scope (honest): this only recovers heads YOLO *detected* (each has a box) that the box-merge wrongly
fused — it does NOT find heads YOLO missed (no box -> no prompt). The spike that justified this is in
docs/SAHI_EVAL_RESULTS.md §6. Standalone — the normal sahi_yolo_pipelined / run_mask_generation are
untouched.

Run:  python src/mask_generation/sahi_yolo_sam/sahi_mask_dedup.py plot_glob=plot_461 limit_images=1
Out:  results/mask_generation/{dataset}/{plot}/sahi_yolo_sam/{experiment_name}/{bboxes,masks,viz}/
"""

import os
import sys
import glob
import shutil
import colorsys
import warnings
import numpy as np
import torch

# yolov5 uses the old torch.cuda.amp.autocast API — harmless, silence just that FutureWarning
warnings.filterwarnings("ignore", message=r".*torch\.cuda\.amp\.autocast.*", category=FutureWarning)

import hydra
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont
from segment_anything import sam_model_registry, SamPredictor

# reuse the SAHI tiling/inference (read-only — we do NOT call merge_preds)
import sahi_yolo_pipelined as sp
from sahi_yolo_pipelined import _iou_ios, load_and_slice, infer_tiles
from wheat_utils.path_utils import get_mask_generation_result_path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_yolo(cfg):
    """Load the wheat YOLO model with conf set to the single SAHI keep line (conf_threshold_good_box),
    exactly like the normal SAHI phase — so the pre-merge boxes are the same ones SAHI would tile."""
    base = os.path.dirname(sp.__file__)
    model = torch.hub.load(os.path.join(base, "..", "yolov5"), 'custom',
                           path=os.path.join(base, "..", "weights", cfg.method.wheat_yolo_model),
                           source='local')
    model.conf = cfg.method.conf_threshold_good_box
    model.iou  = cfg.method.iou_threshold_nms
    model.classes = list(cfg.method.classes_to_detect)
    return model


def box_center(b):
    """Center (x,y) of an xyxy box — for a diagonal head this lands ON the head, not in the empty corner."""
    return [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]


def iou_precollapse(boxes, iou_thr):
    """Greedy NMS that drops only NEAR-IDENTICAL boxes (IoU>=thr), keeping the highest-conf one. This is
    a safe cut of obvious duplicates so we don't run SAM on the same head many times; nested heads have
    LOW IoU so they are never collapsed here. Returns kept row indices."""
    order = np.argsort(-boxes[:, 4])
    suppressed = np.zeros(len(boxes), dtype=bool)
    keep = []
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        iou, _ = _iou_ios(boxes[idx, :4], boxes[:, :4])
        dup = iou >= iou_thr
        dup[idx] = False
        suppressed |= dup
    return sorted(keep)


def sam_clean_mask(predictor, boxes, i, iou_row, ios_row, cfg):
    """One CLEAN mask for box i: positive point at its center + negative points at the centers of
    DISTINCT overlapping neighbours. A negative must have high box-IoS (it's contained/overlapping)
    AND low box-IoU (it's NOT a near-duplicate of the same head) — otherwise we'd put a negative point
    ON head i itself and SAM would return garbage. iou_row/ios_row = box-IoU/IoS of box i vs all boxes."""
    pos = box_center(boxes[i])
    cand = [j for j in range(len(boxes))
            if j != i and ios_row[j] >= cfg.neg_ios_min and iou_row[j] < cfg.neg_iou_max]
    cand.sort(key=lambda j: (box_center(boxes[j])[0] - pos[0]) ** 2 + (box_center(boxes[j])[1] - pos[1]) ** 2)
    negs = [box_center(boxes[j]) for j in cand[:cfg.max_neg]]
    pts = np.array([pos] + negs, dtype=np.float32)
    labs = np.array([1] + [0] * len(negs), dtype=np.int32)
    masks, _, _ = predictor.predict(point_coords=pts, point_labels=labs, multimask_output=False)
    return masks[0].astype(bool)


def crop_mask(full):
    """Shrink a full HxW boolean mask to its tight bounding box -> ((x1,y1,x2,y2), local_bool). Keeps
    memory small (a head is tiny vs the whole image). Returns None if the mask is empty."""
    ys, xs = np.where(full)
    if len(xs) == 0:
        return None
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    return (x1, y1, x2, y2), full[y1:y2, x1:x2]


def mask_ios(A, B):
    """Intersection-over-smaller of two cropped masks (computed only on their bbox overlap, so it's cheap)."""
    (ax1, ay1, ax2, ay2), ma = A
    (bx1, by1, bx2, by2), mb = B
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    sa = ma[iy1 - ay1:iy2 - ay1, ix1 - ax1:ix2 - ax1]
    sb = mb[iy1 - by1:iy2 - by1, ix1 - bx1:ix2 - bx1]
    inter = int(np.logical_and(sa, sb).sum())
    smaller = min(int(ma.sum()), int(mb.sum()))
    return inter / smaller if smaller else 0.0


def mask_dedup(cropped, confs, thr):
    """Greedy dedup on MASK overlap: keep the highest-conf mask, drop any later mask that overlaps it
    (mask-IoS>=thr) — those are the SAME head detected twice. Distinct heads (low overlap) all survive.
    Returns kept indices (into cropped)."""
    order = list(np.argsort(-np.asarray(confs)))
    used = np.zeros(len(cropped), dtype=bool)
    keep = []
    for i in order:
        if used[i]:
            continue
        used[i] = True
        keep.append(i)
        for j in order:
            if not used[j] and mask_ios(cropped[i], cropped[j]) >= thr:
                used[j] = True
    return keep


def _font(size):
    """A readable bold font for the conf labels, falling back to PIL's default if unavailable."""
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def save_outputs(name, H, W, kept_cropped, kept_conf, dirs, save_viz, image_np):
    """Write masks/{name}_NNN.png (full-size binary), bboxes/{name}.pt [N,4], bboxes_with_conf [N,5],
    and (if save_viz) viz/ (colored masks) + yolo_vis/ (boxes + conf labels, like the normal pipeline).
    Boxes are the bounding boxes of the final masks (index-aligned with the masks)."""
    boxes = []
    viz = image_np.astype(np.float32).copy() if save_viz else None
    for idx, ((x1, y1, x2, y2), local) in enumerate(kept_cropped):
        full = np.zeros((H, W), dtype=np.uint8)
        full[y1:y2, x1:x2] = local.astype(np.uint8) * 255
        Image.fromarray(full).save(os.path.join(dirs['masks'], f"{name}_{idx:03d}.png"))
        boxes.append([x1, y1, x2, y2])
        if save_viz:
            r, g, b = colorsys.hsv_to_rgb((idx * 0.61803398875) % 1.0, 0.8, 1.0)
            m = full > 0
            viz[m] = viz[m] * 0.5 + np.array([r * 255, g * 255, b * 255]) * 0.5
    boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    torch.save(boxes_t, os.path.join(dirs['bboxes'], f"{name}.pt"))
    conf_t = torch.tensor([[*boxes[i], kept_conf[i]] for i in range(len(boxes))], dtype=torch.float32).reshape(-1, 5)
    torch.save(conf_t, os.path.join(dirs['bboxes_with_conf'], f"{name}.pt"))
    if save_viz:
        Image.fromarray(viz.astype(np.uint8)).save(os.path.join(dirs['viz'], f"{name}.jpg"), quality=90)
        # yolo_vis: the final boxes (= mask bounding boxes) drawn on the image with conf labels
        ann = Image.fromarray(image_np.astype(np.uint8)).convert("RGB")
        d = ImageDraw.Draw(ann)
        font = _font(28)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            d.text((x1, max(0, y1 - 30)), f"{kept_conf[i]:.2f}", fill=(255, 255, 255),
                   font=font, stroke_width=2, stroke_fill=(0, 130, 0))
        ann.save(os.path.join(dirs['yolo_vis'], f"{name}.jpg"), quality=90)


def select_images(cfg, plot_dir):
    """Images of a plot to process: GT-labeled first if labeled_only + manual_label exists, else all."""
    files = sorted(glob.glob(os.path.join(plot_dir, 'images', '*.png')) +
                   glob.glob(os.path.join(plot_dir, 'images', '*.jpg')))
    if cfg.labeled_only and os.path.isdir(os.path.join(plot_dir, 'manual_label')):
        stems = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(plot_dir, 'manual_label')) if f.endswith('.txt')}
        lab = [f for f in files if os.path.splitext(os.path.basename(f))[0] in stems]
        if lab:
            files = lab
    return files[:cfg.limit_images] if cfg.limit_images > 0 else files


def run(cfg):
    """SAHI tiles (no box-merge) -> SAM clean per-head masks -> mask dedup -> save, per image."""
    print(f"\n{'=' * 60}\n SAHI MASK-DEDUP (experimental)\n{'=' * 60}")
    print(f"  pre_collapse_iou={cfg.pre_collapse_iou} neg_ios_min={cfg.neg_ios_min} "
          f"max_neg={cfg.max_neg} mask_dedup_ios={cfg.mask_dedup_ios}")
    model = load_yolo(cfg)
    print(f"Loading SAM on {DEVICE} ...")
    sam = sam_model_registry["vit_h"](checkpoint=os.path.join(
        os.path.dirname(sp.__file__), "..", "weights", cfg.method.sam_checkpoint)).to(DEVICE)
    predictor = SamPredictor(sam)

    for plot_dir in sorted(glob.glob(os.path.join(cfg.dataset.input_dir, cfg.plot_glob))):
        if not os.path.isdir(os.path.join(plot_dir, 'images')):
            continue
        plot_name = os.path.relpath(plot_dir, cfg.dataset.input_dir)
        out = get_mask_generation_result_path(cfg, plot_name)
        dirs = {k: os.path.join(out, k) for k in ['bboxes', 'bboxes_with_conf', 'masks', 'viz', 'yolo_vis']}
        for d in dirs.values():
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)
        images = select_images(cfg, plot_dir)
        if not images:
            continue
        print(f"\n[{plot_name}] {len(images)} image(s) -> {out}")
        for img_path in images:
            name, img_np, H, W, crops, offsets = load_and_slice(img_path, cfg)
            preds = infer_tiles(model, img_np, crops, offsets, W, H, cfg)   # [N,5], all pre-merge boxes
            if len(preds) == 0:
                save_outputs(name, H, W, [], [], dirs, cfg.save_viz, img_np)
                continue
            keep_idx = iou_precollapse(preds, cfg.pre_collapse_iou)
            boxes = preds[keep_idx]                                          # boxes we will SAM
            predictor.set_image(img_np)                                     # expensive encode, once
            cropped, confs = [], []
            for i in range(len(boxes)):
                iou_row, ios_row = _iou_ios(boxes[i, :4], boxes[:, :4])
                cm = crop_mask(sam_clean_mask(predictor, boxes, i, iou_row, ios_row, cfg))
                if cm is not None:
                    cropped.append(cm)
                    confs.append(float(boxes[i, 4]))
            kept = mask_dedup(cropped, confs, cfg.mask_dedup_ios)
            save_outputs(name, H, W, [cropped[k] for k in kept], [confs[k] for k in kept],
                         dirs, cfg.save_viz, img_np)
            print(f"  {name}: raw {len(preds)} -> pre-collapse {len(boxes)} -> SAM {len(cropped)} "
                  f"-> dedup {len(kept)} heads")
    print("\nDone.")


@hydra.main(version_base=None, config_path="../../../configs/mask_generation", config_name="sahi_mask_dedup")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
