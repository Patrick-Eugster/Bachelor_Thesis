"""
sahi_surgical_dedup.py — EXPERIMENTAL: SAHI with SURGICAL/HYBRID dedup (touches NOTHING in production
or in the v1 mask-dedup).

The v1 mask-dedup (sahi_mask_dedup.py) runs SAM on EVERY box, then dedups on masks. Most boxes are
clean non-overlapping single heads that never needed SAM — re-segmenting them only wastes time and
adds noise. Surgical only spends SAM where there's genuine ambiguity:

  1. SAHI tiles -> YOLO -> all pre-merge boxes                          [reuses sahi_yolo_pipelined]
  2. TIER 2: IoU-NMS collapse (IoU >= nms_merge_iou) = same head detected in two tiles -> keep the
     max-conf box, drop the duplicates. Nested heads have LOW IoU so they SURVIVE this step.
  3. classify the survivors by box overlap:
       TIER 1 (no contained partner)  = clean single head  -> keep the YOLO box AS-IS, no SAM.
       TIER 3 (a contained partner: box-IoS>=contained_ios AND box-IoU<contained_iou_max)
                                       = nested-vs-fragment -> AMBIGUOUS -> SAM decides.
  4. TIER 3 only: SAM one clean mask per ambiguous box (positive center point + negative points on
     distinct overlapping neighbours), then decide split-vs-keep on MASK overlap (mask-IoS).
  5. final box set = tier-1 boxes + tier-3 survivors. Save bboxes/ + bboxes_with_conf/ + yolo_vis/.

Result is a BOX set (same contract as the normal YOLO phase) — the SAM masks here are only an
internal tie-breaker, the real per-head masks still come later in the normal SAM phase.

Honest bound: like v1, this only un-merges heads YOLO DETECTED (each tier-3 box exists because YOLO
boxed it). A nested head with no box can't be recovered — that's a detector limitation. Full write-up
+ the v1 comparison: docs/mask_generation/SAHI_SURGICAL_DEDUP.md. Standalone — production and v1 are untouched.

Run:  python src/mask_generation/sahi_yolo_sam/sahi_surgical_dedup.py plot_glob=plot_461 limit_images=1
Out:  results/mask_generation/{dataset}/{plot}/sahi_yolo_sam/{experiment_name}/{bboxes,masks,viz,yolo_vis}/
"""

import os
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
from PIL import Image, ImageDraw
from segment_anything import sam_model_registry, SamPredictor

# reuse the SAHI tiling/inference (read-only — we never call merge_preds)
import sahi_yolo_pipelined as sp
from sahi_yolo_pipelined import _iou_ios, load_and_slice, infer_tiles
# reuse the pure helpers from the v1 mask-dedup (read-only — v1 is not modified)
from sahi_mask_dedup import (load_yolo, box_center, crop_mask,
                             iou_precollapse, mask_dedup, _font, select_images)
from wheat_utils.path_utils import get_mask_generation_result_path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_contained(boxes, cfg):
    """Flag the boxes that are part of a CONTAINED pair (the only ones that need SAM). A box is
    ambiguous if some OTHER box has high box-IoS (>=contained_ios, it's nested/overlapping) AND low
    box-IoU (<contained_iou_max, so it's NOT a tier-2 duplicate). Boxes with no such partner are
    clean single heads (tier 1). Returns a boolean array aligned with boxes."""
    n = len(boxes)
    ambiguous = np.zeros(n, dtype=bool)
    for i in range(n):
        iou, ios = _iou_ios(boxes[i, :4], boxes[:, :4])
        partner = (ios >= cfg.contained_ios) & (iou < cfg.contained_iou_max)
        partner[i] = False
        if partner.any():
            ambiguous[i] = True
    return ambiguous


def sam_box_mask(predictor, survivors, i, iou_row, ios_row, cfg):
    """SAM mask for box i using BOX + positive point + negative points (the surgical, leak-resistant
    prompt). The detection box bounds the mask to that head's rectangle so SAM can't grab neighbours
    or background — this fixes the point-only leaks (a magenta box around several heads, a giant box
    over no head). The negatives sit on DISTINCT overlapping neighbours (high box-IoS AND low box-IoU)
    to carve them out within the box. Set use_box_prompt=false to fall back to the old point-only prompt."""
    pos = box_center(survivors[i])
    cand = [j for j in range(len(survivors))
            if j != i and ios_row[j] >= cfg.neg_ios_min and iou_row[j] < cfg.neg_iou_max]
    cand.sort(key=lambda j: (box_center(survivors[j])[0] - pos[0]) ** 2 +
                            (box_center(survivors[j])[1] - pos[1]) ** 2)
    negs = [box_center(survivors[j]) for j in cand[:cfg.max_neg]]
    pts = np.array([pos] + negs, dtype=np.float32)
    labs = np.array([1] + [0] * len(negs), dtype=np.int32)
    box = survivors[i, :4].astype(np.float32) if cfg.use_box_prompt else None
    masks, _, _ = predictor.predict(point_coords=pts, point_labels=labs, box=box, multimask_output=False)
    return masks[0].astype(bool)


def save_surgical(name, H, W, boxes, confs, tiers, masks, dirs, save_viz, image_np):
    """Write the final BOX set the same way as the normal YOLO phase: bboxes/{name}.pt [N,4],
    bboxes_with_conf/{name}.pt [N,5]. Tier-3 boxes (SAM-resolved) also get a binary mask in masks/.
    If save_viz: yolo_vis/ draws every box (green = tier1 kept as-is, magenta = tier3 SAM-resolved)
    with conf labels, and viz/ overlays the tier-3 masks so you can see exactly where SAM was spent."""
    boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    torch.save(boxes_t, os.path.join(dirs['bboxes'], f"{name}.pt"))
    conf_t = torch.tensor([[*boxes[i], confs[i]] for i in range(len(boxes))],
                          dtype=torch.float32).reshape(-1, 5)
    torch.save(conf_t, os.path.join(dirs['bboxes_with_conf'], f"{name}.pt"))

    viz = image_np.astype(np.float32).copy() if save_viz else None
    # only tier-3 heads have a SAM mask (tier-1 was kept as a plain box, no SAM ran)
    for idx, (m, t) in enumerate(zip(masks, tiers)):
        if m is None:
            continue
        (x1, y1, x2, y2), local = m
        full = np.zeros((H, W), dtype=np.uint8)
        full[y1:y2, x1:x2] = local.astype(np.uint8) * 255
        Image.fromarray(full).save(os.path.join(dirs['masks'], f"{name}_{idx:03d}.png"))
        if save_viz:
            r, g, b = colorsys.hsv_to_rgb((idx * 0.61803398875) % 1.0, 0.85, 1.0)
            sel = full > 0
            viz[sel] = viz[sel] * 0.5 + np.array([r * 255, g * 255, b * 255]) * 0.5

    if save_viz:
        # viz: tier-3 masks overlaid (above) + outlines of the no-mask boxes (green tier-1 kept as-is,
        # orange tier-3 area-guard fallback where SAM leaked and we kept the raw YOLO box)
        vimg = Image.fromarray(viz.astype(np.uint8)).convert("RGB")
        vd = ImageDraw.Draw(vimg)
        for (x1, y1, x2, y2), t, m in zip(boxes, tiers, masks):
            if t == 1:
                vd.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            elif m is None:                                              # tier-3 area-guard fallback
                vd.rectangle([x1, y1, x2, y2], outline=(255, 140, 0), width=2)
        vimg.save(os.path.join(dirs['viz'], f"{name}.jpg"), quality=90)

        # yolo_vis: every final box with conf labels, colored by outcome:
        #   green = tier-1 (kept as-is) · magenta = tier-3 SAM mask · orange = tier-3 leaked->raw box
        ann = Image.fromarray(image_np.astype(np.uint8)).convert("RGB")
        d = ImageDraw.Draw(ann)
        font = _font(28)
        for (x1, y1, x2, y2), c, t, m in zip(boxes, confs, tiers, masks):
            if t == 1:
                color, stroke = (0, 255, 0), (0, 90, 0)
            elif m is not None:
                color, stroke = (255, 0, 255), (90, 0, 90)
            else:
                color, stroke = (255, 140, 0), (90, 50, 0)
            d.rectangle([x1, y1, x2, y2], outline=color, width=3)
            d.text((x1, max(0, y1 - 30)), f"{c:.2f}", fill=(255, 255, 255),
                   font=font, stroke_width=2, stroke_fill=stroke)
        ann.save(os.path.join(dirs['yolo_vis'], f"{name}.jpg"), quality=90)


def run(cfg):
    """Per image: SAHI tiles (no box-merge) -> tier-2 IoU-NMS -> classify tier1/tier3 -> SAM only on
    tier3 ambiguous boxes -> mask-decide split-vs-keep -> save the final box set."""
    print(f"\n{'=' * 60}\n SAHI SURGICAL DEDUP (experimental)\n{'=' * 60}")
    print(f"  nms_merge_iou={cfg.nms_merge_iou} contained_ios={cfg.contained_ios} "
          f"contained_iou_max={cfg.contained_iou_max} decide_mask_ios={cfg.decide_mask_ios}")
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
                save_surgical(name, H, W, [], [], [], [], dirs, cfg.save_viz, img_np)
                continue

            # TIER 2: collapse same-head duplicates (high IoU). Nested heads (low IoU) survive.
            survivors = preds[iou_precollapse(preds, cfg.nms_merge_iou)]
            # classify survivors: clean singles (tier 1) vs contained pairs (tier 3 -> SAM)
            ambiguous = find_contained(survivors, cfg)
            clean_idx = np.where(~ambiguous)[0]
            amb_idx = np.where(ambiguous)[0]

            # TIER 1: keep the clean YOLO boxes as-is, no SAM
            boxes = [list(survivors[i, :4]) for i in clean_idx]
            confs = [float(survivors[i, 4]) for i in clean_idx]
            tiers = [1] * len(clean_idx)
            masks = [None] * len(clean_idx)

            # TIER 3: SAM only on the ambiguous boxes (box + point + negative prompts; area-guarded).
            # negatives drawn from ALL survivors for context.
            n_sam = n_fallback = 0
            if len(amb_idx) > 0:
                predictor.set_image(img_np)                                # expensive encode, once
                cropped, ac = [], []                                       # valid masks -> mask_dedup pool
                fb_boxes, fb_conf = [], []                                 # area-guard fallbacks -> raw YOLO box
                for i in amb_idx:
                    iou_row, ios_row = _iou_ios(survivors[i, :4], survivors[:, :4])
                    cm = crop_mask(sam_box_mask(predictor, survivors, i, iou_row, ios_row, cfg))
                    n_sam += 1
                    det = survivors[i, :4]
                    det_area = float((det[2] - det[0]) * (det[3] - det[1]))
                    # area guard: a mask whose bbox is much bigger than the detection box LEAKED
                    # (grabbed neighbours / background) -> drop the mask, fall back to the raw YOLO box
                    leaked = cm is None or (cfg.max_area_ratio > 0 and
                             (cm[0][2] - cm[0][0]) * (cm[0][3] - cm[0][1]) > cfg.max_area_ratio * det_area)
                    if leaked:
                        fb_boxes.append([float(det[0]), float(det[1]), float(det[2]), float(det[3])])
                        fb_conf.append(float(survivors[i, 4]))
                        n_fallback += 1
                    else:
                        cropped.append(cm)
                        ac.append(float(survivors[i, 4]))
                kept = mask_dedup(cropped, ac, cfg.decide_mask_ios)         # mask-overlap split-vs-keep
                for k in kept:
                    (x1, y1, x2, y2), _ = cropped[k]
                    boxes.append([float(x1), float(y1), float(x2), float(y2)])
                    confs.append(ac[k]); tiers.append(3); masks.append(cropped[k])
                # area-guard fallbacks: keep the raw YOLO box (no mask) so the head isn't lost
                for b, c in zip(fb_boxes, fb_conf):
                    boxes.append(b); confs.append(c); tiers.append(3); masks.append(None)

            save_surgical(name, H, W, boxes, confs, tiers, masks, dirs, cfg.save_viz, img_np)
            print(f"  {name}: raw {len(preds)} -> tier2 {len(survivors)} "
                  f"-> tier1 {len(clean_idx)} clean + tier3 {len(amb_idx)} ambiguous "
                  f"(SAM {n_sam}, leaked->box {n_fallback}) -> {len(boxes)} heads")
    print("\nDone.")


@hydra.main(version_base=None, config_path="../../../configs/mask_generation", config_name="sahi_surgical_dedup")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
