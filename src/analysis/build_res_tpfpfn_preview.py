"""Draft a resolution comparison for the mask-generation Results: the per-head SAM2 masks at YOLOv5
@1280 versus @4032, stacked as two rows and colored by the instance match at IoU 0.5 --- TP green, FP
red, missed GT blue --- with NO merge/split classes (those are dropped for this figure). Each row is
cropped to the marker ROI (the bounding box of the ground-truth heads) so it is slightly zoomed in. The
matching reuses the mask-generation instance eval so the coloring matches the numbers in the grid. Draft
only: writes ONE preview PNG into the review folder, reads results/ + input_plots/ read-only.

    python src/analysis/build_res_tpfpfn_preview.py
"""
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mask_generation", "evaluation"))
from eval_masks_instance import (load_gt_instances, load_pred_masks,   # noqa: E402
                                 compute_iou_matrix, match_instances)
from scipy.ndimage import find_objects  # noqa: E402

STEM = "IMG_20250715_153912"
LABEL_DIR = "input_plots/phone/field_A/20250715/manual_label"
IMG = f"input_plots/phone/field_A/20250715/images/{STEM}.jpg"
BASE = "results/mask_generation/phone/field_A/20250715/yolo_sam_v1"
ROWS = [("YOLOv5 @1280", "gt_head_sam2"), ("YOLOv5 @4032", "e2_ph_sam2")]
THR = 0.5
PAD = 60            # padding around the GT bounding box when cropping to the ROI
PANEL_W = 1800      # width each row is scaled to
SEP = 14
ROI_DIM = 0.45      # how much to darken outside the marker ROI (1.0 = no change)
C_TP, C_FP, C_FN = (0, 180, 0), (0, 0, 255), (255, 0, 0)   # BGR: green / red / blue (FIP convention)
C_EDGE = (25, 25, 25)   # dark outline drawn around every mask so individual instances stay separate
ALPHA = 0.65        # overlay opacity for the filled TP/FP masks
HATCH_PERIOD = 16   # FN blue stripes: one stripe every HATCH_PERIOD original pixels
HATCH_THICK = 6     # thickness of each FN stripe in original pixels
OUT = "docs/analysis_results/maskgen_phone_fig_review/res_tpfpfn_preview.png"
OUT_THESIS = "thesis/figures/maskgen_phone_res_tpfpfn.jpg"
THESIS_W = 1700     # width cap for the compressed thesis copy


def overlay(exp):
    """TP/FP/FN overlay for one config on the full frame (no merge/split), returned BGR."""
    gt_map, gt_ids, gt_areas = load_gt_instances(LABEL_DIR, STEM)
    preds = load_pred_masks(os.path.join(BASE, exp, "masks"), STEM, gt_map.shape)
    iou, _ = compute_iou_matrix(gt_map, gt_ids, gt_areas, preds)
    pairs = match_instances(iou, THR)
    matched_pred = {i for i, _ in pairs}
    matched_gt = {j for _, j in pairs}

    img = cv2.imread(IMG)
    if img.shape[:2] != gt_map.shape:
        img = cv2.resize(img, (gt_map.shape[1], gt_map.shape[0]))
    # 1) fill the TP predictions solid, then blend at ALPHA so the photo shows through
    over = img.copy()
    fp = np.zeros(gt_map.shape, bool)
    for i, p in enumerate(preds):
        x0, y0, x1, y1 = p["bbox"]
        if i in matched_pred:
            over[y0:y1, x0:x1][p["sub"]] = C_TP
        else:
            fp[y0:y1, x0:x1] |= p["sub"]
    out = cv2.addWeighted(over, ALPHA, img, 1 - ALPHA, 0)

    # 2) the two error classes as striped hatches on top, so where an FP and an FN overlap both stay
    #    visible as a crosshatch instead of one overriding the other: FP = VERTICAL red, FN = HORIZONTAL blue
    fn = np.zeros(gt_map.shape, bool)
    slices = find_objects(gt_map)
    for j, gid in enumerate(gt_ids):
        if j in matched_gt:
            continue
        sl = slices[gid - 1] if gid - 1 < len(slices) else None
        if sl is not None:
            fn[sl] |= gt_map[sl] == gid
    col_stripe = (np.arange(gt_map.shape[1]) % HATCH_PERIOD) < HATCH_THICK
    row_stripe = (np.arange(gt_map.shape[0]) % HATCH_PERIOD) < HATCH_THICK
    out[fp & col_stripe[None, :]] = C_FP
    out[fn & row_stripe[:, None]] = C_FN

    # 3) a dark outline around every mask (predictions and missed GT) so adjacent same-color masks
    #    read as separate instances instead of one blob
    for i, p in enumerate(preds):
        x0, y0, x1, y1 = p["bbox"]
        cnts, _ = cv2.findContours(p["sub"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c + [x0, y0] for c in cnts]
        cv2.drawContours(out, cnts, -1, C_EDGE if i in matched_pred else C_FP, 2)
    for j, gid in enumerate(gt_ids):
        if j in matched_gt:
            continue
        sl = slices[gid - 1] if gid - 1 < len(slices) else None
        if sl is None:
            continue
        sub = (gt_map[sl] == gid).astype(np.uint8)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c + [sl[1].start, sl[0].start] for c in cnts]
        cv2.drawContours(out, cnts, -1, C_FN, 2)
    return out, gt_map


def roi_mask(shape):
    """The marker-ROI region, recovered from a sam_vis image where everything outside the plot hull is
    painted flat grey (114). Non-grey pixels are inside the ROI; hole-filling turns them into one solid
    polygon. Same ROI for both configs (same frame), so any one sam_vis works."""
    sv = cv2.imread(os.path.join(BASE, ROWS[0][1], "sam_vis", f"{STEM}.jpg"))
    if sv.shape[:2] != shape:
        sv = cv2.resize(sv, (shape[1], shape[0]))
    inside = (np.abs(sv.astype(np.int16) - 114) > 15).any(axis=2)
    return binary_fill_holes(inside)


def roi_bbox(gt_map):
    """Bounding box of the GT heads (padded) --- the labeled region, i.e. the marker ROI."""
    ys, xs = np.nonzero(gt_map)
    return (max(0, xs.min() - PAD), max(0, ys.min() - PAD),
            min(gt_map.shape[1], xs.max() + PAD), min(gt_map.shape[0], ys.max() + PAD))


def label(img, text):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
    cv2.rectangle(img, (16, 16), (16 + w + 24, 16 + h + 28), (255, 255, 255), -1)
    cv2.putText(img, text, (28, 16 + h + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 4, cv2.LINE_AA)


def legend(img):
    """Bigger legend with black text on a white panel."""
    items = [("TP", C_TP), ("FP", C_FP), ("missed GT", C_FN)]
    x, y0, row = 20, 16 + 84, 62
    cv2.rectangle(img, (x - 8, y0 - 12), (x + 420, y0 + row * len(items) - 2), (255, 255, 255), -1)
    for k, (name, col) in enumerate(items):
        y = y0 + k * row
        cv2.rectangle(img, (x, y), (x + 56, y + 44), col, -1)
        cv2.rectangle(img, (x, y), (x + 56, y + 44), (25, 25, 25), 2)
        cv2.putText(img, name, (x + 72, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4, cv2.LINE_AA)


def scale_to_w(im, w):
    h = round(im.shape[0] * w / im.shape[1])
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)


def main():
    # crop box is shared across the two rows (same frame + ROI), taken from the first config's GT
    panels = []
    box, roi = None, None
    for i, (name, exp) in enumerate(ROWS):
        over, gt_map = overlay(exp)
        if roi is None:
            roi = roi_mask(gt_map.shape)
            box = roi_bbox(gt_map)
        over[~roi] = (over[~roi].astype(np.float32) * ROI_DIM).astype(np.uint8)   # dim outside the ROI
        x0, y0, x1, y1 = box
        crop = over[y0:y1, x0:x1].copy()
        label(crop, name)
        if i == 0:
            legend(crop)
        panels.append(scale_to_w(crop, PANEL_W))
    hsep = np.full((SEP, PANEL_W, 3), 255, np.uint8)
    stacked = np.vstack([panels[0], hsep, panels[1]])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, stacked)
    print(f"wrote {OUT}  ({stacked.shape[1]}x{stacked.shape[0]})")
    # compressed copy for the thesis (downscaled + jpg so it does not bloat the PDF)
    thesis = stacked
    if thesis.shape[1] > THESIS_W:
        h = round(thesis.shape[0] * THESIS_W / thesis.shape[1])
        thesis = cv2.resize(thesis, (THESIS_W, h), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(OUT_THESIS), exist_ok=True)
    cv2.imwrite(OUT_THESIS, thesis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {OUT_THESIS}  ({thesis.shape[1]}x{thesis.shape[0]})")
    print(f"roi crop box (orig px): x {box[0]}-{box[2]}, y {box[1]}-{box[3]}")


if __name__ == "__main__":
    main()
