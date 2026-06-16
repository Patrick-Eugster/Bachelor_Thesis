"""
Shared helpers for the box-comparison eval tools (eval_compare_3way.py and
eval_compare_nogt.py). These two functions are method-vs-method (does YOLO agree
with SAHI?), which is a different axis from eval_yolo_boxes.py's GT/TP-FP-FN world,
so they live here instead of being bolted onto that file.

  categorize_two_sets  — match two box sets → mutual / A-only / B-only
  draw_overlay         — draw several colored categories of boxes on one image

The actual matching primitives (compute_iou_matrix, match_boxes) are imported from
eval_yolo_boxes.py so there is a single source of truth for "what counts as the same box".
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# sibling import — when a compare script is run with
#   python src/mask_generation/evaluation/eval_compare_*.py
# Python puts this folder on sys.path, so eval_yolo_boxes resolves. Importing it is
# safe: its @hydra.main main() only runs under its own __main__ guard, not on import.
from eval_yolo_boxes import compute_iou_matrix, match_boxes, load_pred_boxes, load_pred_boxes_with_conf


def categorize_two_sets(boxes_a, boxes_b, iou_threshold):
    """Match two box sets against each other and split them into three groups:
    mutual (a box in A that matches a box in B), A-only, B-only. Returns a dict:
      'mutual': list of (a_idx, b_idx, iou)   — the agreeing pairs
      'a_only': list of a_idx                 — boxes only set A has
      'b_only': list of b_idx                 — boxes only set B has
    This is the agreement primitive: eval_compare_nogt uses it directly (A=YOLO, B=SAHI),
    and eval_compare_3way uses it to cross-match YOLO-FP vs SAHI-FP (shared-FP = mutual)."""
    boxes_a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)

    # compute_iou_matrix is (pred, gt) → here A plays "pred", B plays "gt", shape (Na, Nb).
    # match_boxes then gives: tp = agreeing pairs, fp = A rows that matched nothing (A-only),
    # fn = B cols that were never matched (B-only). Same greedy IoU>=thr logic as the GT eval.
    iou_mat = compute_iou_matrix(boxes_a, boxes_b)
    mutual, a_only, b_only = match_boxes(iou_mat, iou_threshold)

    return {
        'mutual': mutual,
        'a_only': a_only,
        'b_only': b_only,
    }


def load_boxes_and_conf(bbox_path, with_conf_path):
    """Load the good boxes (N,4) plus a confidence per box (N,). The good boxes in bboxes/*.pt are
    a bit-identical subset of the 5-col bboxes_with_conf/*.pt rows, so we look up each box's conf by
    exact coordinate match. Returns (boxes, confs); confs is None when the with-conf file is absent
    (older runs without only_labeled_images) → overlays then just skip the conf labels."""
    boxes = load_pred_boxes(bbox_path)
    if not os.path.exists(with_conf_path):
        return boxes, None
    wc = load_pred_boxes_with_conf(with_conf_path)
    if wc is None or len(wc) == 0:
        return boxes, np.full(len(boxes), np.nan, dtype=np.float32)
    # round to 3 dp so the dict key is stable (values are float32 from the same source; collisions improbable)
    cmap = {tuple(np.round(r[:4], 3)): float(r[4]) for r in wc}
    confs = np.array([cmap.get(tuple(np.round(b, 3)), np.nan) for b in boxes], dtype=np.float32)
    return boxes, confs


def fmt_conf_labels(confs):
    """Turn confidences into per-box label strings ('0.83'), empty string where the conf is missing
    (None/NaN). Returns None if confs is None so draw_overlay skips labels for that layer entirely."""
    if confs is None:
        return None
    labels = []
    for c in confs:
        try:
            cf = float(c)
        except (TypeError, ValueError):
            labels.append("")
            continue
        labels.append("" if np.isnan(cf) else f"{cf:.2f}")
    return labels


def _load_font(size):
    """Load a readable TTF font, falling back to PIL's default if it isn't installed
    (same approach as eval_yolo_boxes.save_match_visualization)."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_overlay(image_path, layers, out_path, line_width=4, legend=True, font_size=28, label_font_size=28):
    """Draw several categories of boxes on one image, each category in its own color, and save it.
    `layers` is an ordered dict mapping a label to (rgb_color, boxes) OR (rgb_color, boxes, box_labels),
    where boxes is an (N,4) array of [x1,y1,x2,y2] and the optional box_labels is a list of per-box
    strings (e.g. confidences '0.83'; empty string = no label). Per-box labels are drawn in WHITE
    text with a colored stroke, the same style as eval_yolo_boxes match_viz. Categories are drawn in
    insertion order (later on top). A semi-transparent corner legend is drawn unless legend=False."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    label_font = None   # loaded lazily, only if some layer actually carries per-box labels

    # 1. draw every category's boxes in its color (insertion order = z-order), with optional labels
    for label, spec in layers.items():
        color, boxes = spec[0], spec[1]
        box_labels = spec[2] if len(spec) > 2 else None
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            if box_labels is not None and i < len(box_labels) and box_labels[i]:
                if label_font is None:
                    label_font = _load_font(label_font_size)
                # white text + colored stroke above the box top-left — same look as match_viz
                draw.text((x1, max(0, y1 - label_font_size - 2)), box_labels[i],
                          fill=(255, 255, 255), font=label_font, stroke_width=2, stroke_fill=color)

    # 2. corner legend so a reader knows which color is which (with per-category counts)
    if legend:
        img = _draw_legend(img, layers, font_size)

    img.save(out_path, quality=92)
    print(f"  Overlay saved: {out_path}")


def _draw_legend(img, layers, font_size, panel_alpha=175):
    """Draw a SEMI-TRANSPARENT top-left legend panel so boxes underneath still show through:
    one row per category with a filled color swatch and 'label (count)' text. The panel is drawn
    on a separate RGBA overlay and alpha-composited (a plain RGB image can't hold alpha), with the
    swatches/text fully opaque so the colors read true. Returns a new RGB image with the legend on top."""
    font = _load_font(font_size)
    pad = 10                 # padding inside the panel
    gap = 8                  # vertical gap between rows
    swatch = font_size       # swatch is a square the height of the text
    row_h = font_size + gap

    # transparent overlay we draw the panel onto, then composite over the image
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    # build the rows first so we can size the background panel before drawing it
    rows = []
    max_text_w = 0
    for label, spec in layers.items():
        color, boxes = spec[0], spec[1]
        count = len(np.asarray(boxes, dtype=np.float32).reshape(-1, 4))
        text = f"{label} ({count})"
        # textbbox gives (x0,y0,x1,y1) of the rendered text → width = x1-x0
        bbox = od.textbbox((0, 0), text, font=font)
        max_text_w = max(max_text_w, bbox[2] - bbox[0])
        rows.append((color, text))

    panel_w = pad + swatch + pad + max_text_w + pad
    panel_h = pad + len(rows) * row_h + pad - gap   # no trailing gap after last row

    # white panel at partial alpha → the field + boxes under it stay visible; opaque black border
    od.rectangle([10, 10, 10 + panel_w, 10 + panel_h],
                 fill=(255, 255, 255, panel_alpha), outline=(0, 0, 0, 255), width=2)

    y = 10 + pad
    for color, text in rows:
        x = 10 + pad
        # swatch + text fully opaque so the category color and label read clearly
        od.rectangle([x, y, x + swatch, y + swatch], fill=tuple(color) + (255,), outline=(0, 0, 0, 255), width=1)
        od.text((x + swatch + pad, y), text, fill=(0, 0, 0, 255), font=font)
        y += row_h

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
