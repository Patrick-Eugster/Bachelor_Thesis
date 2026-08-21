"""Builds the qualitative detector figure for the thesis Results: plot 461's one labeled image with the
detections of the three setups (YOLOv5 @1280, SAHI IoS, YOLO11 @3008). One row per setup, the full frame
on the left and a zoom of the marked region on the right (same layout as the input-size figure). Each box
is colored by whether it is a true positive (green), a false positive (red), or a missed GT head / false
negative (blue), matched to the ground truth at IoU 0.5 --- the same match rule as the box scores, so
SAHI's extra boxes over YOLO11 show up as mostly false. Boxes come from bboxes/*.pt (kept boxes only)."""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from PIL import Image

PLOT = "plot_461"
STEM = "FPWW036_SR0461_FIP2_cam_12"
IMG = f"input_plots/fip/{PLOT}/images/{STEM}.png"
GT_TXT = f"input_plots/fip/{PLOT}/manual_label/{STEM}.txt"
OUT = "thesis/figures/maskgen_detector_461.png"

C_TP = "#00c853"    # green — true positive
C_FP = "#ff1744"    # red   — false positive
C_FN = "#2979ff"    # blue  — missed GT head (false negative)
LW_FULL = 0.8
LW_ZOOM = 1.3
IOU_THR = 0.5
CROP = (2100, 1950, 3400, 2901)   # same window as the input-size figure

# (title, bboxes path) for the three setups
SETUPS = [
    ("YOLOv5\n1280 px", f"results/mask_generation/fip/{PLOT}/yolo_sam_v1/fip_imgsz_1280/bboxes/{STEM}.pt"),
    ("SAHI IoS\n1280 px", f"results/mask_generation/fip/{PLOT}/sahi_yolo_sam/fip_sahi_ios/bboxes/{STEM}.pt"),
    ("YOLO11\n3008 px", f"results/mask_generation/fip/{PLOT}/yolo11_sam/fip_yolo11/bboxes/{STEM}.pt"),
]


def load_gt(path, w, h):
    """Reads YOLO-format GT (class cx cy w h, normalized) and returns xyxy pixel boxes."""
    out = []
    for line in open(path):
        p = line.split()
        if len(p) < 5:
            continue
        _, cx, cy, bw, bh = (float(x) for x in p[:5])
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out, dtype=np.float32)


def classify(preds, gts, thr=IOU_THR):
    """Greedy-matches each predicted box to the best still-free GT box with IoU >= thr (the same rule
    the box scores use). Returns tp, fp (predictions) and fn (unmatched GT), all as xyxy arrays."""
    if len(preds) == 0:
        return np.empty((0, 4)), np.empty((0, 4)), gts.copy()
    if len(gts) == 0:
        return np.empty((0, 4)), preds.copy(), np.empty((0, 4))
    px1, py1, px2, py2 = (preds[:, k][:, None] for k in range(4))
    gx1, gy1, gx2, gy2 = (gts[:, k][None, :] for k in range(4))
    iw = np.clip(np.minimum(px2, gx2) - np.maximum(px1, gx1), 0, None)
    ih = np.clip(np.minimum(py2, gy2) - np.maximum(py1, gy1), 0, None)
    inter = iw * ih
    iou = inter / ((px2 - px1) * (py2 - py1) + (gx2 - gx1) * (gy2 - gy1) - inter + 1e-9)
    matched, tp, fp = set(), [], []
    for i in range(len(preds)):
        best_j, best = -1, thr
        for j in range(len(gts)):
            if j in matched:
                continue
            if iou[i, j] >= best:
                best, best_j = iou[i, j], j
        if best_j >= 0:
            matched.add(best_j); tp.append(preds[i])
        else:
            fp.append(preds[i])
    fn = [gts[j] for j in range(len(gts)) if j not in matched]
    return (np.array(tp).reshape(-1, 4), np.array(fp).reshape(-1, 4), np.array(fn).reshape(-1, 4))


def draw(ax, img, groups, lw, crop=None):
    """Draws one panel: the image plus each (boxes, color) group. crop=(x0,y0,x1,y1) zooms via limits."""
    ax.imshow(img)
    for boxes, color in groups:
        for x1, y1, x2, y2 in boxes:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=lw))
    if crop is not None:
        x0, y0, x1, y1 = crop
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)   # inverted y for image coords
    ax.set_xticks([]); ax.set_yticks([])


def main():
    img = np.array(Image.open(IMG).convert("RGB"))
    h, w = img.shape[:2]
    gt = load_gt(GT_TXT, w, h)

    # one row per setup, colored TP/FP/FN; label stacked vertically (name, size, then each count)
    rows = []
    for name, path in SETUPS:
        b = torch.load(path, weights_only=True).numpy()
        tp, fp, fn = classify(b, gt)
        rows.append(([(tp, C_TP), (fp, C_FP), (fn, C_FN)],
                     f"{name}\nTP {len(tp)}\nFP {len(fp)}\nFN {len(fn)}"))

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 3.35 * nrows),
                             gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.015, "hspace": 0.05})
    for i, (groups, title) in enumerate(rows):
        full_ax, zoom_ax = axes[i, 0], axes[i, 1]
        draw(full_ax, img, groups, LW_FULL, crop=None)
        draw(zoom_ax, img, groups, LW_ZOOM, crop=CROP)
        x0, y0, x1, y1 = CROP
        full_ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    edgecolor="white", linewidth=1.6, linestyle="--"))
        full_ax.set_ylabel(title, fontsize=13, fontweight="bold", rotation=0,
                           ha="right", va="center", labelpad=8)
        if i == 0:                                 # column headers on the top row only
            full_ax.set_title("Full frame", fontsize=14)
            zoom_ax.set_title("Zoom", fontsize=14)

    handles = [Line2D([0], [0], color=C_TP, lw=3, label="TP (correct)"),
               Line2D([0], [0], color=C_FP, lw=3, label="FP (false detection)"),
               Line2D([0], [0], color=C_FN, lw=3, label="FN (missed GT head)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.11, right=0.998, top=0.95, bottom=0.01, wspace=0.015, hspace=0.05)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}  (GT={len(gt)})")


if __name__ == "__main__":
    main()
