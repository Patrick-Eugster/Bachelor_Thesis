"""Builds the qualitative input-size figure for the thesis appendix: plot 461's one GT image
with GT boxes and the YOLOv5 detections at each input size (640/1280/1920/2560), full frame.
Boxes are drawn from bboxes/*.pt (kept boxes only, so no conf text and no rejected boxes) and
the GT from manual_label. Full frame first; if the small heads are not visible we switch to a crop."""
import os
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

PLOT = "plot_461"
STEM = "FPWW036_SR0461_FIP2_cam_12"
IMG = f"input_plots/fip/{PLOT}/images/{STEM}.png"
GT_TXT = f"input_plots/fip/{PLOT}/manual_label/{STEM}.txt"
SIZES = [640, 1280, 1920]   # 2560 dropped from the figure: it adds only one box over 1920 (see caption)
BBOX = "results/mask_generation/fip/{plot}/yolo_sam_v1/fip_imgsz_{s}/bboxes/{stem}.pt"
OUT = "thesis/figures/maskgen_inputsize_461.png"

DET_COLOR = "#00e5ff"   # cyan — detections
GT_COLOR = "#ffe100"    # yellow — ground truth
LW_FULL = 0.8           # box line widths — a touch thicker than the old 0.6/1.1 so they read after downscale
LW_ZOOM = 1.3
# crop window for the zoom column (x0,y0,x1,y1), aspect matched to the full image (~1.367) so panels align.
# chosen below the marker plates, in pure vegetation with a mix of head sizes.
CROP = (2100, 1950, 3400, 2901)


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


def draw(ax, img, boxes, color, lw, crop=None):
    """Draws one panel: the image plus its boxes. crop=(x0,y0,x1,y1) zooms via axis limits."""
    ax.imshow(img)
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

    # rows: GT first, then the four sizes
    rows = [(gt, GT_COLOR, f"Ground truth\n{len(gt)} boxes")]
    for s in SIZES:
        b = torch.load(BBOX.format(plot=PLOT, s=s, stem=STEM), weights_only=True).numpy()
        rows.append((b, DET_COLOR, f"{s} px\n{len(b)} boxes"))

    # one subject per row, full frame | zoom (same layout idea as the FIP/phone example figure).
    # the full frame is given a bit more width than the zoom; a left-side title names the row.
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 3.25 * nrows),
                             gridspec_kw={"width_ratios": [1.4, 1.05]})
    for i, (boxes, color, title) in enumerate(rows):
        full_ax, zoom_ax = axes[i, 0], axes[i, 1]
        draw(full_ax, img, boxes, color, LW_FULL, crop=None)
        draw(zoom_ax, img, boxes, color, LW_ZOOM, crop=CROP)
        x0, y0, x1, y1 = CROP
        full_ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    edgecolor="white", linewidth=1.6, linestyle="--"))
        # big readable row label to the left of the full frame
        full_ax.set_ylabel(title, fontsize=14, fontweight="bold", rotation=0,
                           ha="right", va="center", labelpad=14)
        if i == 0:                                 # column headers on the top row only
            full_ax.set_title("Full frame", fontsize=14)
            zoom_ax.set_title("Zoom", fontsize=14)

    fig.tight_layout(h_pad=0.4)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}  ({w}x{h}, GT={len(gt)}, sizes={[len(p[0]) for p in rows[1:]]})")


if __name__ == "__main__":
    main()
