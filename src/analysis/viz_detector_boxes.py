"""Builds the qualitative detector figure for the thesis appendix: plot 461's one labeled image
with the kept detections of the three setups (YOLOv5 @1280, SAHI IoS, YOLO11 @3008), full frame
over a zoom. Boxes are drawn from bboxes/*.pt (kept boxes only). No GT here --- it is shown in the
input-size figure. Same crop region and colors as viz_inputsize_boxes for consistency."""
import os
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
OUT = "thesis/figures/maskgen_detector_461.png"
DET_COLOR = "#00e5ff"
LW = 0.6
CROP = (2100, 1950, 3400, 2901)   # same window as the input-size figure

# (title, bboxes path) for the three setups
SETUPS = [
    ("YOLOv5 (1280 px)", f"results/mask_generation/fip/{PLOT}/yolo_sam_v1/fip_imgsz_1280/bboxes/{STEM}.pt"),
    ("SAHI, IoS (1280 px tiles)", f"results/mask_generation/fip/{PLOT}/sahi_yolo_sam/fip_sahi_ios/bboxes/{STEM}.pt"),
    ("YOLO11 (3008 px)", f"results/mask_generation/fip/{PLOT}/yolo11_sam/fip_yolo11/bboxes/{STEM}.pt"),
]


def draw(ax, img, boxes, lw, crop=None, title=None):
    """Draws one panel: the image plus its boxes; crop=(x0,y0,x1,y1) zooms via axis limits."""
    ax.imshow(img)
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=DET_COLOR, linewidth=lw))
    if crop is not None:
        x0, y0, x1, y1 = crop
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    if title:
        ax.set_title(title, fontsize=18)   # large in-figure so it stays readable after the page downscale
    ax.set_xticks([]); ax.set_yticks([])


def main():
    img = np.array(Image.open(IMG).convert("RGB"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))   # a touch taller so the larger titles don't shrink the panels
    counts = []
    for col, (title, path) in enumerate(SETUPS):
        b = torch.load(path, weights_only=True).numpy()
        counts.append(len(b))
        draw(axes[0, col], img, b, LW, crop=None, title=f"{title}\n{len(b)} boxes")   # two lines so the wide title fits its panel
        draw(axes[1, col], img, b, 1.1, crop=CROP, title=None)
        x0, y0, x1, y1 = CROP
        axes[0, col].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=0.8, linestyle="--"))
    axes[0, 0].set_ylabel("full frame", fontsize=22)
    axes[1, 0].set_ylabel("zoom", fontsize=22)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}  (boxes: {dict(zip([s[0] for s in SETUPS], counts))})")


if __name__ == "__main__":
    main()
