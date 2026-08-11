"""Builds the finished ground-truth figure for the thesis appendix: a phone frame with every
per-head GT mask drawn as a colored overlay, full frame beside a zoom. Reads the per-head instance
map (set0_instances.png, one integer id per head) written by the point-prompt labeling tool."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

STEM = "IMG_20250715_153912"
BASE = "input_plots/phone/field_A/20250715"
IMG = f"{BASE}/input/{STEM}.jpg"
INST = f"{BASE}/manual_label/{STEM}_sets/set0_instances.png"
OUT = "thesis/figures/maskgen_gt_example.png"
CROP = (1450, 700, 2650, 1600)   # dense central region, aspect ~1.33 to match the full frame
ALPHA = 0.5


def color_overlay(img, inst):
    """Blends a distinct random color over each head's pixels, leaving the background untouched."""
    ids = np.unique(inst)
    ids = ids[ids > 0]
    rng = np.random.default_rng(0)                       # fixed seed → same colors every run
    lut = np.zeros((int(inst.max()) + 1, 3), np.float32)
    lut[ids] = rng.uniform(60, 255, size=(len(ids), 3))  # avoid too-dark colors so masks read on the canopy
    col = lut[inst]                                       # per-pixel color, black where background
    fg = (inst > 0)[..., None]
    out = np.where(fg, (1 - ALPHA) * img + ALPHA * col, img)
    return out.astype(np.uint8), len(ids)


def draw(ax, img, crop=None, ylabel=None):
    """Draws one panel; crop=(x0,y0,x1,y1) zooms via axis limits. The label is written
    vertically on the left so it costs no vertical space."""
    ax.imshow(img)
    if crop is not None:
        x0, y0, x1, y1 = crop
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    img = np.array(Image.open(IMG).convert("RGB"))
    inst = np.array(Image.open(INST))                    # uint16 instance-label map
    over, n = color_overlay(img, inst)

    # stacked: full frame big on top, a smaller centered zoom below (so the full frame reads well)
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.62], hspace=0.06)
    ax_full = fig.add_subplot(gs[0])
    draw(ax_full, over, crop=None, ylabel="Full frame")
    x0, y0, x1, y1 = CROP
    ax_full.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                edgecolor="white", linewidth=1.8, linestyle="--"))
    # zoom sits in the middle third of the bottom row so it is narrower than the full frame
    gs_b = gs[1].subgridspec(1, 3, width_ratios=[1, 2.6, 1])
    ax_zoom = fig.add_subplot(gs_b[1])
    draw(ax_zoom, over, crop=CROP, ylabel="Zoom")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}  ({n} heads)")


if __name__ == "__main__":
    main()
