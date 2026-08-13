"""Renders the SAM3 text-prompt appendix figure from the saved per-instance masks
(results/sam3_text_probe_rerun/*.npz), so each instance gets its OWN color instead of the
old green union. No GPU --- pure rendering, re-runnable freely to tweak colors/layout. Shows the
full-frame pass (top) over the four 2x2 tiles (bottom); the full frame returns a few huge regions,
the near tiles many small ones.

Run: python src/analysis/sam3_text_rerun_figure.py
"""
import colorsys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

IMG = "input_plots/phone/field_A/20250715/images/IMG_20250715_153912.jpg"
NPZ = "results/sam3_text_probe_rerun"
OUT = "thesis/figures/sam3_text_example.png"
ALPHA = 0.5
TILES = {"t00": (0, 0), "t01": (0, 1), "t10": (1, 0), "t11": (1, 1)}


def color_overlay(img_rgb, masks, seed=0):
    """Blends each instance mask in its own vivid color; returns (overlay, count). Colors come from
    a saturated HSV wheel so no instance takes on a wheat-brown that would hide it in the canopy."""
    out = img_rgb.astype(np.float32)
    rng = np.random.default_rng(seed)
    for m in masks:
        h, s, v = rng.uniform(0, 1), rng.uniform(0.75, 1.0), rng.uniform(0.9, 1.0)
        color = np.array(colorsys.hsv_to_rgb(h, s, v)) * 255
        sel = m > 0
        out[sel] = (1 - ALPHA) * out[sel] + ALPHA * color
    return out.astype(np.uint8), len(masks)


def panel_overlay(tag, full_img):
    """Loads one panel's masks and paints them over the matching crop of the frame."""
    d = np.load(os.path.join(NPZ, f"{tag}.npz"))
    masks, y0, x0, ph, pw = d["masks"], int(d["y0"]), int(d["x0"]), int(d["ph"]), int(d["pw"])
    crop = full_img[y0:y0 + ph, x0:x0 + pw]
    return color_overlay(crop, masks)


GAP = 26  # equal white gap (px) between tiles, both directions


def annotate_axes(ax, n):
    """Count label in the top-left of an axes (axes-fraction coords)."""
    ax.text(0.015, 0.97, f"{n} instances", transform=ax.transAxes, fontsize=12,
            va="top", ha="left", color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.6))


def annotate_at(ax, x, y, n):
    """Count label at pixel (x, y) in data coords, for the composited tile block."""
    ax.text(x, y, f"{n} instances", fontsize=12, va="top", ha="left", color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.6))


def tile_composite(full_img):
    """Lays the four tile overlays into one image with an equal GAP-px white border between them,
    so the row gap and column gap are identical by construction. Returns (canvas, label positions)."""
    ov = {tag: panel_overlay(tag, full_img) for tag in TILES}   # tag -> (overlay, count)
    th, tw = ov["t00"][0].shape[:2]
    canvas = np.full((2 * th + GAP, 2 * tw + GAP, 3), 255, np.uint8)
    labels = []
    for tag, (r, c) in TILES.items():
        y, x = r * (th + GAP), c * (tw + GAP)
        canvas[y:y + th, x:x + tw] = ov[tag][0]
        labels.append((x + int(0.015 * tw), y + int(0.03 * th), ov[tag][1]))
    return canvas, labels


def main():
    """Builds the stacked per-instance figure and saves it."""
    full_img = np.array(Image.open(IMG).convert("RGB"))

    fig = plt.figure(figsize=(9, 13.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.08)

    ov, n = panel_overlay("full", full_img)
    ax_full = fig.add_subplot(gs[0])
    ax_full.imshow(ov)
    ax_full.set_title("Full frame", fontsize=15, fontweight="bold")
    annotate_axes(ax_full, n)
    ax_full.set_xticks([]); ax_full.set_yticks([])

    canvas, labels = tile_composite(full_img)
    ax_t = fig.add_subplot(gs[1])
    ax_t.imshow(canvas)
    ax_t.set_title(r"Tiled ($2\times2$)", fontsize=15, fontweight="bold", loc="left")
    for x, y, n in labels:
        annotate_at(ax_t, x, y, n)
    ax_t.set_xticks([]); ax_t.set_yticks([])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
