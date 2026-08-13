"""Builds the qualitative SAM3 text-prompt figure for the appendix: the same phone frame
prompted with ``wheat'' at confidence 0.25, shown as a single whole-frame pass (top) and as
the four non-overlapping tiles laid back into their 2x2 positions (bottom). The tiles quarter
the frame, so placing t00/t01/t10/t11 in a grid reconstructs the same image at higher encode
resolution. Reads the probe's saved overlays (green = mask union, red = boxes, with the burned-in
per-panel count label). No re-inference --- just composits the existing overlays."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BASE = "results/sam3_text_probe/phone/field_A_20250715/IMG_20250715_153912/wheat"
FULL = f"{BASE}/full_frame/full__c25.jpg"
TILES = {"t00": (0, 0), "t01": (0, 1), "t10": (1, 0), "t11": (1, 1)}  # tag -> (row, col)
OUT = "thesis/figures/sam3_text_example.png"


def main():
    """Stacks the full-frame overlay over the 2x2 tiled overlays and saves the figure."""
    full = np.array(Image.open(FULL).convert("RGB"))

    fig = plt.figure(figsize=(9, 13.6))
    # top: full frame (one panel); bottom: 2x2 tiles with a thin gap so the seams show
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.08)
    ax_full = fig.add_subplot(gs[0])
    ax_full.imshow(full)
    ax_full.set_title("Full frame", fontsize=15, fontweight="bold")
    ax_full.set_xticks([]); ax_full.set_yticks([])

    gs_t = gs[1].subgridspec(2, 2, wspace=0.02, hspace=0.02)
    for tag, (r, c) in TILES.items():
        tile = np.array(Image.open(f"{BASE}/tiled/{tag}__c25.jpg").convert("RGB"))
        ax = fig.add_subplot(gs_t[r, c])
        ax.imshow(tile)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0 and c == 0:
            ax.set_title(r"Tiled ($2\times2$)", fontsize=15, fontweight="bold", loc="left")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
