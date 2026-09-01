"""Redraw the Figure 4.1 full-frame images at full resolution with the red dashed zoom box burned in.

The thesis figure (fig:fip-vs-phone) draws the dashed box as a tikz overlay on downscaled frames, so the
saved figures/*.jpg are only ~2500-3000 px wide. For the presentation we want the FULL-resolution frames
with the same box. This reads the full-res sources (the pre-shrink backups / raw FIP frame), draws the box
using the exact fractional coordinates from main.tex, and writes them to docs/analysis_results/.

The tikz rectangle coords are fractions of the image, with y measured from the BOTTOM (tikz y-up). We flip
y to top-down for PIL. Read-only on the sources; writes only into docs/analysis_results/fig41_fullres/.
Run from repo root:  python -m src.analysis.draw_fig41_zoom_boxes
"""
import os
from PIL import Image, ImageDraw

OUT = "docs/analysis_results/fig41_fullres"

# (label, source path, (x0, y0_from_bottom, x1, y1_from_bottom)) — coords verbatim from main.tex
FRAMES = [
    ("fip_example",        "archive/thesis_figures_preshrink_backup/fip_example.png",          (0.801, 0.003, 0.998, 0.272)),
    ("phone_fieldA_example", "archive/thesis_figures_preshrink_backup/phone_fieldA_example.jpg", (0.403, 0.517, 0.597, 0.775)),
    ("phone_fieldD_example", "archive/thesis_figures_preshrink_backup/phone_fieldD_example.jpg", (0.403, 0.537, 0.597, 0.795)),
]


def dashed_rectangle(draw, box, width, dash=40, gap=28, color=(255, 0, 0)):
    """Draw a dashed rectangle (PIL has no native dashed stroke), stepping along each edge in dash+gap runs."""
    x0, y0, x1, y1 = box
    edges = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    for (ax, ay), (bx, by) in edges:
        length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        if length == 0:
            continue
        ux, uy = (bx - ax) / length, (by - ay) / length
        pos = 0.0
        while pos < length:
            seg = min(dash, length - pos)
            sx, sy = ax + ux * pos, ay + uy * pos
            ex, ey = ax + ux * (pos + seg), ay + uy * (pos + seg)
            draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
            pos += dash + gap


def main():
    os.makedirs(OUT, exist_ok=True)
    for label, src, (fx0, fyb0, fx1, fyb1) in FRAMES:
        im = Image.open(src).convert("RGB")
        w, h = im.size
        # flip the tikz bottom-up y to PIL top-down
        px0, px1 = fx0 * w, fx1 * w
        py0, py1 = (1 - fyb1) * h, (1 - fyb0) * h
        lw = max(4, round(w / 500))                       # line width scales with the frame
        draw = ImageDraw.Draw(im)
        dashed_rectangle(draw, (px0, py0, px1, py1), width=lw)
        out_path = os.path.join(OUT, f"{label}_boxed.png")
        im.save(out_path)
        print(f"{label:22s} {w}x{h}  box=({px0:.0f},{py0:.0f})-({px1:.0f},{py1:.0f})  -> {out_path}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
