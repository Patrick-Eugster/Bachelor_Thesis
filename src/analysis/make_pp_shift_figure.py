"""Builds the principal-point pixel-shift figure for the thesis Reconstruction results.

Takes one held-out test view of a FIP plot, zooms hard on the white center dot of a
coded marker (the sharpest fiducial in the scene), and shows it three ways side by side:
ground truth, correction OFF (paper_bench_30k), correction ON (paper_bench_30k_pp).
A red crosshair is pinned to the GT dot center, so the OFF dot visibly sits ~7 px to the
side while the ON dot returns to the crosshair. Also prints the measured sub-pixel offset
of each render's dot vs the GT dot, which is the number quoted in the figure caption.

Reproduces thesis/figures/pp_zoom_shift.png. Run from the repo root:
    python src/analysis/make_pp_shift_figure.py                          # default: plot 467, view 0
    python src/analysis/make_pp_shift_figure.py --plot 462 --view 1      # different plot / test view
    python src/analysis/make_pp_shift_figure.py --dotx 1378 --doty 1916  # point at another marker dot
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# the two experiment folders are the pixel-shift A/B: same 30k diffgs run, flag off vs on
EXP_OFF = "paper_bench_30k"
EXP_ON = "paper_bench_30k_pp"
BASE = "results/reconstruction/fip/plot_{plot}/vanilla_3dgs/{exp}/test/ours_30000/{kind}/{view:05d}.png"


def load(plot, exp, kind, view, mode):
    """Loads one test image (kind is 'gt' or 'renders') as a numpy array in the given PIL mode."""
    path = BASE.format(plot=plot, exp=exp, kind=kind, view=view)
    return np.asarray(Image.open(path).convert(mode)).astype(np.float32)


def dot_centroid(gray, cx, cy, win=18):
    """Brightness-weighted centroid of the marker's center dot inside a small window.
    Keeps only the brightest core (>60% of peak) so the surrounding dark disc is ignored."""
    p = gray[cy - win:cy + win, cx - win:cx + win].copy()
    p -= p.min()
    p[p < 0.6 * p.max()] = 0
    ys, xs = np.mgrid[0:p.shape[0], 0:p.shape[1]]
    m = p.sum()
    return cx - win + (xs * p).sum() / m, cy - win + (ys * p).sum() / m


def make_figure(plot, view, dot_xy, out_path, half=40, up=10, grid_step=5):
    """Crops around the GT dot, upscales nearest-neighbor, overlays a source-pixel grid +
    a red crosshair fixed at the GT dot center, and saves the 3-panel GT/OFF/ON figure."""
    # the approximate dot location is refined to a sub-pixel centroid on each arm
    gt_g = load(plot, EXP_ON, "gt", view, "L")
    off_g = load(plot, EXP_OFF, "renders", view, "L")
    on_g = load(plot, EXP_ON, "renders", view, "L")
    gx, gy = dot_centroid(gt_g, *dot_xy)
    ox, oy = dot_centroid(off_g, *dot_xy)
    nx, ny = dot_centroid(on_g, *dot_xy)
    print(f"GT dot   {gx:.2f} {gy:.2f}")
    print(f"OFF dot  {ox:.2f} {oy:.2f}  (offset {ox - gx:+.2f}, {oy - gy:+.2f})")
    print(f"ON  dot  {nx:.2f} {ny:.2f}  (offset {nx - gx:+.2f}, {ny - gy:+.2f})")

    gt = load(plot, EXP_ON, "gt", view, "RGB").astype(np.uint8)
    off = load(plot, EXP_OFF, "renders", view, "RGB").astype(np.uint8)
    on = load(plot, EXP_ON, "renders", view, "RGB").astype(np.uint8)

    ox0, oy0 = round(gx) - half, round(gy) - half  # crop origin in source pixels
    box = (ox0, oy0, ox0 + 2 * half, oy0 + 2 * half)

    def zoom(a):
        im = Image.fromarray(a).crop(box)
        return im.resize((2 * half * up, 2 * half * up), Image.NEAREST)

    panels = [zoom(gt), zoom(off), zoom(on)]
    labels = ["ground truth", "correction off", "correction on"]
    w = h = 2 * half * up
    gap, lab = 16, 42
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except OSError:
        fnt = ImageFont.load_default()

    cvs = Image.new("RGB", (w * 3 + gap * 2, h + lab), "white")
    crx, cry = (gx - ox0) * up, (gy - oy0) * up  # GT dot center in upscaled local coords
    for j, im in enumerate(panels):
        x0 = j * (w + gap)
        cvs.paste(im, (x0, lab))
        d = ImageDraw.Draw(cvs)
        # source-pixel grid (one line every grid_step original pixels)
        for k in range(0, 2 * half + 1, grid_step):
            d.line([(x0 + k * up, lab), (x0 + k * up, lab + h)], fill=(255, 255, 255), width=1)
            d.line([(x0, lab + k * up), (x0 + w, lab + k * up)], fill=(255, 255, 255), width=1)
        # red crosshair fixed at the GT dot center, same absolute coords in all three panels
        d.line([(x0 + crx, lab), (x0 + crx, lab + h)], fill=(255, 30, 30), width=3)
        d.line([(x0, lab + cry), (x0 + w, lab + cry)], fill=(255, 30, 30), width=3)
        tw = d.textlength(labels[j], font=fnt)
        d.text((x0 + (w - tw) / 2, 7), labels[j], fill="black", font=fnt)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cvs.save(out_path)
    print("saved", out_path, cvs.size)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default="467", help="FIP plot number")
    ap.add_argument("--view", type=int, default=0, help="test view index (00000.png = 0)")
    # approximate pixel location of a marker's center dot in the full-res image;
    # the default is the center marker of plot_467 view 0, refined to sub-pixel below
    ap.add_argument("--dotx", type=int, default=1378)
    ap.add_argument("--doty", type=int, default=1916)
    ap.add_argument("--out", default="thesis/figures/pp_zoom_shift.png")
    args = ap.parse_args()
    make_figure(args.plot, args.view, (args.dotx, args.doty), args.out)
