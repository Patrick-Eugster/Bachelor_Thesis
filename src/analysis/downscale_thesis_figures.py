"""Downscales a couple of oversized thesis figure PNGs to 300 dpi at their render width.

The mask-gen result figures were saved as ~2100 px RGBA screenshots (~6-8 MB each), which is
well above the 300 dpi they render at across the text column. This shrinks the long side to
1850 px (true 300 dpi at ~6.1 in), drops the unused alpha channel, and re-compresses the PNG
losslessly. Originals are copied to figures/_highres_originals/ first so nothing is lost, and
the in-place files keep their names so main.tex needs no change. Run from the thesis dir:
    python ../src/analysis/downscale_thesis_figures.py
"""
import os
import shutil

from PIL import Image

FIG_DIR = "figures"
BACKUP = os.path.join(FIG_DIR, "_highres_originals")
LONG_SIDE = 1850  # ~300 dpi at the ~6.1 in text column
TARGETS = ["maskgen_inputsize_461.png", "maskgen_detector_461.png"]


def downscale(name):
    """Backup then shrink one figure to LONG_SIDE, RGBA->RGB, max PNG compression."""
    src = os.path.join(FIG_DIR, name)
    os.makedirs(BACKUP, exist_ok=True)
    backup = os.path.join(BACKUP, name)
    if not os.path.exists(backup):  # never overwrite an existing pristine backup
        shutil.copy2(src, backup)

    before = os.path.getsize(src)
    im = Image.open(src)
    w, h = im.size
    # flatten alpha onto white so the box overlays keep their colors
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")

    scale = LONG_SIDE / max(w, h)
    if scale < 1.0:
        im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    im.save(src, "PNG", optimize=True, compress_level=9)

    after = os.path.getsize(src)
    print(f"{name}: {w}x{h} -> {im.size[0]}x{im.size[1]}  "
          f"{before/1048576:.1f} MB -> {after/1048576:.1f} MB")


if __name__ == "__main__":
    for t in TARGETS:
        downscale(t)
