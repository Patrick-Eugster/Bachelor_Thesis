"""Crops an identical center-ish region from the phone recon qualitative images
(15k default render, 30k AbsGS render, GT) so the three can be compared at the
same zoom. The crop is horizontally centered and shifted 175 px up from the
vertical center, per the chosen region for the densification/sharpness figure.
Output goes next to the full frames in docs/analysis_results/phone_recon_qualitative/."""
from pathlib import Path
from PIL import Image

DEST = Path("docs/analysis_results/phone_recon_qualitative")
SHIFT_UP = 175      # pixels above the vertical center
CROP_W = 1280       # wide rectangle (2:1) so three crops stack down one page
CROP_H = 640

# the three unedited full frames (same held-out view 00004, A/0715 opencv)
srcs = {
    "gt":          DEST / "A0715_00004_gt.png",
    "15k_default": DEST / "A0715_00004_render_15k_default.png",
    "30k_absgs":   DEST / "A0715_00004_render_30k_absgs.png",
}

# all three must share the same size so one absolute box is comparable
sizes = {k: Image.open(p).size for k, p in srcs.items()}
assert len(set(sizes.values())) == 1, f"size mismatch: {sizes}"
W, H = next(iter(sizes.values()))

cx, cy = W // 2, H // 2 - SHIFT_UP
box = (cx - CROP_W // 2, cy - CROP_H // 2, cx + CROP_W // 2, cy + CROP_H // 2)
print(f"image size {W}x{H}, crop box {box} ({CROP_W}x{CROP_H}, centered, {SHIFT_UP}px up)")

for k, p in srcs.items():
    crop = Image.open(p).crop(box)
    out = DEST / f"A0715_00004_zoom_{k}.png"
    crop.save(out)
    print(f"wrote {out}")
