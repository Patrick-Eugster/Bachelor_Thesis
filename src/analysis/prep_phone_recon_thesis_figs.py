"""Prepares thesis-ready (downscaled JPG) versions of the phone recon qualitative
images from the unedited originals in docs/analysis_results/phone_recon_qualitative/.
Full frames are downscaled so the thesis PDF does not balloon; the 2:1 zoom crops
are kept at native width. Outputs land in thesis/figures/."""
from pathlib import Path
from PIL import Image

SRC = Path("docs/analysis_results/phone_recon_qualitative")
DST = Path("thesis/figures")
FULL_W = 2200   # downscaled width for the full frames (JPG)
QUALITY = 90

jobs = [
    # (source, output, target_width or None to keep native)
    ("A0715_00004_gt.png",                "phone_recon_full_gt.jpg",   FULL_W),
    ("A0715_00004_render_15k_default.png","phone_recon_full_15k.jpg",  FULL_W),
    ("A0715_00004_render_30k_absgs.png",  "phone_recon_full_30k.jpg",  FULL_W),
    ("A0715_00004_zoom_gt.png",           "phone_recon_zoom_gt.jpg",   None),
    ("A0715_00004_zoom_15k_default.png",  "phone_recon_zoom_15k.jpg",  None),
    ("A0715_00004_zoom_30k_absgs.png",    "phone_recon_zoom_30k.jpg",  None),
]

for src_name, out_name, target_w in jobs:
    im = Image.open(SRC / src_name).convert("RGB")
    if target_w and im.width > target_w:
        h = round(im.height * target_w / im.width)
        im = im.resize((target_w, h), Image.LANCZOS)
    out = DST / out_name
    im.save(out, "JPEG", quality=QUALITY)
    print(f"{out}  {im.size}  {out.stat().st_size/1e6:.2f} MB")
