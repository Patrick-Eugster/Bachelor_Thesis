"""Builds the resolution-1-vs-resolution-2 zoom figure for the reconstruction
Results (fig:recon-res). Crops the same scene region from a plot_461 test render
trained at resolution 1 and at resolution 2, and shows them at equal display size so
the detail loss at resolution 2 is visible (the metric numbers are not comparable
across resolutions because each is scored against its own-resolution ground truth).

The resolution-2 render is half the linear size, so its crop uses half the
coordinates and is upscaled (bicubic) to the resolution-1 crop size for a fair
same-size comparison. Output: thesis/figures/recon_res_zoom.png.

Run: python src/analysis/make_res_zoom.py
"""
from PIL import Image, ImageDraw, ImageFont

PLOT = "results/reconstruction/fip/plot_461/vanilla_3dgs"
R1 = f"{PLOT}/res1_absgrad_15k/test/ours_15000/renders/00005.png"
R2 = f"{PLOT}/res2_absgrad_15k/test/ours_15000/renders/00005.png"
OUT = "thesis/figures/recon_res_zoom.png"

# crop region in resolution-1 pixels (x0, y0, width, height); res2 uses half of these.
# hard zoom on the bottom-left corner (image is 4095x2996)
X0, Y0, CW, CH = 0, 2636, 360, 360

GAP = 12
LABEL_H = 34


def label_strip(panel, label):
    """Adds a white label strip with centered text on top of a panel image."""
    out = Image.new("RGB", (panel.width, panel.height + LABEL_H), "white")
    out.paste(panel, (0, LABEL_H))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    w = draw.textlength(label, font=font)
    draw.text(((panel.width - w) / 2, 10), label, fill="black", font=font)
    return out


def main():
    """Crops the same region from both renders (res2 upscaled to match) and composes
    them side by side at equal display size."""
    im1 = Image.open(R1).convert("RGB").crop((X0, Y0, X0 + CW, Y0 + CH))
    # res2 render is half-size, so halve the crop box then upscale back to CW x CH
    im2 = Image.open(R2).convert("RGB").crop((X0 // 2, Y0 // 2, X0 // 2 + CW // 2, Y0 // 2 + CH // 2))
    im2 = im2.resize((CW, CH), Image.BICUBIC)

    left = label_strip(im1, "Resolution 1")
    right = label_strip(im2, "Resolution 2")
    canvas = Image.new("RGB", (CW * 2 + GAP, CH + LABEL_H), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (CW + GAP, 0))
    canvas.save(OUT)
    print(f"wrote {OUT}  ({canvas.size[0]}x{canvas.size[1]}), res1-crop=({X0},{Y0},{CW},{CH})")


if __name__ == "__main__":
    main()
