"""Builds the qualitative AbsGS-vs-default zoom figure for the reconstruction
Results: crops the same awn-rich region from a FIP test render produced with default
densification and with AbsGS, and composes them side by side with labels. Supports
Overleaf review comment #18 (show the recovered fine detail, not only the metrics).

Both renders are the same held-out test camera of plot_461, so the crop region lines
up pixel for pixel. Output: thesis/figures/recon_absgrad_zoom.png.

Run: python src/analysis/make_absgrad_zoom.py
"""
from PIL import Image, ImageDraw, ImageFont

PLOT = "results/reconstruction/fip/plot_461/vanilla_3dgs"
DEFAULT = f"{PLOT}/test_gsplat_full/test/ours_30000/renders/00000.png"
ABSGRAD = f"{PLOT}/test_absgrad/test/ours_30000/renders/00000.png"
OUT = "thesis/figures/recon_absgrad_zoom.png"

# crop region in original-image pixels (x0, y0, width, height); placed on the region
# where the two renders differ most (measured render-vs-render, lower-centre awns)
X0, Y0, CW, CH = 2300, 2420, 340, 340

GAP = 20          # white gap between the two panels
LABEL_H = 60      # height of the label strip on top of each panel


def labeled_crop(path, label):
    """Crops the region from a render and adds a white label strip on top."""
    img = Image.open(path).convert("RGB").crop((X0, Y0, X0 + CW, Y0 + CH))
    panel = Image.new("RGB", (CW, CH + LABEL_H), "white")
    panel.paste(img, (0, LABEL_H))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except OSError:
        font = ImageFont.load_default()
    w = draw.textlength(label, font=font)
    draw.text(((CW - w) / 2, 8), label, fill="black", font=font)
    return panel


def main():
    """Composes the two labeled crops into one side-by-side figure."""
    left = labeled_crop(DEFAULT, "Default densification")
    right = labeled_crop(ABSGRAD, "AbsGS")
    canvas = Image.new("RGB", (CW * 2 + GAP, CH + LABEL_H), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (CW + GAP, 0))
    canvas.save(OUT)
    print(f"wrote {OUT}  ({canvas.size[0]}x{canvas.size[1]}), crop=({X0},{Y0},{CW},{CH})")


if __name__ == "__main__":
    main()
