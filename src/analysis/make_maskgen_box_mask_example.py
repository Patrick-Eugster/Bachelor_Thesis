"""Builds the pedagogical box + binary-mask + overlay figure for the Methods
mask-generation section (Figure fig:maskgen-box-mask). Uses one FIP plot_461
image, where heads barely overlap so a plain binary union still reads as one
blob per head. Three stacked panels: the existing yolo_vis boxes (blue kept /
red below-threshold, with confidence labels), the SAM binary masks white on
black, and those masks tinted over the crop. Panel labels are baked in."""
import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

STEM = "FPWW036_SR0461_1_FIP2_cam_01"
BASE = "results/mask_generation/fip/plot_461/yolo_sam_v1/fipseg_yv5_1280_c35"
IMG = f"input_plots/fip/plot_461/images/{STEM}.png"
CROP = (0, 698, 1150, 1448)  # left edge, ~300px above centre, marker-free
LABELS = ["YOLO boxes", "SAM masks", "Masks over image"]
OUTD = "docs/analysis_results/maskgen_box_mask_example"
FIG = "thesis/figures/maskgen_box_mask.jpg"


def label(im, text):
    """Draws a white label on a black tab in the top-left of a panel."""
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    tb = dr.textbbox((14, 12), text, font=font)
    dr.rectangle([tb[0] - 10, tb[1] - 8, tb[2] + 10, tb[3] + 8], fill=(0, 0, 0))
    dr.text((14, 12), text, fill=(255, 255, 255), font=font)
    return im


def main():
    """Cut the crop, take the yolo_vis boxes as the top panel, union the kept
    heads' SAM masks for the binary and overlay panels, label each, stack."""
    x0, y0, x1, y1 = CROP
    b = torch.load(f"{BASE}/bboxes/{STEM}.pt", weights_only=False).numpy()
    im = np.array(Image.open(IMG).convert("RGB"))
    H, W = im.shape[:2]

    cx = (b[:, 0] + b[:, 2]) / 2
    cy = (b[:, 1] + b[:, 3]) / 2
    inb = np.where((cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1))[0]

    uni = np.zeros((H, W), bool)
    for i in inb:
        uni |= np.array(Image.open(f"{BASE}/masks/{STEM}_{i:03d}.png")) > 127
    sub = uni[y0:y1, x0:x1]
    crop = im[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]

    # top panel: the existing yolo_vis boxes (blue kept, red below threshold)
    p1 = np.array(Image.open(f"{BASE}/yolo_vis/{STEM}.jpg").crop((x0, y0, x1, y1)).convert("RGB"))
    # middle panel: binary union of the kept heads' masks
    p2 = np.zeros((ch, cw, 3), np.uint8)
    p2[sub] = 255
    # bottom panel: masks tinted over the crop
    p3 = crop.astype(float)
    p3[sub] = 0.55 * p3[sub] + 0.45 * np.array([60.0, 170.0, 255.0])
    p3 = p3.clip(0, 255).astype(np.uint8)

    panels = [np.array(label(Image.fromarray(p), t)) for p, t in zip((p1[:ch, :cw], p2, p3), LABELS)]

    gap = 18
    stack = np.full((ch * 3 + gap * 2, cw, 3), 255, np.uint8)
    for i, p in enumerate(panels):
        stack[i * (ch + gap):i * (ch + gap) + ch] = p

    os.makedirs(OUTD, exist_ok=True)
    Image.fromarray(stack).save(f"{OUTD}/stack_boxes_binary_overlay.png")
    Image.fromarray(stack).save(FIG, quality=92)
    print("heads", len(inb), "panel", cw, ch, "-> saved", FIG)


if __name__ == "__main__":
    main()
