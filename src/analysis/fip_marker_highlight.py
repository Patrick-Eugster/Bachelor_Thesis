"""Draws a bright highlight around the marker box that gets lifted into 3D, on the full-frame detector
and SAM visualizations for the three plot_466 views used in the Markers appendix. The marker-box
coordinates come from the fipseg_yv5_1280_c35 kept boxes matched to the reprocessed marker centers. The
same box rectangle is highlighted on both the yolo_vis and sam_vis image so the reader's eye finds the
marker among the hundreds of boxes, while the full frame is kept. Also saves a tight crop of each box so
its printed confidence can be read. Overwrites the images in thesis/figures/marker_fp/. Run from repo root."""
import numpy as np
import cv2
from PIL import Image

SRC = "results/mask_generation/fip/plot_466/yolo_sam_v1/fipseg_yv5_1280_c35"
OUT = "thesis/figures/marker_fp"
CROP = "docs/analysis_results/appendix_marker_fp/confcrop"
# view -> the marker box (x0,y0,x1,y1) found earlier (kept box on the marker plate)
BOXES = {
    "FPWW036_SR0466_FIP2_cam_02": (1960, 924, 2493, 1328),
    "FPWW036_SR0466_FIP2_cam_05": (672, 1327, 975, 1856),
    "FPWW036_SR0466_FIP2_cam_06": (476, 1377, 761, 1918),
}
PAD = 22
HL = (255, 0, 255)   # magenta highlight (BGR)


OUT_WIDTH = 1600   # downsample the full-frame 4096px vis to this width — plenty for a half-page panel


def highlight(src_jpg, box, out_jpg):
    """Draws a thick magenta rectangle around the box, downsamples to OUT_WIDTH, and writes it out."""
    img = cv2.imread(src_jpg)
    x0, y0, x1, y1 = box
    cv2.rectangle(img, (x0 - PAD, y0 - PAD), (x1 + PAD, y1 + PAD), HL, 18)
    h, w = img.shape[:2]
    img = cv2.resize(img, (OUT_WIDTH, round(h * OUT_WIDTH / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_jpg, img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    import os
    os.makedirs(CROP, exist_ok=True)
    for stem, box in BOXES.items():
        for kind in ("yolo_vis", "sam_vis"):
            tag = "yolo_boxes" if kind == "yolo_vis" else "sam_masks"
            highlight(f"{SRC}/{kind}/{stem}.jpg", box, f"{OUT}/{stem}_{tag}.jpg")
        # crop the box region from the yolo vis so the printed confidence is legible
        img = cv2.imread(f"{SRC}/yolo_vis/{stem}.jpg")
        x0, y0, x1, y1 = box
        c = img[max(0, y0 - 60):y1 + 30, max(0, x0 - 30):x1 + 30]
        cv2.imwrite(f"{CROP}/{stem}_boxcrop.jpg", c)
    print(f"highlighted 6 images in {OUT}/ ; conf crops in {CROP}/")


if __name__ == "__main__":
    main()
