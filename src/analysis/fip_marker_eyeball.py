"""Makes eyeball crops around every calibration marker on the 7 FIP GT images, with each detector's kept
boxes drawn, so a human can check directly whether YOLOv5 or YOLO11 puts a box on a marker. For each
plot's GT-labeled image we take the marker centers from marker_projections.csv, crop a window around each
marker, draw the kept YOLOv5-1280 boxes in green and the kept YOLO11 boxes in blue, and mark the marker
center (red dot) with the R=200 proximity circle used in the count. One PNG per marker goes to
docs/analysis_results/marker_check/. Run from repo root."""
import csv
import glob
import os
import numpy as np
import cv2
import torch
from PIL import Image

PLOTS = [f"plot_{n}" for n in range(461, 468)]
MARKER_CSV = "demoanlage2025_v0/FIP_single_row_exp_2024_reprocessed/{plot}/marker_projections.csv"
BOX = {
    "yolov5": ("results/mask_generation/fip/{plot}/yolo_sam_v1/fip_imgsz_1280/bboxes_with_conf", (0, 220, 0)),
    "yolo11": ("results/mask_generation/fip/{plot}/yolo11_sam/yolo11_eval/bboxes_with_conf", (255, 0, 0)),
}
KEEP = 0.35
R = 200
WIN = 380      # half-size of the crop window around each marker
OUTDIR = "docs/analysis_results/marker_check"


def gt_stem(plot):
    return os.path.splitext(os.path.basename(glob.glob(f"input_plots/fip/{plot}/manual_label/*.txt")[0]))[0]


def kept_boxes(plot, stem, d):
    f = f"{d.format(plot=plot)}/{stem}.pt"
    if not os.path.exists(f):
        return np.zeros((0, 5))
    t = torch.load(f, weights_only=True).numpy()
    return t[t[:, 4] >= KEEP]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    n = 0
    for plot in PLOTS:
        stem = gt_stem(plot)
        img = cv2.cvtColor(np.array(Image.open(f"input_plots/fip/{plot}/images/{stem}.png").convert("RGB")), cv2.COLOR_RGB2BGR)
        markers = [(float(r["X"]), float(r["Y"]), r["Marker"])
                   for r in csv.DictReader(open(MARKER_CSV.format(plot=plot))) if r["Camera"] == stem]
        boxes = {name: kept_boxes(plot, stem, d) for name, (d, _) in BOX.items()}
        for mx, my, mname in markers:
            canvas = img.copy()
            for name, (_, color) in BOX.items():
                for x1, y1, x2, y2, c in boxes[name]:
                    cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.circle(canvas, (int(mx), int(my)), R, (0, 0, 255), 3)      # proximity radius
            cv2.circle(canvas, (int(mx), int(my)), 12, (0, 0, 255), -1)    # marker center
            x0, y0 = max(0, int(mx) - WIN), max(0, int(my) - WIN)
            x1, y1 = int(mx) + WIN, int(my) + WIN
            crop = canvas[y0:y1, x0:x1]
            out = f"{OUTDIR}/{plot}_{stem}_{mname.replace(' ', '')}.png"
            cv2.imwrite(out, crop)
            n += 1
    print(f"wrote {n} marker crops to {OUTDIR}/")
    print("green = kept YOLOv5-1280 boxes | blue = kept YOLO11 boxes | red dot+circle = marker center (R=200)")


if __name__ == "__main__":
    main()
