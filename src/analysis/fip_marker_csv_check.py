"""Fast visual check of the marker_projections.csv positions against our GT images, for deciding whether
a CSV-based marker exclusion would line up. For each of the 7 FIP GT-labeled images it draws, on the full
frame (no crop), every marker center from the reprocessed marker_projections.csv as a dot plus the
exclusion circle that a CSV-radius eval would remove. If the circles sit on the plates across all 7, the
CSV is aligned to our images and the exclusion would work. Output to docs/analysis_results/marker_csv_check/.
Run from repo root."""
import csv
import glob
import os
import cv2
from PIL import Image
import numpy as np

PLOTS = [f"plot_{n}" for n in range(461, 468)]
MARKER_CSV = "demoanlage2025_v0/FIP_single_row_exp_2024_reprocessed/{plot}/marker_projections.csv"
R = 275          # candidate exclusion radius (px)
OUTDIR = "docs/analysis_results/marker_csv_check"


def gt_stem(plot):
    return os.path.splitext(os.path.basename(glob.glob(f"input_plots/fip/{plot}/manual_label/*.txt")[0]))[0]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for plot in PLOTS:
        stem = gt_stem(plot)
        img = cv2.cvtColor(np.array(Image.open(f"input_plots/fip/{plot}/images/{stem}.png").convert("RGB")), cv2.COLOR_RGB2BGR)
        markers = [(r["Marker"], float(r["X"]), float(r["Y"]))
                   for r in csv.DictReader(open(MARKER_CSV.format(plot=plot))) if r["Camera"] == stem]
        for name, x, y in markers:
            cv2.circle(img, (int(x), int(y)), R, (255, 0, 255), 11)    # magenta exclusion circle (high contrast on wheat, matches appendix marker highlight)
            cv2.circle(img, (int(x), int(y)), 14, (0, 0, 255), -1)     # red center dot
        out = f"{OUTDIR}/{plot}_{stem}_csvcheck.jpg"
        cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{plot}: {len(markers)} marker(s) -> {out}")
    print(f"\nmagenta circle = candidate R={R}px exclusion, red dot = CSV marker center | {os.path.abspath(OUTDIR)}")


if __name__ == "__main__":
    main()
