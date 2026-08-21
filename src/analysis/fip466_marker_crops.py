"""Shows the plot_466 coded-marker false-positive story as image crops for eyeballing. For each of the
two clearly-visible calibration markers in cam_12, it stacks three crops side by side: the raw
undistorted frame, the YOLOv5-1280 eval2d overlay (markers painted red = FP), and the YOLO11 eval2d
overlay (markers left clean). Red in the eval overlays = false positive (invented head), green = TP,
blue = missed head. Output goes to docs/analysis_results/ so nothing touches the thesis yet."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

P = "plot_466"
STEM = "FPWW036_SR0466_FIP2_cam_12"
RAW = f"input_plots/fip/{P}/images/{STEM}.png"
EVAL = "results/reconstruction/fip/{p}/vanilla_3dgs/fipseg15k_pp/segmentation_3d/{d}/eval_2d/{stem}_eval2d.png"
OUT = "docs/analysis_results/fip466_marker_fp_crops.png"

# marker crop windows in the original 4096x2996 frame (x0,y0,x1,y1), read off the eval2d overlays.
MARKERS = {
    "left marker":  (40,  1320, 780,  2040),
    "right marker": (2100, 1040, 2950, 1720),
}
COLS = [("raw frame", RAW), ("YOLOv5-1280 (FP=red)", "1280"), ("YOLO11 (clean)", "yolo11")]
DET = {"1280": "seg_yv5_1280", "yolo11": "seg_yolo11"}


def load(col_key):
    """Loads the full image for one column (raw frame or one detector's eval2d overlay)."""
    if col_key == RAW:
        return np.array(Image.open(RAW).convert("RGB"))
    return np.array(Image.open(EVAL.format(p=P, d=DET[col_key], stem=STEM)).convert("RGB"))


def main():
    imgs = {key: load(src) for _, src in COLS for key in [src]}
    nrows, ncols = len(MARKERS), len(COLS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    for r, (mname, (x0, y0, x1, y1)) in enumerate(MARKERS.items()):
        for c, (title, src) in enumerate(COLS):
            ax = axes[r, c]
            ax.imshow(imgs[src][y0:y1, x0:x1])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=13)
            if c == 0:
                ax.set_ylabel(mname, fontsize=13, fontweight="bold")
    fig.suptitle("plot_466 cam_12 — calibration markers segmented as heads by YOLOv5-1280 (red), not by YOLO11",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
