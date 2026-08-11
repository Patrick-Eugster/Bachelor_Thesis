"""Plots the matched-pair IoU distribution for three phone configurations, pooled
over the 6 phone GT images and all anchored on the winner YOLO11 + per-head + SAM2.
The other two each change one axis: SAHI + per-head + SAM2 varies only the detector,
and YOLO11 + per-tile + SAM2 varies only the granularity (the cheaper practical
choice, since per-head is far slower in SAM). Each matched prediction/GT-head pair
contributes its mask IoU; only pairs at or above the 0.5 matching gate are kept, so
the mean of each distribution equals that configuration's matched-mask IoU in the
results table. The figure shows the spread behind those single means --- most matched
heads are segmented well (mass at 0.8-0.95) with a thin low-quality tail near the 0.5
gate, and dropping to per-tile only shifts the distribution slightly left.

Run: python src/analysis/plot_matched_iou_hist.py
Output: thesis/figures/maskgen_phone_matched_iou.png
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "results/mask_generation/phone/evaluation"
CONFIGS = [
    ("YOLO11 + per head + SAM2", f"{BASE}/yolo11_sam/masks_instance/gt_head_sam2/eval_masks_instance.json", "#4C72B0"),
    ("SAHI + per head + SAM2",   f"{BASE}/sahi_yolo_sam/masks_instance/gt_head_sam2/eval_masks_instance.json", "#DD8452"),
    ("YOLO11 + per tile + SAM2", f"{BASE}/yolo11_sam/masks_instance/gt_tile_sam2/eval_masks_instance.json", "#55A868"),
]
OUT = "thesis/figures/maskgen_phone_matched_iou.png"
GATE = 0.5   # matching gate used everywhere in the results table


def pooled_ious(path):
    """Pools every matched-pair mask IoU over the GT images and keeps the 0.5-gate pairs."""
    d = json.load(open(path))
    vals = []
    for im in d["images"]:
        vals += im["matched_ious"]
    vals = np.array(vals)
    return vals[vals >= GATE], d["n_images"]


def main():
    bins = np.linspace(GATE, 1.0, 21)
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for label, path, color in CONFIGS:
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} --- run the phone grid eval first")
        v, n_img = pooled_ious(path)
        mean = v.mean()
        ax.hist(v, bins=bins, histtype="step", linewidth=1.8, color=color,
                label=f"{label}  (mean {mean:.3f}, n={len(v)})")
        ax.axvline(mean, color=color, linestyle="--", linewidth=1.0, alpha=0.7)
        print(f"{label}: {len(v)} pairs over {n_img} images, "
              f"mean {mean:.3f}, median {np.median(v):.3f}")
    ax.set_xlabel("matched-pair mask IoU")
    ax.set_ylabel("matched heads")
    ax.set_xlim(GATE, 1.0)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
