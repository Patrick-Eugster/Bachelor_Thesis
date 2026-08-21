"""Top-down map of head centroids colored by size — WHERE the over-merged (big) heads sit vs normal ones.

Answers "are the big/same-colour heads in the dense center?" For one seg run: compute each head's
centroid, project onto the marker plane (same _fit_plane as the ROI cull), and scatter them — normal
heads small/grey, big heads (> big_thresh Gaussians) red and size-scaled — with the markers + plot hull
drawn in. Also prints a numeric center-concentration check: median radial distance (from the plot
centroid, in-plane) of big vs normal heads, so the "center" claim is quantified, not just visual.

Read-only. Usage:
  python src/analysis/seg_head_size_map.py --seg path/to/segmentation_3d/EXP \
    --plot_dir input_plots/phone/field_A/20250715/opencv --out_dir docs/analysis_results/seg_head_sizes
"""
import argparse
import os

import numpy as np
import torch
from plyfile import PlyData

from segmentation_3d.seg_roi import _load_markers, _fit_plane, _convex_hull_2d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True, help="segmentation_3d/EXP dir (gaussians.ply + all_obj_labels.pth)")
    ap.add_argument("--plot_dir", required=True, help="the SfM source dir (holds logs/marker_points3d.json)")
    ap.add_argument("--big_thresh", type=int, default=800, help="heads with > this many Gaussians = likely merged")
    ap.add_argument("--out_dir", default="docs/analysis_results/seg_head_sizes")
    a = ap.parse_args()

    L = torch.load(os.path.join(a.seg, "all_obj_labels.pth"), map_location="cpu")
    ply = PlyData.read(os.path.join(a.seg, "gaussians.ply"))["vertex"]
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype(np.float64)

    # per-head centroid + size (skip empty heads)
    counts = L[1:].sum(dim=1).numpy()
    cent, size = [], []
    for h in range(1, L.shape[0]):
        if counts[h - 1] == 0:
            continue
        cent.append(xyz[L[h].numpy()].mean(0)); size.append(int(counts[h - 1]))
    cent = np.array(cent); size = np.array(size)

    # project head centroids onto the marker plane -> in-plane (u,v)
    markers = _load_markers(a.plot_dir)
    c, u, v, n = _fit_plane(markers, xyz)
    basis = np.stack([u, v], axis=1)
    uv = (cent - c) @ basis
    mk_uv = (markers - c) @ basis
    plot_center = uv.mean(0)

    big = size > a.big_thresh
    # numeric center-concentration check: radial distance from plot centroid, in-plane
    rad = np.linalg.norm(uv - plot_center, axis=1)
    print(f"heads: {len(size)} | big (>{a.big_thresh}g): {int(big.sum())} | normal: {int((~big).sum())}")
    print(f"median in-plane distance from plot centroid:  big={np.median(rad[big]):.3f} u   "
          f"normal={np.median(rad[~big]):.3f} u")
    print(f"(big/normal ratio {np.median(rad[big])/np.median(rad[~big]):.2f} — <1 means big heads sit MORE central)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(uv[~big, 0], uv[~big, 1], s=6, c="0.7", label=f"normal head (<={a.big_thresh}g)")
        ax.scatter(uv[big, 0], uv[big, 1], s=size[big] / 25.0, c="crimson", alpha=0.7,
                   edgecolors="k", linewidths=0.3, label=f"big head (>{a.big_thresh}g, size∝Gaussians)")
        hull = _convex_hull_2d(mk_uv)
        hp = np.vstack([hull, hull[0]])
        ax.plot(hp[:, 0], hp[:, 1], "b-", lw=1, label="marker hull (plot ROI)")
        ax.scatter(mk_uv[:, 0], mk_uv[:, 1], marker="^", s=120, c="blue", label="markers")
        ax.set_aspect("equal"); ax.set_xlabel("in-plane u (model units)"); ax.set_ylabel("in-plane v")
        ax.set_title(f"Head centroids by size (top-down)  —  {os.path.basename(a.seg)}")
        ax.legend(loc="best", fontsize=8); fig.tight_layout()
        os.makedirs(a.out_dir, exist_ok=True)
        out = os.path.join(a.out_dir, "head_size_map.png")
        fig.savefig(out, dpi=140); plt.close()
        print(f"wrote {out}")
    except Exception as e:
        print(f"(map skipped: {e})")


if __name__ == "__main__":
    main()
