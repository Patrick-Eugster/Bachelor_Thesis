"""
seg_roi_keep_fraction.py — sanity-check the 3D segmentation ROI cull (F1/F2) on a trained model
WITHOUT running the (Euler-only) segmentation.

Loads a trained point_cloud.ply + the plot's marker_points3d.json, builds the ROI keep-mask with
segmentation_3d.seg_roi, and reports how many Gaussians survive F1 (roi_cull) and F1+F2
(roi_cull+marker_exclude), plus the in-plane hull fraction and the height distribution of the
in-hull Gaussians (so you can see the head band isn't clipping real heads).

Run (from repo root, PYTHONPATH=src):
  PYTHONPATH=src python src/analysis/seg_roi_keep_fraction.py \
    --ply results/reconstruction/phone/field_A/20250715/opencv/vanilla_3dgs/baseline/point_cloud/iteration_15000/point_cloud.ply \
    --plot_dir input_plots/phone/field_A/20250715/opencv
"""

import argparse

import numpy as np
import torch
from plyfile import PlyData

from segmentation_3d import seg_roi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply", required=True, help="trained point_cloud.ply")
    ap.add_argument("--plot_dir", required=True, help="plot source folder (holds logs/marker_points3d.json)")
    ap.add_argument("--buffer_m", type=float, default=0.25)
    ap.add_argument("--marker_radius_m", type=float, default=0.12)
    args = ap.parse_args()

    v = PlyData.read(args.ply)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    xyz_t = torch.from_numpy(xyz)
    print(f"total gaussians: {len(xyz)}")

    print("\n--- F1 only (roi_cull) ---")
    k1 = seg_roi.build_roi_keep_mask(xyz_t, args.plot_dir, roi_cull=True, marker_exclude=False,
                                     roi_buffer_m=args.buffer_m, marker_radius_m=args.marker_radius_m)
    print("\n--- F1+F2 (roi_cull + marker_exclude) ---")
    k2 = seg_roi.build_roi_keep_mask(xyz_t, args.plot_dir, roi_cull=True, marker_exclude=True,
                                     roi_buffer_m=args.buffer_m, marker_radius_m=args.marker_radius_m)
    if k1 is None:
        print("\nROI could not be built (missing/partial markers) — nothing to report.")
        return
    print(f"\nF2 alone removed near-marker (stakes): {int(k1.sum() - k2.sum())} gaussians")

    # height breakdown of the in-hull Gaussians — shows the head band isn't clipping heads
    m = seg_roi._load_markers(args.plot_dir)
    c, u, vv, n = seg_roi._fit_plane(m, xyz)
    basis = np.stack([u, vv], axis=1)
    d = xyz - c
    uv = d @ basis
    h = d @ n
    hull = seg_roi._convex_hull_2d((m - c) @ basis)
    in_hull = seg_roi._inside_convex_hull(uv, hull, args.buffer_m)
    hh = h[in_hull]
    pct = {p: round(float(np.percentile(hh, p)), 2) for p in [1, 5, 25, 50, 75, 95, 99]}
    print(f"\nin-hull fraction (buffer {args.buffer_m} m): {100 * in_hull.mean():.1f}%")
    print(f"in-hull height percentiles above marker plane (m): {pct}")
    print("  (heads sit just above the plane; the head band default (-0.5, 1.5) m keeps all of them)")


if __name__ == "__main__":
    main()
