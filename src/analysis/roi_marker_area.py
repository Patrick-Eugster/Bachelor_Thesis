"""Approximate the metric area of each phone plot's ROI from the triangulated markers.

The ROI is the convex hull of a session's coded markers. We read the triangulated marker 3D points
(logs/marker_points3d.json, in arbitrary COLMAP units) and the tape-derived metric scale
(logs/marker_scale.json, metres per COLMAP unit), fit the marker plane by PCA, take the 2D convex-hull
area in that plane, and convert to m^2 with scale^2. We also report the buffered area the segmentation
actually culls with (roi_buffer_m outward, default 0.25 m): buffering a polygon by d adds
perimeter*d + pi*d^2.

Read-only: scans input_plots/phone for sessions that have BOTH json files and writes one summary JSON to
docs/analysis_results/. Run from repo root:  python -m src.analysis.roi_marker_area
"""
import glob
import json
import math
import os

import numpy as np
from scipy.spatial import ConvexHull

ROI_BUFFER_M = 0.25   # segmentation_3d default roi_buffer_m — the outward cull margin
OUT = "docs/analysis_results/roi_marker_area.json"


def hull_area_m2(points_xyz, scale):
    """Fit the marker plane by PCA, project to 2D, and return (area_m2, perimeter_m, planarity_cm_rms,
    extent_m (w,h), hull_ids_idx) for the convex hull of the markers scaled to metres."""
    P = np.asarray(points_xyz, float)
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c)          # rows: two in-plane axes, then the plane normal
    b1, b2, n = Vt[0], Vt[1], Vt[2]
    uv = np.c_[(P - c) @ b1, (P - c) @ b2]    # 2D coords in the marker plane
    h = ConvexHull(uv)
    area_m2 = h.volume * scale ** 2           # for a 2D hull scipy stores area in .volume
    per_m = h.area * scale                    # ...and perimeter in .area
    resid = (P - c) @ n                       # out-of-plane spread = how planar the markers are
    ext = ((uv[:, 0].max() - uv[:, 0].min()) * scale,
           (uv[:, 1].max() - uv[:, 1].min()) * scale)
    return area_m2, per_m, resid.std() * scale * 100.0, ext, list(h.vertices)


def buffered_area(area_m2, per_m, d=ROI_BUFFER_M):
    """area after growing the polygon outward by d metres (Minkowski sum with a disk of radius d)."""
    return area_m2 + per_m * d + math.pi * d ** 2


def main():
    rows = []
    # every session dir is input_plots/phone/<field>/<date>; use its top-level logs/ marker files
    for scale_path in sorted(glob.glob("input_plots/phone/*/*/logs/marker_scale.json")):
        sess_logs = os.path.dirname(scale_path)
        pts_path = os.path.join(sess_logs, "marker_points3d.json")
        if not os.path.exists(pts_path):
            continue
        session = os.path.relpath(os.path.dirname(sess_logs), "input_plots/phone")
        sc = json.load(open(scale_path))
        scale = sc.get("scale_metric")
        reliable = sc.get("scale_reliable")
        pj = json.load(open(pts_path))
        pts = pj.get("points3d", {})
        if scale is None or len(pts) < 3:
            rows.append(dict(session=session, status="skipped (no scale or <3 markers)",
                             n_markers=len(pts)))
            continue
        ids = sorted(pts, key=int)
        xyz = [pts[m]["xyz"] for m in ids]
        area, per, planar_cm, ext, hidx = hull_area_m2(xyz, scale)
        rows.append(dict(
            session=session, n_markers=len(ids), marker_ids=[int(m) for m in ids],
            scale_m_per_unit=round(scale, 6), scale_reliable=reliable,
            roi_area_m2=round(area, 3),
            roi_area_buffered_m2=round(buffered_area(area, per), 3),
            roi_perimeter_m=round(per, 3),
            extent_m=[round(ext[0], 2), round(ext[1], 2)],
            planarity_cm_rms=round(planar_cm, 2),
            hull_all_markers=(len(hidx) == len(ids)),
        ))
        print(f"{session:24s}  area={area:5.2f} m^2  (buffered {buffered_area(area, per):5.2f})  "
              f"{ext[0]:.2f}x{ext[1]:.2f} m  planar {planar_cm:.1f} cm rms  "
              f"{len(ids)} markers{'' if len(hidx)==len(ids) else '  (some interior!)'}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"roi_buffer_m": ROI_BUFFER_M, "sessions": rows}, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
