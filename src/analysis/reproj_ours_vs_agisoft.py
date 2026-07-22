"""Answer two questions with real numbers:
(Q1) Was the >12px 'POOR' an artifact of not applying lens distortion? -> take one distorted model, split
     its reprojection error by radial distance from image centre. If error is small in the centre and blows
     up at the edges, the 'error' is uncorrected lens distortion (grows with radius), not pose error.
(Q2) Is Agisoft better than our COLMAP on reprojection? -> for each session compute the SAME clean
     reprojection (undistorted SIMPLE_PINHOLE models) for OUR sparse/0 and Agisoft's agisoft/sparse/0,
     apples-to-apples.

CPU-only, read-only. Run:  python src/analysis/reproj_ours_vs_agisoft.py
"""

import os
import glob
import numpy as np
import pycolmap as pc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def simple_pinhole_median(model_dir):
    """Median reprojection error (px) of a SIMPLE_PINHOLE model, manual f*x/z+c projection (no distortion)."""
    rec = pc.Reconstruction(model_dir)
    errs = []
    for im in rec.images.values():
        cam = rec.cameras[im.camera_id]
        T = im.cam_from_world(); R = T.rotation.matrix(); t = np.array(T.translation)
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
        for p2d in im.points2D:
            if not p2d.has_point3D():
                continue
            X = rec.points3D[p2d.point3D_id].xyz
            xc = R @ X + t
            if xc[2] <= 0:
                continue
            errs.append(np.hypot(f * xc[0] / xc[2] + cx - p2d.xy[0], f * xc[1] / xc[2] + cy - p2d.xy[1]))
    return (float(np.median(errs)) if errs else None), rec.num_reg_images()


def radial_diagnostic(distorted_model):
    """(Q1) reprojection error vs radius-from-centre on a distorted FULL_OPENCV model, projected WITHOUT
    applying distortion (manual pinhole). If it grows with radius, that IS the lens distortion."""
    rec = pc.Reconstruction(distorted_model)
    cam = list(rec.cameras.values())[0]
    print(f"   camera model = {cam.model.name}, params = {[round(p,3) for p in cam.params]}")
    W, H = cam.width, cam.height
    f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
    rows = []
    for im in rec.images.values():
        T = im.cam_from_world(); R = T.rotation.matrix(); t = np.array(T.translation)
        for p2d in im.points2D:
            if not p2d.has_point3D():
                continue
            X = rec.points3D[p2d.point3D_id].xyz
            xc = R @ X + t
            if xc[2] <= 0:
                continue
            u = f * xc[0] / xc[2] + cx; v = f * xc[1] / xc[2] + cy   # pinhole, NO distortion
            err = np.hypot(u - p2d.xy[0], v - p2d.xy[1])
            r = np.hypot(p2d.xy[0] - W / 2, p2d.xy[1] - H / 2) / (0.5 * np.hypot(W, H))  # 0=centre,1=corner
            rows.append((r, err))
    rows = np.array(rows)
    print(f"   reprojection error (px) by radius-from-centre (pinhole projection, distortion NOT applied):")
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        m = rows[(rows[:, 0] >= lo) & (rows[:, 0] < hi)]
        if len(m):
            print(f"     r {lo:.1f}-{hi:.1f}: median {np.median(m[:,1]):6.1f} px   (n={len(m)})")


def main():
    print("=== Q1: is the distorted >12px just uncorrected lens distortion? (field_A/20250603) ===")
    dist = os.path.join(REPO, "demoanlage2025_v0", "field_A", "20250603", "processed", "colmap_distorted", "sparse", "0")
    if os.path.isdir(dist):
        radial_diagnostic(dist)
    else:
        print(f"   missing {dist}")

    print("\n=== Q2: reprojection OURS vs AGISOFT (both undistorted SIMPLE_PINHOLE, clean, px) ===")
    print(f"  {'session':<34} {'OURS':>8} {'AGISOFT':>9}")
    sessions = sorted(glob.glob(os.path.join(REPO, "input_plots", "phone", "*", "*", "agisoft", "sparse", "0")))
    ours_all, agi_all = [], []
    for agi in sessions:
        sess = agi[:agi.index(os.sep + "agisoft")]
        parts = sess.split(os.sep); name = f"{parts[-2]}/{parts[-1]}"
        ours_dir = os.path.join(sess, "sparse", "0")
        o = simple_pinhole_median(ours_dir)[0] if os.path.isdir(ours_dir) else None
        a = simple_pinhole_median(agi)[0]
        if o is not None:
            ours_all.append(o)
        if a is not None:
            agi_all.append(a)
        os_ = f"{o:.2f}" if o is not None else "-"
        as_ = f"{a:.2f}" if a is not None else "-"
        print(f"  {name:<34} {os_:>8} {as_:>9}")
    print(f"\n  median across sessions:  OURS {np.median(ours_all):.2f} px   AGISOFT {np.median(agi_all):.2f} px")


if __name__ == "__main__":
    main()
