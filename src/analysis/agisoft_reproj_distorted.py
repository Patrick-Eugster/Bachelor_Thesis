"""Reproduce the docs' `pose_reproj` on Agisoft's DISTORTED export (colmap_distorted, FULL_OPENCV) for every
available session, to (a) list the sessions flagged unreliable (>12 px) and (b) reconcile against the
undistorted-model number from agisoft_pose_reproj.py. Per image: median reprojection of its own scene
points using pycolmap's distortion-aware projection; per session: median over images. Sorted worst->best.

CPU-only, read-only. Run:  python src/analysis/agisoft_reproj_distorted.py
"""

import os
import glob
import numpy as np
import pycolmap as pc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
POOR_THRESH = 12.0


def image_pose_reproj(im, rec, max_pts=80):
    """Median scene-point reprojection error (px) for one image, distortion-aware (im.project_point)."""
    errs = []
    p2ds = [p for p in im.points2D if p.has_point3D()]
    if not p2ds:
        return None
    step = max(1, len(p2ds) // max_pts)
    for p2d in p2ds[::step]:
        X = rec.points3D[p2d.point3D_id].xyz
        try:
            uv = im.project_point(X)          # distortion-correct projection (FULL_OPENCV), or None if behind
        except Exception:
            uv = None
        if uv is None:
            continue
        errs.append(np.hypot(uv[0] - p2d.xy[0], uv[1] - p2d.xy[1]))
    return float(np.median(errs)) if errs else None


def session_median(model_dir):
    rec = pc.Reconstruction(model_dir)
    per_img = [r for im in rec.images.values() if (r := image_pose_reproj(im, rec)) is not None]
    if not per_img:
        return None, rec.num_reg_images()
    return float(np.median(per_img)), rec.num_reg_images()


def main():
    models = sorted(set(
        glob.glob(os.path.join(REPO, "demoanlage2025_v0", "field_*", "*", "processed", "colmap_distorted", "sparse", "0"))
        + glob.glob(os.path.join(REPO, "demoanlage2025_v0", "demoanlage2025_v0_additions", "field_*", "*", "processed", "colmap_distorted", "sparse", "0"))
    ))
    rows = []
    for m in models:
        parts = m.split(os.sep)
        # find field_X and the date directly after it
        fi = next(i for i, p in enumerate(parts) if p.startswith("field_"))
        name = f"{parts[fi]}/{parts[fi+1]}"
        try:
            med, nreg = session_median(m)
            rows.append((name, med, nreg))
        except Exception as e:
            rows.append((name, None, str(e)[:30]))

    rows.sort(key=lambda r: (r[1] is None, -(r[1] or 0)))   # worst (highest) first
    print(f"\nAgisoft DISTORTED-export pose_reproj (median scene-point reproj px), worst -> best "
          f"(POOR = >{POOR_THRESH:g} px):\n")
    print(f"  {'session':<34} {'pose_reproj':>11} {'#imgs':>6}   flag")
    npoor = 0
    for name, med, nreg in rows:
        if med is None:
            print(f"  {name:<34} {'ERR':>11}   {nreg}")
            continue
        flag = "<-- POOR" if med > POOR_THRESH else ""
        if med > POOR_THRESH:
            npoor += 1
        print(f"  {name:<34} {med:>10.1f}p {nreg:>6}   {flag}")
    print(f"\n  {npoor} session(s) exceed {POOR_THRESH:g} px (flagged unreliable).")


if __name__ == "__main__":
    main()
