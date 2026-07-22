"""How precise is Agisoft, in its OWN reconstruction? Computes the internal reprojection error per session:
for every image, project each observed 3D point through that image's Agisoft pose and measure the pixel
gap to where the feature was actually detected. Low = the geometry self-closes (poses trustworthy);
high = Agisoft couldn't fit its own data (poses unreliable). Reports per-session median / p90 / max and
sorts best->worst so we can say concretely how good the best is and how bad the worst is.

CPU-only, reads the small COLMAP-txt models under agisoft/sparse/0. Run:
  python src/analysis/agisoft_pose_reproj.py
"""

import os
import glob
import numpy as np
import pycolmap as pc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def session_reproj(model_dir):
    """Median/p90/max reprojection error (px) over all point observations in an Agisoft model."""
    rec = pc.Reconstruction(model_dir)
    errs = []
    for im in rec.images.values():
        cam = rec.cameras[im.camera_id]
        T = im.cam_from_world()
        R = T.rotation.matrix()
        t = np.array(T.translation)
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]   # SIMPLE_PINHOLE
        for p2d in im.points2D:
            if not p2d.has_point3D():
                continue
            X = rec.points3D[p2d.point3D_id].xyz
            xc = R @ X + t
            if xc[2] <= 0:
                continue
            u = f * xc[0] / xc[2] + cx
            v = f * xc[1] / xc[2] + cy
            errs.append(np.hypot(u - p2d.xy[0], v - p2d.xy[1]))
    errs = np.array(errs)
    return (float(np.median(errs)), float(np.percentile(errs, 90)), float(errs.max()),
            rec.num_reg_images(), len(errs))


def main():
    rows = []
    for model in sorted(glob.glob(os.path.join(REPO, "input_plots", "phone", "*", "*", "agisoft", "sparse", "0"))):
        parts = model.split(os.sep)
        name = f"{parts[-5]}/{parts[-4]}"   # field/date (e.g. field_D/20250706)
        try:
            med, p90, mx, nreg, nobs = session_reproj(model)
            rows.append((name, med, p90, mx, nreg, nobs))
        except Exception as e:
            rows.append((name, None, None, None, None, str(e)[:40]))

    rows.sort(key=lambda r: (r[1] is None, r[1] if r[1] is not None else 1e9))
    print(f"\nAgisoft internal reprojection error (px), best -> worst:\n")
    print(f"  {'session':<32} {'median':>8} {'p90':>8} {'max':>10} {'#imgs':>6} {'#obs':>9}")
    for name, med, p90, mx, nreg, nobs in rows:
        if med is None:
            print(f"  {name:<32} ERR {nobs}")
        else:
            print(f"  {name:<32} {med:>8.2f} {p90:>8.2f} {mx:>10.1f} {nreg:>6} {nobs:>9}")
    good = [r for r in rows if r[1] is not None]
    if good:
        print(f"\n  BEST median  = {good[0][1]:.2f} px ({good[0][0]})")
        print(f"  WORST median = {good[-1][1]:.2f} px ({good[-1][0]})")


if __name__ == "__main__":
    main()
