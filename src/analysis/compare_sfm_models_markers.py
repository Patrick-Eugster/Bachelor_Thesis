"""Compare SfM models (e.g. incremental sparse/0 vs global sparse_glomap/0) by how well each one's
camera poses let the coded markers reproject onto Agisoft GT — using an IDENTICAL method for every model
so the comparison is fair.

For each model and each marker: triangulate the marker from the SAME 2D detections (plain DLT over all
detections registered in that model), then reproject the 3D point into every GT frame and report the
median pixel error vs Agisoft GT (target 1..6 -> our codes via TARGET_TO_CODE). Lower = better poses.
Also reports each model's pairwise-triangulation scatter (wide-parallax pairs) as a pose-consistency check.

Reprojection error vs GT is scale/frame-independent (2D), so models in different frames compare directly.

Usage:
  python src/analysis/compare_sfm_models_markers.py --field field_D --plot 20250706 \
      --models sparse/0 sparse_glomap/0
"""

import os
import csv
import json
import argparse
from itertools import combinations

import numpy as np
import pycolmap as pc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TARGET_TO_CODE = {1: 113, 2: 105, 3: 89, 4: 101, 5: 85, 6: 77}


def load_model(model_dir):
    """Return (poses{name:(R,t)}, f, cx, cy) from a COLMAP model (bin or txt) via pycolmap."""
    rec = pc.Reconstruction(model_dir)
    poses = {}
    for img in rec.images.values():
        T = img.cam_from_world()
        poses[img.name] = (T.rotation.matrix(), np.array(T.translation))
    cam = list(rec.cameras.values())[0]                 # single shared camera (SIMPLE_PINHOLE)
    f, cx, cy = float(cam.params[0]), float(cam.params[1]), float(cam.params[2])
    return poses, f, cx, cy, rec.num_points3D()


def triangulate(dets, poses, K):
    """Plain DLT from [(name,(x,y)), ...] using the model's poses."""
    A = []
    for c, (x, y) in dets:
        P = K @ np.hstack([poses[c][0], poses[c][1].reshape(3, 1)])
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]


def gt_median(X, gt, poses, f, cx, cy):
    e = []
    for c, g in gt.items():
        if c in poses:
            R, t = poses[c]
            Xc = R @ X + t
            if Xc[2] > 0:
                e.append(np.hypot(f * Xc[0] / Xc[2] + cx - g[0], f * Xc[1] / Xc[2] + cy - g[1]))
    return (float(np.median(e)), len(e)) if e else (None, 0)


def parallax(a, b, X, poses):
    ca = -poses[a][0].T @ poses[a][1]
    cb = -poses[b][0].T @ poses[b][1]
    da, db = ca - X, cb - X
    da /= np.linalg.norm(da); db /= np.linalg.norm(db)
    return np.degrees(np.arccos(np.clip(da @ db, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="field_D")
    ap.add_argument("--plot", default="20250706")
    ap.add_argument("--models", nargs="+", default=["sparse/0", "sparse_glomap/0"])
    args = ap.parse_args()

    sess = os.path.join(REPO, "input_plots", "phone", args.field, args.plot)
    gt_csv = os.path.join(REPO, "demoanlage2025_v0", "demoanlage2025_v0_additions",
                          args.field, args.plot, "processed", "marker_projections.csv")
    tri = json.load(open(os.path.join(sess, "logs", "marker_triangulation.json")))
    gtrows = list(csv.DictReader(open(gt_csv)))

    print(f"=== {args.field}/{args.plot}: per-marker median reprojection error vs Agisoft GT (px) ===")
    print("(IDENTICAL DLT-from-all-detections on every model — lower = better poses)\n")

    results = {}
    for model in args.models:
        mdir = os.path.join(sess, model)
        if not (os.path.exists(os.path.join(mdir, "images.bin")) or os.path.exists(os.path.join(mdir, "images.txt"))):
            print(f"  {model}: MISSING"); continue
        poses, f, cx, cy, npts = load_model(mdir)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        row = {}
        allX = []
        for tnum, code in TARGET_TO_CODE.items():
            dets = [(o["cam"], tuple(o["xy"])) for o in tri[str(code)]
                    if o.get("src") == "detected" and o["cam"] in poses]
            gt = {r["Camera"] + ".jpg": (float(r["X"]), float(r["Y"]))
                  for r in gtrows if r["Marker"] == f"target {tnum}"}
            if len(dets) < 2 or not gt:
                row[code] = None; continue
            X = triangulate(dets, poses, K)
            med, n = gt_median(X, gt, poses, f, cx, cy)
            row[code] = med
            if code == 89:
                # pairwise scatter for the canary marker
                pts = [triangulate([a, b], poses, K) for a, b in combinations(dets, 2)
                       if parallax(a[0], b[0], triangulate([a, b], poses, K), poses) > 30]
                row["_scatter89"] = float(np.linalg.norm(np.array(pts).max(0) - np.array(pts).min(0))) if len(pts) >= 2 else None
        row["_npts"] = npts
        results[model] = row

    codes = [113, 105, 89, 101, 85, 77]
    print(f"{'model':<20} {'#pts':>6}  " + "  ".join(f"m{c:>3}" for c in codes) + "   scatter89")
    for model, row in results.items():
        cells = "  ".join((f"{row[c]:4.0f}" if row.get(c) is not None else "  --") for c in codes)
        sc = f"{row['_scatter89']:.2f}" if row.get("_scatter89") is not None else "--"
        print(f"{model:<20} {row['_npts']:>6}  {cells}    {sc}")
    print("\nmedians across the 6 markers:")
    for model, row in results.items():
        vals = [row[c] for c in codes if row.get(c) is not None]
        print(f"  {model:<20} median-of-markers = {np.median(vals):6.1f} px" if vals else f"  {model}: no data")


if __name__ == "__main__":
    main()
