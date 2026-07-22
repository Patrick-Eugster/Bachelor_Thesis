"""Re-score SfM experiment models (baseline / GLOMAP / SPLG / LoFTR / ALIKED-keypoints / resolution) with
the PHYSICAL-GT marker metric, replacing the discredited marker-reprojection metric.

For each model: triangulate the 6 markers from OUR marker detections + THAT model's poses (DLT), compute the
(up to 15) pairwise marker distances, fit ONE best scale to the survey distances (models are up to scale),
and report the median abs distance error vs SURVEY and vs TAPE (cm) = geometry/shape distortion. Scale is
fitted per model so different arbitrary scales compare fairly; the residual is pure SHAPE error = the
drift-sensitive signal. Non-circular (survey/tape are physical), ~cm resolution (floor: tape-vs-survey).

Usage:
  python src/analysis/rescore_models_geometry.py --field field_D --plot 20250706 \
      --models sparse/0 sparse_glomap/0 sparse_splg/0 sparse_aliked_r2048/0 sparse_loftr/0
"""

import os
import re
import json
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
import pycolmap as pc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
META = os.path.join(REPO, "demoanlage2025_v0", "metadata", "markers")
TAPE_XLSX = os.path.join(META, "Demoanlage-2025-markers-manual-distances.xlsx")
TARGET_TO_CODE = {1: 113, 2: 105, 3: 89, 4: 101, 5: 85, 6: 77}


def load_survey_cm(field):
    out = {}
    for line in open(os.path.join(META, f"{field}_coordinates.txt")):
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        p = line.split(",")
        m = re.match(r"target\s*(\d+)", p[0])
        if m and int(m.group(1)) in TARGET_TO_CODE and len(p) >= 4:
            out[TARGET_TO_CODE[int(m.group(1))]] = np.array([float(p[1]), float(p[2]), float(p[3])]) * 100.0
    return out


def load_tape_cm(field):
    letter = field.split("_")[-1]
    df = pd.read_excel(TAPE_XLSX, sheet_name=f"plot {letter}", header=None)
    tnum_col = {j: df.iloc[0, j] for j in range(1, df.shape[1])}
    out = {}
    for i in range(1, df.shape[0]):
        ti = df.iloc[i, 0]
        for j in range(1, df.shape[1]):
            v, tj = df.iloc[i, j], tnum_col[j]
            if pd.notna(v) and pd.notna(ti) and pd.notna(tj):
                ci, cj = TARGET_TO_CODE.get(int(ti)), TARGET_TO_CODE.get(int(tj))
                if ci and cj and ci != cj:
                    out[frozenset((ci, cj))] = float(v)
    return out


def load_dets(sess):
    tri = json.load(open(os.path.join(sess, "logs", "marker_triangulation.json")))
    return {code: [(o["cam"], tuple(o["xy"])) for o in tri.get(str(code), []) if o.get("src") == "detected"]
            for code in TARGET_TO_CODE.values()}


def triangulate(dets, poses, K):
    A = []
    for c, (x, y) in dets:
        R, t = poses[c]
        P = K @ np.hstack([R, t.reshape(3, 1)])
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    if len(A) < 4:
        return None
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]


def _reproj(c, xy, X, poses, K):
    R, t = poses[c]
    uv = K @ (R @ X + t)
    if uv[2] <= 0:
        return 1e9
    return np.hypot(uv[0] / uv[2] - xy[0], uv[1] / uv[2] - xy[1])


def robust_triangulate(dets, poses, K, thresh=8.0, min_obs=3):
    """RANSAC-lite: find the pair of observations whose triangulation has the MOST inliers (reproj<thresh),
    then refine on those inliers. Robust to a high fraction of outlier (canopy false-positive) detections,
    which a single DLT-over-all-observations is NOT — this is what made the old one-pass version wrong."""
    if len(dets) < 2:
        return None
    best_inl, bestX = [], None
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            X = triangulate([dets[i], dets[j]], poses, K)
            if X is None:
                continue
            inl = [o for o in dets if _reproj(o[0], o[1], X, poses, K) < thresh]
            if len(inl) > len(best_inl):
                best_inl, bestX = inl, X
    if len(best_inl) >= min_obs:
        Xr = triangulate(best_inl, poses, K)          # refine on the consensus set
        return Xr if Xr is not None else bestX
    return bestX                                       # too few inliers: best 2-view estimate


def model_marker_dists(model_dir, dets):
    rec = pc.Reconstruction(model_dir)
    poses = {}
    for im in rec.images.values():
        T = im.cam_from_world()
        poses[im.name] = (T.rotation.matrix(), np.array(T.translation))
    cam = list(rec.cameras.values())[0]
    f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    pos = {}
    for code, d in dets.items():
        dd = [(c, xy) for c, xy in d if c in poses]
        if len(dd) >= 2:
            X = robust_triangulate(dd, poses, K)
            if X is not None:
                pos[code] = X
    return ({frozenset((a, b)): float(np.linalg.norm(pos[a] - pos[b])) for a, b in combinations(pos, 2)},
            rec.num_reg_images(), len(pos))


def bestfit_resid(our, ref):
    keys = set(our) & set(ref)
    if len(keys) < 2:
        return None, 0
    o = np.array([our[k] for k in keys])
    r = np.array([ref[k] for k in keys])
    s = np.median(r / o)                          # robust single scale (models are up to scale)
    return float(np.median(np.abs(o * s - r))), len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--plot", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    sess = os.path.join(REPO, "input_plots", "phone", args.field, args.plot)
    survey = load_survey_cm(args.field)
    try:
        tape = load_tape_cm(args.field)
    except Exception:
        tape = {}
    d_sur = {frozenset((a, b)): float(np.linalg.norm(survey[a] - survey[b]))
             for a, b in combinations(survey, 2)}
    dets = load_dets(sess)

    if not args.quiet:
        print(f"\n=== {args.field}/{args.plot}: marker-geometry SHAPE error vs physical GT (cm, best-fit scale) ===")
        print(f"  {'model':<26} {'#reg':>5} {'#mk':>4} {'vs SURVEY':>10} {'vs TAPE':>9}")
    rows = []
    for model in args.models:
        mdir = os.path.join(sess, model)
        if not (os.path.exists(os.path.join(mdir, "images.bin")) or os.path.exists(os.path.join(mdir, "images.txt"))):
            if not args.quiet:
                print(f"  {model:<26} MISSING")
            continue
        d_our, nreg, nmk = model_marker_dists(mdir, dets)
        rs, ns = bestfit_resid(d_our, d_sur)
        rt, nt = bestfit_resid(d_our, tape)
        rows.append((model, nreg, nmk, rs, rt))
        if not args.quiet:
            fs = f"{rs:.2f} (n{ns})" if rs is not None else "-"
            ft = f"{rt:.2f} (n{nt})" if rt is not None else "-"
            print(f"  {model:<26} {nreg:>5} {nmk:>4} {fs:>10} {ft:>9}")
    return rows


if __name__ == "__main__":
    main()
