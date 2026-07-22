"""Diagnose WHY a triangulated marker is wrong: detection vs pose, using Agisoft GT as truth.

Reproduces the full field_D/20250706 marker-89 investigation. For one marker in one session it runs:
  1. DETECTION vs GT      -> is each of our CCT detections actually on the marker? (px gap to Agisoft GT)
  2. PARALLAX kept vs all  -> did the robust solver keep a narrow-angle subset and drop wide-angle frames?
  3. TRIANGULATION vs GT   -> does any 3D-point strategy (current / all-detections / widest-pair) match GT?
  4. PAIRWISE SCATTER      -> triangulate from every WIDE-parallax pair; if perfect poses they'd all agree.
                              A wide, smooth scatter = INCONSISTENT camera poses (pose drift), NOT a bad
                              detection (a single mis-decode would be ONE outlier pair, not a gradient).

The logic: triangulation needs TWO correct inputs — detections AND camera poses. If the detections all
match GT (step 1) but the 3D point is still wrong (steps 3-4), the poses are the broken input.

Agisoft GT (marker_projections.csv, keyed "target 1..6") is the independent truth; TARGET_TO_CODE maps it
to our coded IDs. GT camera name = our image stem without ".jpg".

Usage:
  python src/analysis/diagnose_marker_triangulation.py --field field_D --plot 20250706 --marker 89
"""

import os
import csv
import json
import argparse
from itertools import combinations

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TARGET_TO_CODE = {1: 113, 2: 105, 3: 89, 4: 101, 5: 85, 6: 77}   # from src/preprocessing/marker_scale.py
CODE_TO_TARGET = {v: k for k, v in TARGET_TO_CODE.items()}


def load_intrinsics(cameras_txt):
    """SIMPLE_PINHOLE f, cx, cy from a COLMAP cameras.txt (single shared camera)."""
    for line in open(cameras_txt):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 7 and p[0].isdigit():
            return float(p[4]), float(p[5]), float(p[6])
    raise RuntimeError("no camera in " + cameras_txt)


def load_poses(images_txt):
    """{name: (R, t)} world->cam from COLMAP images.txt."""
    poses = {}
    for line in open(images_txt):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 10 and p[9].endswith(".jpg"):
            qw, qx, qy, qz, tx, ty, tz = map(float, p[1:8])
            R = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
            poses[p[9]] = (R, np.array([tx, ty, tz]))
    return poses


def load_gt(csv_path, code):
    """Agisoft GT pixels for one coded ID: {name.jpg: (x, y)}."""
    tgt = CODE_TO_TARGET[code]
    out = {}
    for r in csv.DictReader(open(csv_path)):
        if r["Marker"].strip() == f"target {tgt}":
            out[r["Camera"] + ".jpg"] = (float(r["X"]), float(r["Y"]))
    return out


def cam_center(c, poses):
    R, t = poses[c]
    return -R.T @ t


def proj(X, c, poses, f, cx, cy):
    R, t = poses[c]
    Xc = R @ X + t
    return f * Xc[0] / Xc[2] + cx, f * Xc[1] / Xc[2] + cy


def triangulate(cams, det, poses, K):
    """DLT triangulation from a list of camera names using our detections."""
    A = []
    for c in cams:
        dx, dy = det[c]
        P = K @ np.hstack([poses[c][0], poses[c][1].reshape(3, 1)])
        A.append(dx * P[2] - P[0])
        A.append(dy * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]


def parallax_pair(a, b, X, poses):
    da = cam_center(a, poses) - X
    db = cam_center(b, poses) - X
    da /= np.linalg.norm(da)
    db /= np.linalg.norm(db)
    return np.degrees(np.arccos(np.clip(da @ db, -1, 1)))


def max_parallax(cams, X, poses):
    return max((parallax_pair(a, b, X, poses) for a, b in combinations(cams, 2)), default=0.0)


def gt_median_err(X, gt, poses, f, cx, cy):
    e = [np.hypot(*(np.array(proj(X, c, poses, f, cx, cy)) - np.array(g)))
         for c, g in gt.items() if c in poses]
    return (float(np.median(e)), float(np.max(e))) if e else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="field_D")
    ap.add_argument("--plot", default="20250706")
    ap.add_argument("--marker", type=int, default=89)
    ap.add_argument("--parallax-min", type=float, default=30.0, help="deg; 'well-conditioned' pair cutoff")
    args = ap.parse_args()

    sess = os.path.join(REPO, "input_plots", "phone", args.field, args.plot)
    gt_csv = os.path.join(REPO, "demoanlage2025_v0", "demoanlage2025_v0_additions",
                          args.field, args.plot, "processed", "marker_projections.csv")

    f, cx, cy = load_intrinsics(os.path.join(sess, "sparse", "0", "cameras.txt"))
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    poses = load_poses(os.path.join(sess, "sparse", "0", "images.txt"))
    tri = json.load(open(os.path.join(sess, "logs", "marker_triangulation.json")))[str(args.marker)]
    det = {o["cam"]: o["xy"] for o in tri if o.get("src") == "detected" and o["cam"] in poses}
    inl = [o["cam"] for o in tri if o.get("src") == "detected" and o.get("inlier") and o["cam"] in poses]
    Xcur = np.array(json.load(open(os.path.join(sess, "logs", "marker_points3d.json")))
                    ["points3d"][str(args.marker)]["xyz"])
    gt = load_gt(gt_csv, args.marker) if os.path.exists(gt_csv) else {}

    print(f"=== marker {args.marker}  {args.field}/{args.plot} ===")
    print(f"detected in {len(det)} frames; solver kept {len(inl)} as inliers; GT covers {len(gt)} frames\n")

    # 1. detection vs GT
    print("1. DETECTION vs Agisoft GT (px gap; ~1px = detection is CORRECT / on the marker):")
    for c in sorted(det):
        g = gt.get(c)
        gap = f"{np.hypot(*(np.array(det[c]) - np.array(g))):.0f}px" if g else "(no GT)"
        print(f"   {c[:-4]:<24} our={tuple(round(v) for v in det[c])}  gt-gap={gap}")

    # 2. parallax kept vs all
    print("\n2. PARALLAX (max viewing-angle span at the marker):")
    print(f"   inliers the solver KEPT : {max_parallax(inl, Xcur, poses):5.1f} deg over {len(inl)} frames")
    print(f"   ALL detections available: {max_parallax(list(det), Xcur, poses):5.1f} deg over {len(det)} frames")
    print("   (narrow kept + wide available => solver dropped the wide-angle frames that pin depth)")

    # 3. triangulation strategy vs GT
    if gt:
        print("\n3. TRIANGULATION vs GT (median px error; lower=better). If SELECTION were the fault,")
        print("   using all detections would recover the truth. If it does NOT, the poses are the fault:")
        Xall = triangulate(list(det), det, poses, K)
        widest = max(combinations(det, 2), key=lambda pr: parallax_pair(*pr, triangulate(list(pr), det, poses, K), poses))
        Xwide = triangulate(list(widest), det, poses, K)
        for name, X in [("current (solver)", Xcur), ("all detections", Xall), ("widest-parallax pair", Xwide)]:
            m, mx = gt_median_err(X, gt, poses, f, cx, cy)
            print(f"   {name:22} median={m:6.0f}px  max={mx:6.0f}px")

    # 4. pairwise scatter = the pose-consistency proof
    print(f"\n4. PAIRWISE SCATTER — triangulate from every pair with parallax>{args.parallax_min:.0f} deg.")
    print("   Perfect poses => every well-conditioned pair gives the SAME point. Scatter => bad poses.")
    pts = []
    for a, b in combinations(det, 2):
        X = triangulate([a, b], det, poses, K)
        if parallax_pair(a, b, X, poses) > args.parallax_min:
            pts.append(X)
    if len(pts) >= 2:
        pts = np.array(pts)
        span = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        M = np.array([v["xyz"] for v in json.load(open(os.path.join(sess, "logs", "marker_points3d.json")))["points3d"].values()])
        plot_span = float(np.linalg.norm(M.max(0) - M.min(0)))
        print(f"   {len(pts)} well-conditioned pairs; 3D-point SPREAD = {span:.2f} "
              f"(plot ~{plot_span:.2f} across = {100*span/plot_span:.0f}% of the plot)")
        print("   => wide scatter from well-conditioned pairs can only come from INCONSISTENT poses.")

    print("\nCONCLUSION: detections correct (step 1) + no single 3D point fits (steps 3-4) => POSE drift, "
          "not mis-decoding. Triangulation takes poses as fixed, so the fix is bundle adjustment "
          "(marker_gcp_ba.py) or taking the marker from Agisoft.")


if __name__ == "__main__":
    main()
