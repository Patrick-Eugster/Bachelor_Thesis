"""Visualise COLMAP matches colour-coded by epipolar (Sampson) residual, to see with our own eyes
whether high-residual mid-baseline matches connect the SAME physical wheat head (=> pose error) or
DIFFERENT heads (=> false match). Picks the image pair in a chosen time-gap window with the most matches.

Outputs (to --out dir):
  <pair>_composite.jpg  : the two images side by side, a sample of matches drawn as lines
                          (green Sampson<5px, orange 5-30px, red >30px)
  <pair>_crops.jpg      : zoomed 160px crop of kp1 (left) vs kp2 (right) for N high-residual matches,
                          stacked - so we can compare the actual local patches match-by-match.

Usage:
  python src/analysis/viz_matches_residual.py --field field_D --plot 20250706 --min_gap 15 --max_gap 60
"""

import os
import re
import sqlite3
import argparse

import numpy as np
import cv2
import pycolmap as pc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAX_IMAGE_ID = 2147483647


def img_time(name):
    m = re.search(r"_(\d{8})_(\d{2})(\d{2})(\d{2})", name)
    return int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4)) if m else None


def pair_images(pid):
    i2 = pid % MAX_IMAGE_ID
    return (pid - i2) // MAX_IMAGE_ID, i2


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def sampson_one(F, x1, x2):
    x1h = np.array([x1[0], x1[1], 1.0]); x2h = np.array([x2[0], x2[1], 1.0])
    Fx1 = F @ x1h; Ftx2 = F.T @ x2h
    num = (x2h @ Fx1) ** 2
    den = Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2
    return np.sqrt(num / den) if den > 1e-12 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="field_D")
    ap.add_argument("--plot", default="20250706")
    ap.add_argument("--model", default="sparse/0")
    ap.add_argument("--img_subdir", default="input_uniform")
    ap.add_argument("--min_gap", type=float, default=15)
    ap.add_argument("--max_gap", type=float, default=60)
    ap.add_argument("--n_lines", type=int, default=40)
    ap.add_argument("--n_crops", type=int, default=8)
    ap.add_argument("--crop_which", choices=["high", "low"], default="high",
                    help="high=suspected false matches (>30px); low=clean matches (<5px, should be same head)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sess = os.path.join(REPO, "input_plots", "phone", args.field, args.plot)
    rec = pc.Reconstruction(os.path.join(sess, args.model))
    cam = list(rec.cameras.values())[0]
    f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]); Kinv = np.linalg.inv(K)
    pose = {}
    for im in rec.images.values():
        T = im.cam_from_world()
        pose[im.name] = (T.rotation.matrix(), np.array(T.translation), img_time(im.name))

    con = sqlite3.connect(os.path.join(sess, "distorted", "database.db"))
    names = {i: n for i, n in con.execute("SELECT image_id, name FROM images")}
    kps = {}
    for iid, rows, cols, data in con.execute("SELECT image_id, rows, cols, data FROM keypoints"):
        if data and rows:
            kps[iid] = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)[:, :2].copy()
    # pick pair in the time window with the most matches
    best = None
    for pid, rows, cols, data in con.execute("SELECT pair_id, rows, cols, data FROM two_view_geometries"):
        if not data or not rows:
            continue
        i1, i2 = pair_images(pid)
        n1, n2 = names.get(i1), names.get(i2)
        if n1 not in pose or n2 not in pose:
            continue
        t1, t2 = pose[n1][2], pose[n2][2]
        if t1 is None or t2 is None:
            continue
        gap = abs(t1 - t2)
        if args.min_gap <= gap < args.max_gap and (best is None or rows > best[1]):
            best = (pid, rows, i1, i2, gap, np.frombuffer(data, dtype=np.uint32).reshape(rows, cols)[:, :2])
    con.close()
    if best is None:
        print("no pair in time window"); return
    pid, rows, i1, i2, gap, m = best
    n1, n2 = names[i1], names[i2]
    print(f"pair: {n1}  <->  {n2}   time-gap={gap:.0f}s   matches={rows}")

    R1, t1, _ = pose[n1]; R2, t2, _ = pose[n2]
    Rrel = R2 @ R1.T; trel = t2 - Rrel @ t1
    F = Kinv.T @ skew(trel) @ Rrel @ Kinv

    x1 = kps[i1][m[:, 0]]; x2 = kps[i2][m[:, 1]]
    d = np.array([sampson_one(F, a, b) for a, b in zip(x1, x2)])

    out = args.out or os.path.join(REPO, "docs", "analysis_results",
                                   "repetitive_false_matches", args.plot)
    os.makedirs(out, exist_ok=True)
    imgA = cv2.imread(os.path.join(sess, args.img_subdir, n1))
    imgB = cv2.imread(os.path.join(sess, args.img_subdir, n2))
    H, W = imgA.shape[:2]
    comp = np.hstack([imgA, imgB])

    # draw a sample: prefer a mix of low and high residual
    lo_idx = np.where(d < 5)[0]; hi_idx = np.where(d > 30)[0]
    rng = np.random.default_rng(0)
    pick = np.concatenate([rng.choice(lo_idx, min(args.n_lines // 2, len(lo_idx)), replace=False),
                           rng.choice(hi_idx, min(args.n_lines // 2, len(hi_idx)), replace=False)])
    for k in pick:
        p1 = (int(x1[k][0]), int(x1[k][1])); p2 = (int(x2[k][0]) + W, int(x2[k][1]))
        col = (0, 200, 0) if d[k] < 5 else (0, 165, 255) if d[k] < 30 else (0, 0, 255)
        cv2.circle(comp, p1, 10, col, -1); cv2.circle(comp, p2, 10, col, -1)
        cv2.line(comp, p1, p2, col, 2)
    cv2.putText(comp, f"{n1} <-> {n2}  gap={gap:.0f}s  green<5px orange<30 red>30",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
    comp_tag = "composite_overlap" if args.crop_which == "low" else "composite_nonoverlap"
    comp_path = os.path.join(out, f"{args.plot}_{comp_tag}.jpg")
    cv2.imwrite(comp_path, comp)

    # crop montage: high-residual matches (suspected false) OR low-residual (should be same head)
    C = 90
    if args.crop_which == "low":
        cand = np.where(d < 5)[0]
        hi_sel = cand[np.argsort(d[cand])][:args.n_crops] if len(cand) else []
    else:
        hi_sel = hi_idx[np.argsort(-d[hi_idx])][:args.n_crops] if len(hi_idx) else []
    tiles = []
    for k in hi_sel:
        def crop(img, p):
            x, y = int(p[0]), int(p[1])
            pad = cv2.copyMakeBorder(img, C, C, C, C, cv2.BORDER_CONSTANT)
            t = pad[y:y + 2 * C, x:x + 2 * C].copy()
            cv2.drawMarker(t, (C, C), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
            return cv2.resize(t, (200, 200))
        row = np.hstack([crop(imgA, x1[k]), np.full((200, 8, 3), 255, np.uint8), crop(imgB, x2[k])])
        cv2.putText(row, f"{d[k]:.0f}px", (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(row)
    if tiles:
        body = np.vstack(tiles)
        # header strip explaining the figure (for the report)
        hdr = np.full((110, body.shape[1], 3), 30, np.uint8)
        cv2.putText(hdr, f"{args.field}/{args.plot}: high-residual matches between two photos",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(hdr, f"LEFT = patch in photo 1 ({n1})", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(hdr, f"RIGHT = patch in photo 2 ({n2}), gap {gap:.0f}s", (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(hdr, "red cross = matched keypoint;  red number = epipolar error (px)", (10, 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        montage = np.vstack([hdr, body])
        tag = "correct_matches_overlap" if args.crop_which == "low" else "false_match_crops"
        crops_path = os.path.join(out, f"{args.plot}_{tag}.jpg")
        cv2.imwrite(crops_path, montage)
        print("wrote", comp_path, "and", crops_path)
    else:
        print("wrote", comp_path)
    print(f"residual stats: <5px={np.mean(d<5)*100:.0f}%  5-30px={np.mean((d>=5)&(d<=30))*100:.0f}%  "
          f">30px={np.mean(d>30)*100:.0f}%  (n={len(d)})")


if __name__ == "__main__":
    main()
