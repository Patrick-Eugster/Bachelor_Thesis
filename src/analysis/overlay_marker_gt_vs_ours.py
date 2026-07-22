"""Overlay Agisoft GROUND-TRUTH marker positions vs OUR triangulated positions, per image.

For a phone session we have two independent sources for each coded marker:
  1. Agisoft GT  -> per-image 2D pixel where Agisoft detected the marker
                    (demoanlage .../processed/marker_projections.csv, keyed "target 1..6")
  2. Ours        -> our triangulated 3D point (logs/marker_points3d.json) projected into
                    each image using the sparse/0 pose + intrinsics.

This draws BOTH on every image so we can see, per view:
  - solid circle + "IDgt"  = Agisoft ground truth (the real marker location)
  - cross      + "ID"      = our estimate (3D point projected in)
  - a line between them     = our error for that marker in that view (long line = we're off)

Agisoft uses "target 1..6"; we key everything by the coded ID via TARGET_TO_CODE.
READ-ONLY on the dataset: only writes overlay JPGs to the output folder.

Usage:
  python src/analysis/overlay_marker_gt_vs_ours.py --field field_D --plot 20250706
"""

import os
import csv
import glob
import json
import argparse

import numpy as np
import cv2

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Agisoft target number -> our coded ID (from src/preprocessing/marker_scale.py)
TARGET_TO_CODE = {1: 113, 2: 105, 3: 89, 4: 101, 5: 85, 6: 77}

# a distinct BGR colour per coded ID so the same marker is one colour everywhere
COLORS = {77: (0, 0, 255), 85: (0, 165, 255), 89: (0, 255, 255),
          101: (0, 255, 0), 105: (255, 128, 0), 113: (255, 0, 255)}


def load_gt(csv_path):
    """Read Agisoft's per-image marker pixels, keyed by our coded ID.
    Returns {image_stem: {code: (x, y)}}. Skips targets not in the mapping."""
    gt = {}
    for r in csv.DictReader(open(csv_path)):
        tnum = int(r["Marker"].replace("target", "").strip())
        if tnum not in TARGET_TO_CODE:
            continue
        code = TARGET_TO_CODE[tnum]
        gt.setdefault(r["Camera"], {})[code] = (float(r["X"]), float(r["Y"]))
    return gt


def load_intrinsics(cameras_txt):
    """Grab SIMPLE_PINHOLE f, cx, cy from a COLMAP cameras.txt (single shared camera)."""
    for line in open(cameras_txt):
        if line.startswith("#"):
            continue
        p = line.split()
        # SIMPLE_PINHOLE row: CAM_ID MODEL W H f cx cy  (7 tokens)
        if len(p) >= 7 and p[0].isdigit():
            f, cx, cy = float(p[4]), float(p[5]), float(p[6])
            return f, cx, cy
    raise RuntimeError("no camera found")


def load_poses(images_txt):
    """Read COLMAP image poses -> {name: (R, t)} (world->cam)."""
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


def project(xyz, R, t, f, cx, cy):
    """Project a world point into the image; returns (u, v) or None if behind the camera."""
    Xc = R @ xyz + t
    if Xc[2] <= 0:
        return None
    return f * Xc[0] / Xc[2] + cx, f * Xc[1] / Xc[2] + cy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="field_D")
    ap.add_argument("--plot", default="20250706")
    ap.add_argument("--downscale", type=int, default=2, help="save at 1/N size (default 2)")
    ap.add_argument("--out-subdir", default="marker_vis_agisoft_gt")
    args = ap.parse_args()

    sess = os.path.join(REPO, "input_plots", "phone", args.field, args.plot)
    csv_path = os.path.join(REPO, "demoanlage2025_v0", "demoanlage2025_v0_additions",
                            args.field, args.plot, "processed", "marker_projections.csv")
    out_dir = os.path.join(sess, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    gt = load_gt(csv_path)
    f, cx, cy = load_intrinsics(os.path.join(sess, "sparse", "0", "cameras.txt"))
    poses = load_poses(os.path.join(sess, "sparse", "0", "images.txt"))
    pts3d = {int(k): np.array(v["xyz"])
             for k, v in json.load(open(os.path.join(sess, "logs", "marker_points3d.json")))["points3d"].items()}

    imgs = sorted(glob.glob(os.path.join(sess, "images", "*.jpg")))
    print(f"{len(imgs)} images; GT covers {len(gt)} images; markers 3D: {sorted(pts3d)}")
    per_marker_gap = {c: [] for c in pts3d}

    for path in imgs:
        name = os.path.basename(path)
        stem = name[:-4]
        img = cv2.imread(path)
        H, W = img.shape[:2]
        R_t = poses.get(name)

        # our estimate: project every triangulated 3D marker into this image (cross)
        ours = {}
        if R_t is not None:
            for code, xyz in pts3d.items():
                uv = project(xyz, R_t[0], R_t[1], f, cx, cy)
                if uv is not None:
                    ours[code] = uv

        gt_here = gt.get(stem, {})

        # draw the error line first (so the markers sit on top)
        for code, (gx, gy) in gt_here.items():
            if code in ours:
                ux, uy = ours[code]
                d = ((ux - gx) ** 2 + (uy - gy) ** 2) ** 0.5
                per_marker_gap[code].append(d)
                cv2.line(img, (int(gx), int(gy)),
                         (int(np.clip(ux, 0, W)), int(np.clip(uy, 0, H))), (0, 255, 255), 4)

        # GT = solid circle + "IDgt"  (the truth)
        for code, (gx, gy) in gt_here.items():
            col = COLORS.get(code, (255, 255, 255))
            cv2.circle(img, (int(gx), int(gy)), 34, col, -1)
            cv2.circle(img, (int(gx), int(gy)), 34, (0, 0, 0), 3)
            cv2.putText(img, f"{code}gt", (int(gx) + 40, int(gy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, col, 5)

        # ours = hollow cross + "ID"  (our estimate)
        for code, (ux, uy) in ours.items():
            col = COLORS.get(code, (255, 255, 255))
            p = (int(np.clip(ux, 4, W - 4)), int(np.clip(uy, 4, H - 4)))
            cv2.drawMarker(img, p, col, cv2.MARKER_TILTED_CROSS, 70, 8)
            cv2.putText(img, f"{code}", (p[0] + 30, p[1] + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, col, 5)

        small = cv2.resize(img, (W // args.downscale, H // args.downscale))
        cv2.imwrite(os.path.join(out_dir, stem + "_gt.jpg"), small)

    print(f"\nwrote overlays to {out_dir}")
    print("\nper-marker median error (our projection vs Agisoft GT), where both exist:")
    for code in sorted(per_marker_gap):
        g = per_marker_gap[code]
        if g:
            print(f"   marker {code}: n={len(g):2d}  median={np.median(g):7.1f}px  max={np.max(g):7.1f}px")
        else:
            print(f"   marker {code}: no overlapping GT+ours frames")


if __name__ == "__main__":
    main()
