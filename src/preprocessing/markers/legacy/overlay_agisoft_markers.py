"""Overlay Agisoft's ground-truth marker projections onto the images.

Agisoft (in demoanlage2025_v0_additions/.../processed/marker_projections.csv) gives,
per image, the 2D pixel position of each coded marker it detected:
    Marker, Camera, X, Y, Pinned
This script draws those GT points on the images so we can (a) see which markers
Agisoft found in which views, and (b) use them as ground truth to score our own
marker detectors (v1-v6, v7-CCT) and to cut clean, correctly-centred marker crops.

It is READ-ONLY w.r.t. the dataset: it only writes overlay PNGs to OUT_SUBDIR.

Coordinate-space note: the CSV camera names match our undistorted images/ filenames,
but the bundled reconstruction is colmap_distorted (FULL_OPENCV). Whether the X,Y are
in undistorted (our images/) or distorted space is resolved empirically by running
this on --images-dir images and eyeballing whether edge markers land correctly.

Usage:
  python src/preprocessing/overlay_agisoft_markers.py            # field_A/20250609, our images/
  python src/preprocessing/overlay_agisoft_markers.py --limit 6
  python src/preprocessing/overlay_agisoft_markers.py --images-dir <dir> --name-suffix _DISTORTED
"""

import os
import csv
import glob
import argparse
from collections import defaultdict, Counter

import cv2

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# distinct BGR colours per marker id (target 1..6)
COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0),
    (0, 255, 255), (255, 0, 255), (255, 255, 0),
    (0, 128, 255), (128, 0, 255),
]


def default_csv(field, plot):
    return os.path.join(
        REPO, "demoanlage2025_v0", "demoanlage2025_v0_additions",
        field, plot, "processed", "marker_projections.csv")


def load_projections(csv_path):
    """Read marker_projections.csv -> {camera: [(marker, x, y, pinned), ...]}.
    Also returns a Counter of how many views each marker appears in."""
    by_cam = defaultdict(list)
    per_marker = Counter()
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            cam = r["Camera"]
            mk = r["Marker"]
            x, y = float(r["X"]), float(r["Y"])
            pinned = str(r.get("Pinned", "")).strip().lower() == "true"
            by_cam[cam].append((mk, x, y, pinned))
            per_marker[mk] += 1
    return by_cam, per_marker


def marker_color(marker):
    """Stable colour from the trailing number of 'target N' (fallback: hash)."""
    digits = "".join(ch for ch in marker if ch.isdigit())
    idx = (int(digits) - 1) if digits else hash(marker)
    return COLORS[idx % len(COLORS)]


def find_image(images_dir, cam, suffix, exts=(".jpg", ".JPG", ".png", ".jpeg")):
    """Locate the image file for a camera name, optionally with a rename suffix
    (Agisoft distorted images are named <cam>_<N>.jpg). Suffix '*' globs any."""
    for ext in exts:
        p = os.path.join(images_dir, cam + suffix + ext)
        if os.path.isfile(p):
            return p
    if suffix == "*":
        hits = glob.glob(os.path.join(images_dir, glob.escape(cam) + "_*"))
        hits = [h for h in hits if not h.endswith("Zone.Identifier")]
        if hits:
            return sorted(hits)[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="field_A")
    ap.add_argument("--plot", default="20250609")
    ap.add_argument("--csv", default=None, help="override marker_projections.csv path")
    ap.add_argument("--images-dir", default=None,
                    help="dir with images to overlay on (default: our undistorted images/)")
    ap.add_argument("--name-suffix", default="",
                    help="suffix between camera name and extension (e.g. '*' for Agisoft _N names)")
    ap.add_argument("--out-subdir", default="marker_vis_agisoft_gt")
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    args = ap.parse_args()

    session = os.path.join(REPO, "input_plots", "phone", args.field, args.plot)
    csv_path = args.csv or default_csv(args.field, args.plot)
    images_dir = args.images_dir or os.path.join(session, "images")
    out_dir = os.path.join(session, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(csv_path):
        raise SystemExit(f"marker_projections.csv not found: {csv_path}")

    by_cam, per_marker = load_projections(csv_path)
    print(f"CSV:        {csv_path}")
    print(f"images-dir: {images_dir}  (suffix '{args.name_suffix}')")
    print(f"out:        {out_dir}")
    print(f"markers (total views each): {dict(per_marker)}")
    print(f"cameras with >=1 marker:    {len(by_cam)}")

    cams = sorted(by_cam.keys())
    if args.limit:
        cams = cams[:args.limit]

    missing = drawn = 0
    for cam in cams:
        img_path = find_image(images_dir, cam, args.name_suffix)
        if img_path is None:
            missing += 1
            continue
        img = cv2.imread(img_path)
        if img is None:
            missing += 1
            continue
        H, W = img.shape[:2]
        rad = max(10, int(0.012 * max(H, W)))   # dot radius scales with image
        for mk, x, y, pinned in by_cam[cam]:
            c = marker_color(mk)
            p = (int(round(x)), int(round(y)))
            cv2.circle(img, p, rad, c, 3)
            cv2.drawMarker(img, p, c, cv2.MARKER_CROSS, rad, 2)  # crosshair = exact pt
            label = mk.replace("target ", "T")
            cv2.putText(img, label, (p[0] + rad + 4, p[1] - rad),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, c, 3)
        cv2.imwrite(os.path.join(out_dir, cam + ".jpg"), img)
        drawn += 1

    print(f"\noverlays written: {drawn}   images not found: {missing}")
    print(f"inspect: {out_dir}")


if __name__ == "__main__":
    main()
