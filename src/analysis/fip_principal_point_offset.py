"""Measures how far each camera's principal point sits from the image center.

For every cameras.txt under a dataset, reads (cx, cy) from the intrinsics and reports the
offset from the image center: horizontal dcx = cx - W/2, vertical dcy = cy - H/2, and the
2D magnitude sqrt(dcx^2 + dcy^2). The magnitude is the honest "off center by" number, since
averaging the signed dcx/dcy across plots would let opposite signs cancel out.

This is the source of the "off center by up to ~90 px" statement in the thesis. Run from the
repo root:
    python src/analysis/fip_principal_point_offset.py                       # default: FIP
    python src/analysis/fip_principal_point_offset.py --dataset phone       # phone (expect ~0)
"""

import argparse
import glob
import math
import os


def read_principal_point(path):
    """Parses one COLMAP cameras.txt and returns (W, H, cx, cy) for its first camera."""
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        W, H, model, params = int(p[2]), int(p[3]), p[1], list(map(float, p[4:]))
        # cx, cy live at different offsets depending on the camera model
        if model == "PINHOLE":
            cx, cy = params[2], params[3]
        elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            cx, cy = params[1], params[2]
        else:  # OPENCV etc. keep cx, cy as the last two positional params before distortion
            cx, cy = params[2], params[3]
        return W, H, cx, cy
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fip", help="input_plots/<dataset> to scan")
    args = ap.parse_args()

    pattern = f"input_plots/{args.dataset}/**/sparse/0/cameras.txt"
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print("no cameras.txt found under", pattern)
        return

    mags = []
    print(f"{'plot':<28}{'WxH':<12}{'dcx':>6}{'dcy':>6}{'|offset|':>10}")
    for f in files:
        got = read_principal_point(f)
        if not got:
            continue
        W, H, cx, cy = got
        dcx, dcy = cx - W / 2, cy - H / 2
        mag = math.hypot(dcx, dcy)
        mags.append(mag)
        # a short label from the path, e.g. plot_467 or field_A/20250715
        label = f.replace("input_plots/" + args.dataset + "/", "").replace("/sparse/0/cameras.txt", "")
        print(f"{label:<28}{f'{W}x{H}':<12}{dcx:>+6.0f}{dcy:>+6.0f}{mag:>10.1f}")

    if mags:
        print("-" * 62)
        print(f"2D offset magnitude: min {min(mags):.1f} px, max {max(mags):.1f} px, "
              f"mean {sum(mags)/len(mags):.1f} px  (n={len(mags)})")


if __name__ == "__main__":
    main()
