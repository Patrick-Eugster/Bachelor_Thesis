"""Phase 1 test: forced-centre decode at GT positions, scored with the same table.

Same harness/markers as phase0b, but instead of stock CCT_extract (which blob-searches
and grabs arcs) we call decode_at_center() — forcing the decode onto the disk at the
GT centre. If targets 2/3/6 flip orange->green here, forcing the centre is the fix.
This isolates the DECODER fix (centre supplied by GT, no detector yet).

Run:
  python src/preprocessing/test_cct_phase1_forced.py
  python src/preprocessing/test_cct_phase1_forced.py --ring 2.5
"""
import os
import sys
import csv
import argparse
from collections import defaultdict, Counter

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from cct_forced_decode import decode_at_center, DEFAULT_CFG  # noqa: E402

DEGENERATE = {0, 4095, 2047}


def gt_path(field, plot):
    return os.path.join(REPO, "demoanlage2025_v0", "demoanlage2025_v0_additions",
                        field, plot, "processed", "marker_projections.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="field_A")
    ap.add_argument("--plot", default="20250609")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    images_dir = os.path.join(REPO, "input_plots", "phone", args.field, args.plot, "images")
    by_cam = defaultdict(list)
    for r in csv.DictReader(open(gt_path(args.field, args.plot))):
        by_cam[r["Camera"]].append((r["Marker"], float(r["X"]), float(r["Y"])))
    cams = sorted(by_cam.keys())
    if args.limit:
        cams = cams[:args.limit]

    # decode one code per GT marker (forced at its centre)
    records = []                       # (mk, code|None)
    codes_by_target = defaultdict(Counter)
    n_disk = n_valid = 0
    for cam in cams:
        img = cv2.imread(os.path.join(images_dir, cam + ".jpg"))
        if img is None:
            continue
        for mk, x, y in by_cam[cam]:
            code, info = decode_at_center(img, x, y, DEFAULT_CFG)
            n_disk += info["disk"]
            n_valid += info["valid_cct"]
            if code is not None and code not in DEGENERATE:
                codes_by_target[mk][code] += 1
                records.append((mk, code))
            else:
                records.append((mk, None))

    mode = {mk: (c.most_common(1)[0][0] if c else None) for mk, c in codes_by_target.items()}
    owners = defaultdict(list)
    for mk, m in mode.items():
        if m is not None:
            owners[m].append(mk)
    collisions = {m for m, ts in owners.items() if len(ts) > 1}

    # per-marker green/red/orange/gray counts (same logic as the phase0b table)
    by_marker = defaultdict(Counter)
    for mk, code in records:
        m = mode.get(mk)
        if m is not None and m in collisions:
            st = "orange"
        elif code is None:
            st = "gray"
        elif code == m:
            st = "green"
        else:
            st = "red"
        by_marker[mk][st] += 1

    n = len(records)
    print(f"forced-centre decode at GT positions ({args.field}/{args.plot})")
    print(f"disk found: {n_disk}/{n}   valid-CCT: {n_valid}/{n}\n")
    print(f"{'marker':>9} {'views':>5} | {'green':>5} {'red':>4} {'orange':>6} {'gray':>4}  verdict")
    for mk in sorted(by_marker):
        c = by_marker[mk]; tot = sum(c.values())
        g, red, orng, gry = c["green"], c["red"], c["orange"], c["gray"]
        if orng:
            v = f"ORANGE (id {mode[mk]} collides)"
        elif g >= red + gry and g > 0:
            v = f"mostly GREEN (id {mode[mk]})"
        elif g > 0:
            v = f"mixed (id {mode[mk]})"
        else:
            v = "no good views"
        print(f"  {mk:>9} {tot:>5} | {g:>5} {red:>4} {orng:>6} {gry:>4}  {v}")
    print(f"\nPER-TARGET decoded-code spread:")
    for mk in sorted(codes_by_target):
        print(f"  {mk:>9}: {dict(codes_by_target[mk])}")


if __name__ == "__main__":
    main()
