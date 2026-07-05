#!/usr/bin/env python3
"""Verify whether two segmentation_3d runs produced the SAME result — deterministically, no eyeballing.

The definitive check is the **md5 of `all_obj_labels.pth`** (each Gaussian's head-ID). If that matches,
the two runs produced a bit-identical 3D segmentation. When it differs, the script drills into WHAT
differs (head count, matched masks, 2D IoU, results.csv rows) so you can see how far apart they are.

Exit code is 0 when identical / matches the expected md5, else 1 — so it can gate a script.

Usage:
    # compare two run folders
    python src/analysis/compare_seg_runs.py <run_a_dir> <run_b_dir>

    # check one run against a known-good reference md5 (the "gate")
    python src/analysis/compare_seg_runs.py <run_dir> --expect-md5 0bcd708dfe026d1a4ecd2f3f0d68c386

    # just fingerprint one run
    python src/analysis/compare_seg_runs.py <run_dir>

A "run folder" is a .../segmentation_3d/<exp_name>/ directory (it holds all_obj_labels.pth,
seg_summary.json, results.csv, and optionally eval_2d/metrics_2d.json).
"""
import argparse
import hashlib
import json
import os
import sys


def file_md5(path):
    """md5 of a file, or None if it doesn't exist."""
    if not os.path.isfile(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    """Parse a JSON file, or None if missing/unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def fingerprint(run):
    """Collect the identity-relevant numbers for one seg run folder."""
    summ = load_json(os.path.join(run, "seg_summary.json")) or {}
    m2 = load_json(os.path.join(run, "eval_2d", "metrics_2d.json"))
    iou = m2[-1].get("iou") if isinstance(m2, list) and m2 else None  # per-camera list; last == Mean row
    rcsv = os.path.join(run, "results.csv")
    nrows = sum(1 for _ in open(rcsv)) if os.path.isfile(rcsv) else None
    return {
        "md5": file_md5(os.path.join(run, "all_obj_labels.pth")),
        "heads": summ.get("wheat_heads_found"),
        "matched": summ.get("masks_matched"),
        "total": summ.get("total_masks"),
        "iou": iou,
        "csv_rows": nrows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", help="segmentation_3d/<exp> folder")
    ap.add_argument("run_b", nargs="?", default=None, help="second folder to compare against")
    ap.add_argument("--expect-md5", default=None, help="known-good all_obj_labels.pth md5 to gate against")
    args = ap.parse_args()

    a = fingerprint(args.run_a)
    if a["md5"] is None:
        sys.exit(f"ERROR: no all_obj_labels.pth in {args.run_a} (run not finished / wrong path)")

    # --- mode 1: compare two runs ---
    if args.run_b:
        b = fingerprint(args.run_b)
        if b["md5"] is None:
            sys.exit(f"ERROR: no all_obj_labels.pth in {args.run_b}")
        print(f"A: {args.run_a}")
        print(f"B: {args.run_b}\n")
        for name, key in [("all_obj_labels md5", "md5"), ("heads", "heads"),
                          ("matched masks", "matched"), ("total masks", "total"),
                          ("2D IoU", "iou"), ("results.csv rows", "csv_rows")]:
            va, vb = a[key], b[key]
            print(f"  [{'OK  ' if va == vb else 'DIFF'}] {name:20} A={va}  B={vb}")
        identical = a["md5"] == b["md5"]
        print()
        if identical:
            print("VERDICT: ✅ BYTE-IDENTICAL (same all_obj_labels.pth md5)")
        else:
            print("VERDICT: ❌ DIFFERENT — the 3D segmentation is NOT identical")
            if a["heads"] is not None and b["heads"] is not None:
                print(f"         head delta {(b['heads'] or 0) - (a['heads'] or 0):+d}, "
                      f"match delta {(b['matched'] or 0) - (a['matched'] or 0):+d}")
        sys.exit(0 if identical else 1)

    # --- mode 2: gate one run against an expected md5 ---
    if args.expect_md5:
        same = a["md5"] == args.expect_md5
        print(f"run      : {args.run_a}")
        print(f"md5      = {a['md5']}")
        print(f"expected = {args.expect_md5}")
        print(f"heads={a['heads']}  matched={a['matched']}  total={a['total']}  iou={a['iou']}")
        print("\nVERDICT: " + ("✅ MATCHES expected md5 — lossless" if same else "❌ md5 MISMATCH — NOT lossless"))
        sys.exit(0 if same else 1)

    # --- mode 3: fingerprint one run ---
    print(f"run: {args.run_a}")
    for k, v in a.items():
        print(f"  {k:12} = {v}")


if __name__ == "__main__":
    main()
