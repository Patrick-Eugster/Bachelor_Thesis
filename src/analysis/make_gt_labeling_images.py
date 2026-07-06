#!/usr/bin/env python3
"""Produce marker-ROI-masked copies of the selected ground-truth images, for hand-labeling.

For each image in the GT selection we grey-out everything outside the plot's marker polygon
(reusing the pipeline's roi_mask.apply_roi, same buffer as the phone runs) and save the copy
to input_plots/phone/<field>/<date>/<out_subdir>/<stem>.jpg. Labeling on this greyed image
means you only see the plot, so you won't box neighbour-plot heads — and since the grey-out
only touches pixels OUTSIDE the polygon, the boxes are valid on the original image too.

The originals in images/ are never modified.

Usage:
    python src/analysis/make_gt_labeling_images.py                 # uses the default selection json
    python src/analysis/make_gt_labeling_images.py --selection /path/to/gt_selection.json
    python src/analysis/make_gt_labeling_images.py --buffer_frac 0.05 --out_subdir gt_labeling
"""
import argparse
import glob
import json
import os
import sys

import cv2

# reuse the pipeline's ROI projection + greyout — do NOT reimplement
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mask_generation"))
import roi_mask  # noqa: E402

PHONE_ROOT = "/workspace/input_plots/phone"
DEFAULT_SELECTION = ("/tmp/claude-0/-workspace/e6a109d2-41f2-481c-a1e9-2e00efdd4d44/"
                     "scratchpad/gt_selection.json")


def build_cfg(buffer_frac):
    """The roi config block apply_roi expects — same settings as the phone mask-gen runs."""
    return {
        "roi": {
            "enabled": True,
            "source": "markers",
            "min_markers": 3,
            "buffer_frac": buffer_frac,   # grow the plot polygon so boundary heads stay visible
            "buffer_px": 0,
            "fallback": "none",           # no markers -> save the plain image (all sessions have markers)
            "fill": [114, 114, 114],      # neutral grey outside the ROI
        }
    }


def find_image(session_dir, stem):
    """Locate images/<stem>.* (phone is .jpg). Returns path or None."""
    hits = glob.glob(os.path.join(session_dir, "images", stem + ".*"))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=DEFAULT_SELECTION, help="gt_selection.json ({session: [{stem,...}]})")
    ap.add_argument("--buffer_frac", type=float, default=0.05, help="ROI buffer as fraction of image short side")
    ap.add_argument("--out_subdir", default="gt_labeling", help="folder name created inside each session")
    args = ap.parse_args()

    with open(args.selection) as f:
        selection = json.load(f)
    cfg = build_cfg(args.buffer_frac)

    n_ok, n_roi, n_fallback, n_missing = 0, 0, 0, 0
    for session, picks in selection.items():
        session_dir = os.path.join(PHONE_ROOT, session)
        out_dir = os.path.join(session_dir, args.out_subdir)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== {session} → {args.out_subdir}/ ===")
        for p in picks:
            stem = p["stem"]
            img_path = find_image(session_dir, stem)
            if img_path is None:
                print(f"   MISSING  {stem}  (no images/{stem}.* found)")
                n_missing += 1
                continue
            img = cv2.imread(img_path)  # BGR; apply_roi's grey fill is channel-symmetric
            if img is None:
                print(f"   UNREADABLE  {stem}")
                n_missing += 1
                continue
            h, w = img.shape[:2]
            # did this image get a real marker polygon, or would it fall back?
            poly, _ = roi_mask._base_polygon(img_path, cfg, w, h)
            tag = "ROI(marker)" if poly is not None else "full-frame(fallback)"
            if poly is not None:
                n_roi += 1
            else:
                n_fallback += 1
            out = roi_mask.apply_roi(img, img_path, cfg)
            out_path = os.path.join(out_dir, stem + ".jpg")
            cv2.imwrite(out_path, out)
            n_ok += 1
            print(f"   {tag:22} {stem}")

    print(f"\nDone: wrote {n_ok} images  (marker-ROI {n_roi}, fallback {n_fallback}, missing {n_missing})")
    print(f"Each session now has a '{args.out_subdir}/' folder — load it into yolo-mark-pwa, box the heads,")
    print("then move the exported <stem>.txt into that session's manual_label/ folder.")


if __name__ == "__main__":
    main()
