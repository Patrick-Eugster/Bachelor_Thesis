"""Score a marker-detector version against the Agisoft GT projections.

Replaces eyeballing with numbers, and makes a colour-coded overlay so manual
inspection is unambiguous.

Honest-scoring design (the loose-radius trap):
  - matching a detection to a GT marker within a BIG radius would let a dot sitting
    on a code-ARC count as "correct". So we DON'T reduce scoring to one radius:
  - RECALL is reported at several TIGHT tolerances (5/10/15/20/30/40 px). A real
    centre-hit is recalled even at 5px; v6's arc-drift passes @40 but fails @10 ->
    the gap exposes the drift, and can't be gamed by a loose radius.
  - LOCALIZATION error is a CONTINUOUS number (px distance det->GT), so an arc-hit
    shows up as a large error instead of a free pass.

Asymmetric GT (user point): Agisoft has ~zero false positives but MISSES many
visible plates. So a detection that matches no GT is NOT auto-wrong -> we label it a
"candidate extra" (yellow), to be confirmed by the decode, never auto-counted as FP.

Overlay colours (per image) — strong/high-contrast on green canopy + white plates:
  BLUE    = GT found (a detection within --tol)            [true positive]
  RED     = GT missed (no detection within --tol)          [false negative]
  MAGENTA = detection matching no GT                        [candidate extra]

READ-ONLY w.r.t. dataset. Overlays -> marker_vis_score_<version>/.

Usage:
  python src/preprocessing/score_markers_vs_gt.py --version v6
  python src/preprocessing/score_markers_vs_gt.py --version v6 --tol 12 --no-overlay
  python src/preprocessing/score_markers_vs_gt.py --version v2 v3 v4 v5 v6   # table only
"""

import os
import csv
import json
import math
import argparse
from collections import defaultdict

import cv2

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOLS = [5, 10, 15, 20, 30, 40]
ASSOC = 60.0  # px: generous radius just to ASSOCIATE a det to a GT (not for scoring)

# strong, high-contrast colours (BGR) — chosen to pop on BOTH green canopy and white
# plates (avoid green/yellow which blend into wheat). found=blue, missed=red, extra=magenta.
COL_FOUND = (255, 0, 0)      # blue
COL_MISSED = (0, 0, 255)     # red
COL_EXTRA = (255, 0, 255)    # magenta
LINE_TH = 4


def gt_csv(field, plot):
    return os.path.join(REPO, "demoanlage2025_v0", "demoanlage2025_v0_additions",
                        field, plot, "processed", "marker_projections.csv")


def load_gt(field, plot):
    by_cam = defaultdict(list)
    for r in csv.DictReader(open(gt_csv(field, plot))):
        by_cam[r["Camera"]].append((r["Marker"], float(r["X"]), float(r["Y"])))
    return by_cam


def load_dets(field, plot, version):
    p = os.path.join(REPO, "input_plots", "phone", field, plot, "logs",
                     f"marker_detections_{version}.json")
    per = json.load(open(p))["per_image"]
    out = {}
    for cam, lst in per.items():
        name = cam[:-4] if cam.lower().endswith(".jpg") else cam
        pts = []
        if isinstance(lst, list):
            for d in lst:
                c = d.get("center") if isinstance(d, dict) else d
                if c:
                    pts.append((float(c[0]), float(c[1])))
        out[name] = pts
    return out


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def match_image(gt_pts, det_pts):
    """Greedy one-to-one association within ASSOC radius (by nearest distance).
    Returns: matches [(gi, di, dist)], missed_gt idx set, extra_det idx set."""
    pairs = []
    for gi, (_, gx, gy) in enumerate(gt_pts):
        for di, d in enumerate(det_pts):
            dd = dist((gx, gy), d)
            if dd <= ASSOC:
                pairs.append((dd, gi, di))
    pairs.sort()
    used_g, used_d, matches = set(), set(), []
    for dd, gi, di in pairs:
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi); used_d.add(di); matches.append((gi, di, dd))
    missed = set(range(len(gt_pts))) - used_g
    extra = set(range(len(det_pts))) - used_d
    return matches, missed, extra


def score_version(field, plot, version, gt_by_cam, tol, overlay, limit):
    dets_by_cam = load_dets(field, plot, version)
    images_dir = os.path.join(REPO, "input_plots", "phone", field, plot, "images")
    out_dir = os.path.join(REPO, "input_plots", "phone", field, plot,
                           f"marker_vis_score_{version}")
    if overlay:
        os.makedirs(out_dir, exist_ok=True)

    cams = sorted(gt_by_cam.keys())
    if limit:
        cams = cams[:limit]

    n_gt = n_extra = 0
    recall_hits = {t: 0 for t in TOLS}
    loc_errors = []           # nearest-det distance for each GT that has a det within ASSOC

    for cam in cams:
        gt_pts = gt_by_cam[cam]
        det_pts = dets_by_cam.get(cam, [])
        n_gt += len(gt_pts)
        # nearest det distance per GT (for recall@r + loc error)
        for (_, gx, gy) in gt_pts:
            if det_pts:
                nd = min(dist((gx, gy), d) for d in det_pts)
                for t in TOLS:
                    if nd <= t:
                        recall_hits[t] += 1
                if nd <= ASSOC:
                    loc_errors.append(nd)
        matches, missed, extra = match_image(gt_pts, det_pts)
        n_extra += len(extra)

        if overlay:
            img = cv2.imread(os.path.join(images_dir, cam + ".jpg"))
            if img is None:
                continue
            rad = max(14, int(0.014 * max(img.shape[:2])))
            matched_g = {gi for gi, _, dd in matches if dd <= tol}
            for gi, (mk, gx, gy) in enumerate(gt_pts):
                p = (int(round(gx)), int(round(gy)))
                col = COL_FOUND if gi in matched_g else COL_MISSED
                cv2.circle(img, p, rad, col, LINE_TH)
                cv2.drawMarker(img, p, col, cv2.MARKER_CROSS, rad, LINE_TH)
                cv2.putText(img, mk.replace("target ", "T"), (p[0] + rad + 3, p[1] - rad),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 3)
            matched_d = {di for _, di, dd in matches if dd <= tol}
            for di, d in enumerate(det_pts):
                p = (int(round(d[0])), int(round(d[1])))
                if di in matched_d:
                    cv2.circle(img, p, 7, COL_FOUND, -1)            # matched det = blue dot
                else:
                    cv2.circle(img, p, rad, COL_EXTRA, LINE_TH)     # extra = magenta
            cv2.imwrite(os.path.join(out_dir, cam + ".jpg"), img)

    # aggregate
    loc_errors.sort()
    med = loc_errors[len(loc_errors) // 2] if loc_errors else float("nan")
    p90 = loc_errors[int(0.9 * len(loc_errors))] if loc_errors else float("nan")
    row = {
        "version": version, "n_gt": n_gt, "n_extra": n_extra,
        "loc_med": med, "loc_p90": p90,
        **{f"R@{t}": recall_hits[t] / n_gt for t in TOLS},
    }
    if overlay:
        row["overlay"] = out_dir
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", nargs="+", default=["v6"])
    ap.add_argument("--field", default="field_A")
    ap.add_argument("--plot", default="20250609")
    ap.add_argument("--tol", type=int, default=15, help="overlay green/red threshold (px)")
    ap.add_argument("--no-overlay", dest="overlay", action="store_false")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    gt = load_gt(args.field, args.plot)
    n_gt_total = sum(len(v) for v in gt.values())
    print(f"GT: {n_gt_total} marker projections over {len(gt)} images "
          f"({args.field}/{args.plot})")
    print(f"recall@r = fraction of GT markers with a detection within r px "
          f"(higher=better); loc=px error (lower=better); extras=candidate (NOT auto-FP)\n")

    rows = [score_version(args.field, args.plot, v, gt, args.tol, args.overlay, args.limit)
            for v in args.version]

    hdr = ["ver", "n_gt"] + [f"R@{t}" for t in TOLS] + ["loc_med", "loc_p90", "extras"]
    print("  ".join(f"{h:>7}" for h in hdr))
    for r in rows:
        cells = [r["version"], r["n_gt"]] + [f"{r[f'R@{t}']:.2f}" for t in TOLS] + \
                [f"{r['loc_med']:.1f}", f"{r['loc_p90']:.1f}", r["n_extra"]]
        print("  ".join(f"{str(c):>7}" for c in cells))
    if args.overlay:
        print("\noverlays:")
        for r in rows:
            print(f"  {r['version']}: {r.get('overlay')}")


if __name__ == "__main__":
    main()
