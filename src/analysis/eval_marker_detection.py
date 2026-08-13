"""Experiment C: marker-detection accuracy vs hand-verified ground truth.

Compares OUR detected 2D marker positions (logs/marker_triangulation.json, src=="detected") against
the supervisor's hand-verified positions (agisoft/marker_projections.csv, Pinned=="True"). Both live
in the original 4032x3024 photo space (verified empirically: median agreement 0.8 px), so it is a
direct pixel comparison with no rescaling.

Reference source per session (--gt_source auto): the expanded HAND-VERIFIED ground truth in
input_plots/<session>/agisoft/marker_projections.csv when present (only field_D/20250627 --- the one
true GT), otherwise Agisoft's OWN marker projections (Pinned==True, mostly Agisoft's automatic
coded-target detection --- a reference, NOT ground truth) in
demoanlage2025_v0/.../<session>/processed/marker_projections.csv. On field_D/20250627 the two agree to
0 px on their overlap (the verified set just adds 204 more sightings), so on the other seven sessions
this measures our-detector-vs-Agisoft-detector AGREEMENT, not accuracy against truth. Reports, per session:
  - localization pixel error over correctly matched markers (median / mean / p90 / p95),
  - recall  = correctly detected sightings / all verified sightings (we process every image, so every
              verified sighting is an opportunity),
  - precision = correctly detected / all our detections,
  - gross mis-decodes (matched same code+image but far apart) and false positives (no GT there).
A detection is "correct" if it matches a GT pin of the same marker in the same image within --thresh px.
"""
import argparse
import csv
import json
import os
import statistics as st

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TARGET_TO_CODE = {1: 113, 2: 105, 3: 89, 4: 101, 5: 85, 6: 77}


def resolve_gt_csv(field, plot, source):
    """Pick the marker CSV: expanded hand-verified (input_plots) if present, else the demoanlage
    Agisoft pins. source in {auto, input_plots, demoanlage}."""
    ip = os.path.join(REPO, "input_plots", "phone", field, plot, "agisoft", "marker_projections.csv")
    dm = os.path.join(REPO, "demoanlage2025_v0", "demoanlage2025_v0_additions", field, plot,
                      "processed", "marker_projections.csv")
    if source == "input_plots":
        return ip, "input_plots(verified)"
    if source == "demoanlage":
        return dm, "demoanlage(agisoft-pins)"
    if os.path.exists(ip):
        return ip, "input_plots(verified)"
    return dm, "demoanlage(agisoft-pins)"


def load_gt(csv_path):
    """Hand-placed sightings: {(code, image_stem): (x, y)} from Pinned==True rows."""
    gt = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if str(r.get("Pinned")).strip().lower() != "true":
                continue
            code = TARGET_TO_CODE.get(int(r["Marker"].split()[-1]))
            stem = os.path.splitext(r["Camera"])[0]
            try:
                gt[(code, stem)] = (float(r["X"]), float(r["Y"]))
            except ValueError:
                pass
    return gt


def load_ours(base):
    """Our detections: {(code, image_stem): (x, y)} from triangulation obs with src=='detected'."""
    tri = json.load(open(os.path.join(base, "logs", "marker_triangulation.json")))
    ours = {}
    for code, obs in tri.items():
        for o in obs:
            if o.get("src") == "detected":
                ours[(int(code), os.path.splitext(o["cam"])[0])] = tuple(o["xy"])
    return ours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="field_D")
    ap.add_argument("--plot", default="20250627")
    ap.add_argument("--thresh", type=float, default=15.0, help="px: max distance for a correct match")
    ap.add_argument("--gt_source", default="auto", choices=["auto", "input_plots", "demoanlage"])
    args = ap.parse_args()
    base = os.path.join(REPO, "input_plots", "phone", args.field, args.plot)

    gt_csv, gt_tag = resolve_gt_csv(args.field, args.plot, args.gt_source)
    gt = load_gt(gt_csv)
    ours = load_ours(base)

    # classify every one of our detections against the GT pin of the same (code, image)
    loc_err, gross, fp = [], 0, 0
    for k, (ox, oy) in ours.items():
        if k in gt:
            gx, gy = gt[k]
            d = ((ox - gx) ** 2 + (oy - gy) ** 2) ** 0.5
            if d <= args.thresh:
                loc_err.append(d)
            else:
                gross += 1                      # same code+image but far -> mis-decode / mis-placed
        else:
            fp += 1                             # we called this code here, GT has no such pin

    n_correct = len(loc_err)
    n_gt = len(gt)                              # every verified sighting is an opportunity (all imgs processed)
    n_det = len(ours)
    recall = n_correct / n_gt if n_gt else 0.0
    precision = n_correct / n_det if n_det else 0.0

    def pct(x):
        return f"{x*100:.1f}%"

    print("+" + "-" * 60 + "+")
    print(f"| MARKER-DETECTION ACCURACY  {args.field}/{args.plot}  [{gt_tag}]".ljust(61) + "|")
    print("+" + "-" * 60 + "+")
    print(f"  reference sightings (Pinned):         {n_gt}")
    print(f"  our detections:                       {n_det}")
    print(f"  correct (<= {args.thresh:.0f} px):                    {n_correct}")
    print(f"  gross mis-decodes (matched, > thresh): {gross}")
    print(f"  false positives (no GT pin):           {fp}")
    print(f"  --")
    print(f"  RECALL    (correct / {n_gt} GT):          {pct(recall)}  ({n_correct}/{n_gt})")
    print(f"  PRECISION (correct / {n_det} det):         {pct(precision)}  ({n_correct}/{n_det})")
    if loc_err:
        loc_err.sort()
        p90 = loc_err[int(0.90 * (len(loc_err) - 1))]
        p95 = loc_err[int(0.95 * (len(loc_err) - 1))]
        print(f"  --")
        print(f"  localization error (correct matches, px):")
        print(f"     median {st.median(loc_err):.2f}   mean {st.mean(loc_err):.2f}   "
              f"p90 {p90:.2f}   p95 {p95:.2f}   max {max(loc_err):.2f}")
    print("+" + "-" * 60 + "+")


if __name__ == "__main__":
    main()
