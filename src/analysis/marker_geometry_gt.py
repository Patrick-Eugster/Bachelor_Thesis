"""Validate our triangulated marker GEOMETRY against TWO independent physical references — tape
marker-to-marker distances and surveyed marker XYZ — using all 15 pairwise distances (not just scale).

Distances are FRAME-INVARIANT, so no alignment is needed; we apply the metric scale from marker_scale.json
(tape-derived) so our lengths are in cm. Reports, per session:
  - our vs SURVEY  (survey did NOT set our scale -> independent-ish)
  - our vs TAPE    (tape DID set our scale -> partly circular; shown for completeness)
  - TAPE vs SURVEY (the ground-truth-vs-ground-truth disagreement = the resolution FLOOR: we can't claim
    to be better than the two references agree with each other)

Usage:  python src/analysis/marker_geometry_gt.py            # all sessions
        python src/analysis/marker_geometry_gt.py --field field_D --plot 20250706
"""

import os
import re
import glob
import json
import argparse
from itertools import combinations

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
META = os.path.join(REPO, "demoanlage2025_v0", "metadata", "markers")
TAPE_XLSX = os.path.join(META, "Demoanlage-2025-markers-manual-distances.xlsx")
TARGET_TO_CODE = {1: 113, 2: 105, 3: 89, 4: 101, 5: 85, 6: 77}


def load_survey_cm(field):
    """{code: XYZ in cm} from field_<L>_coordinates.txt (survey metres -> cm)."""
    path = os.path.join(META, f"{field}_coordinates.txt")
    out = {}
    for line in open(path):
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        parts = line.split(",")
        m = re.match(r"target\s*(\d+)", parts[0])
        if not m:
            continue
        t = int(m.group(1))
        if t in TARGET_TO_CODE and len(parts) >= 4:
            out[TARGET_TO_CODE[t]] = np.array([float(parts[1]), float(parts[2]), float(parts[3])]) * 100.0
    return out


def load_tape_cm(field):
    """{frozenset(codeA,codeB): distance_cm} from the 'plot <L>' sheet (upper-triangular matrix)."""
    letter = field.split("_")[-1]
    df = pd.read_excel(TAPE_XLSX, sheet_name=f"plot {letter}", header=None)
    tnum_col = {j: df.iloc[0, j] for j in range(1, df.shape[1])}
    out = {}
    for i in range(1, df.shape[0]):
        ti = df.iloc[i, 0]
        for j in range(1, df.shape[1]):
            v = df.iloc[i, j]
            tj = tnum_col[j]
            if pd.notna(v) and pd.notna(ti) and pd.notna(tj):
                ci, cj = TARGET_TO_CODE.get(int(ti)), TARGET_TO_CODE.get(int(tj))
                if ci and cj and ci != cj:
                    out[frozenset((ci, cj))] = float(v)
    return out


def our_positions_cm(sess):
    """{code: XYZ in cm} = our triangulated marker xyz * metric scale (m) * 100."""
    d = json.load(open(os.path.join(sess, "logs", "marker_points3d.json")))
    sc = json.load(open(os.path.join(sess, "logs", "marker_scale.json"))).get("scale_metric")
    if sc is None:
        return None
    return {int(k): np.array(v["xyz"]) * sc * 100.0 for k, v in d["points3d"].items()}


def dists(pos_cm):
    return {frozenset((a, b)): float(np.linalg.norm(pos_cm[a] - pos_cm[b]))
            for a, b in combinations(pos_cm, 2)}


def med_abs_err(da, db):
    keys = set(da) & set(db)
    e = [abs(da[k] - db[k]) for k in keys]
    return (float(np.median(e)), len(e)) if e else (None, 0)


def process(field, plot):
    sess = os.path.join(REPO, "input_plots", "phone", field, plot)
    if not os.path.exists(os.path.join(sess, "logs", "marker_points3d.json")):
        return None
    ours = our_positions_cm(sess)
    if not ours or len(ours) < 3:
        return None
    survey = load_survey_cm(field)
    try:
        tape = load_tape_cm(field)
    except Exception:
        tape = {}
    d_our = dists(ours)
    d_sur = dists(survey) if len(survey) >= 3 else {}
    d_tap = tape
    o_sur = med_abs_err(d_our, d_sur)
    o_tap = med_abs_err(d_our, d_tap)
    t_sur = med_abs_err(d_tap, d_sur)   # GT-vs-GT floor
    return {"field": field, "plot": plot,
            "our_vs_survey_cm": o_sur, "our_vs_tape_cm": o_tap, "tape_vs_survey_cm": t_sur}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default=None)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()
    if args.field and args.plot:
        sessions = [(args.field, args.plot)]
    else:
        sessions = [(f, os.path.basename(p)) for f in ("field_A", "field_D")
                    for p in sorted(glob.glob(os.path.join(REPO, "input_plots", "phone", f, "2025*")))]

    print("MARKER GEOMETRY vs physical GT — median abs error over the (up to 15) pairwise distances, cm\n")
    print(f"  {'session':<28} {'our vs SURVEY':>14} {'our vs TAPE':>12} {'TAPE vs SURVEY (floor)':>22}")
    tvs = []
    for field, plot in sessions:
        r = process(field, plot)
        if r is None:
            continue
        def fmt(x):
            return f"{x[0]:.2f} (n{x[1]})" if x[0] is not None else "-"
        print(f"  {field+'/'+plot:<28} {fmt(r['our_vs_survey_cm']):>14} {fmt(r['our_vs_tape_cm']):>12} "
              f"{fmt(r['tape_vs_survey_cm']):>22}")
        if r["tape_vs_survey_cm"][0] is not None:
            tvs.append(r["tape_vs_survey_cm"][0])
    if tvs:
        print(f"\n  GT-vs-GT floor (tape vs survey) median across sessions = {np.median(tvs):.2f} cm "
              f"-> we cannot meaningfully claim marker geometry better than this.")
    print("  (our scale comes FROM tape, so 'our vs TAPE' is partly circular; 'our vs SURVEY' is the honest one.)")


if __name__ == "__main__":
    main()
