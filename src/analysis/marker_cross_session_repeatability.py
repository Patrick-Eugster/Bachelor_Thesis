"""Cross-session marker REPEATABILITY via pairwise DISTANCES (robust to the near-coplanar marker geometry).

The 6 markers are physically STATIC all season, so every session should reproduce the same marker geometry.
We measure repeatability as the consistency of the (up to 15) marker-to-marker DISTANCES across sessions.

WHY DISTANCES, not aligned positions: the markers are near-coplanar (survey Z spans ~10 cm vs ~2 m
in-plane), which makes 3D Umeyama alignment ill-conditioned out-of-plane (it flips/tilts) and hypersensitive
to a single mis-triangulated marker — an early alignment-based version reported 70 cm "distortion" for
sessions whose geometry is actually fine to ~1.7 cm. Pairwise distances are frame- and flip-invariant and,
with a median, robust to one bad marker. So per distance we report the spread (std) across sessions; low =
repeatable. Also flags which sessions deviate most from the cross-session median geometry.

Precision (repeatability), not absolute accuracy; ground-plane only.

Usage:  python src/analysis/marker_cross_session_repeatability.py --field field_D
"""

import os
import re
import glob
import json
import argparse

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
META = os.path.join(REPO, "demoanlage2025_v0", "metadata", "markers")
TARGET_TO_CODE = {1: 113, 2: 105, 3: 89, 4: 101, 5: 85, 6: 77}


def load_survey_cm(field):
    out = {}
    for line in open(os.path.join(META, f"{field}_coordinates.txt")):
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        p = line.split(",")
        m = re.match(r"target\s*(\d+)", p[0])
        if m and int(m.group(1)) in TARGET_TO_CODE and len(p) >= 4:
            out[TARGET_TO_CODE[int(m.group(1))]] = np.array([float(p[1]), float(p[2]), float(p[3])]) * 100.0
    return out


def our_positions_cm(sess):
    pj = os.path.join(sess, "logs", "marker_points3d.json")
    sj = os.path.join(sess, "logs", "marker_scale.json")
    if not (os.path.exists(pj) and os.path.exists(sj)):
        return None
    sc = json.load(open(sj)).get("scale_metric")
    if sc is None:
        return None
    d = json.load(open(pj))
    return {int(k): np.array(v["xyz"]) * sc * 100.0 for k, v in d["points3d"].items()}


def pair_dists(pos):
    from itertools import combinations
    return {frozenset((a, b)): float(np.linalg.norm(pos[a] - pos[b])) for a, b in combinations(pos, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default=None, help="field_A or field_D (default: both, reported separately)")
    args = ap.parse_args()
    fields = [args.field] if args.field else ["field_A", "field_D"]

    for field in fields:
        sessions = sorted(glob.glob(os.path.join(REPO, "input_plots", "phone", field, "2025*")))
        per_session = {}                          # plot -> {pair: dist_cm}
        for sp in sessions:
            plot = os.path.basename(sp)
            ours = our_positions_cm(sp)
            if ours and len(ours) >= 3:
                per_session[plot] = pair_dists(ours)
        if len(per_session) < 2:
            print(f"\n=== {field}: <2 sessions, skipping ==="); continue

        # cross-session std of each pairwise distance (flip-invariant, no alignment)
        all_pairs = set().union(*[set(d) for d in per_session.values()])
        pair_std = {}
        for pr in all_pairs:
            vals = [d[pr] for d in per_session.values() if pr in d]
            if len(vals) >= 2:
                pair_std[pr] = (float(np.std(vals)), float(np.median(vals)), len(vals))

        print(f"\n=== {field}: cross-session marker repeatability via pairwise distances (cm) ===")
        stds = [v[0] for v in pair_std.values()]
        print(f"  OVERALL repeatability = median over the {len(pair_std)} distances of their "
              f"cross-session std = {np.median(stds):.2f} cm  (best {min(stds):.2f}, worst {max(stds):.2f})")

        # per-session deviation: median |dist - cross-session median| over that session's pairs
        med_geom = {pr: v[1] for pr, v in pair_std.items()}
        print("  per-session deviation from the cross-session median geometry (high = that session is off):")
        devs = []
        for plot, d in per_session.items():
            e = [abs(d[pr] - med_geom[pr]) for pr in d if pr in med_geom]
            devs.append((plot, float(np.median(e)) if e else None))
        for plot, dv in sorted(devs, key=lambda x: -(x[1] or 0)):
            print(f"    {plot:<26} {dv:5.2f} cm" if dv is not None else f"    {plot:<26} -")


if __name__ == "__main__":
    main()
