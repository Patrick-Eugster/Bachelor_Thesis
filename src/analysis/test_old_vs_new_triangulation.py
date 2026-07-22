"""Prove the OLD (pre-RANSAC) rescore ruler was wrong, by running the EXACT old function on synthetic
ground truth alongside the new one.

The old model_marker_dists (recovered verbatim from this session's transcript) triangulated each marker
with a single DLT over ALL its detections -- no outlier rejection:

    if len(dd) >= 2:
        X = triangulate(dd, poses, K)     # <-- the old version

The new one swaps that single line for robust_triangulate (RANSAC-lite). Everything else is identical.
This script injects known canopy-style outlier detections and prints the SHAPE error (cm vs known truth)
that EACH ruler would have reported -- i.e. the actual metric number that drove our (wrong) conclusions.

Run:  python src/analysis/test_old_vs_new_triangulation.py
"""

import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rescore_models_geometry import triangulate, robust_triangulate, bestfit_resid  # noqa: E402
from test_synthetic_marker_metric import build_truth, make_detections, true_dists, RNG  # noqa: E402


def old_model_marker_dists(dets, poses, K):
    """EXACT old (pre-RANSAC) logic: single DLT over ALL detections per marker, no outlier rejection."""
    pos = {}
    for code, d in dets.items():
        dd = [(c, xy) for c, xy in d if c in poses]
        if len(dd) >= 2:
            X = triangulate(dd, poses, K)
            if X is not None:
                pos[code] = X
    return {frozenset((a, b)): float(np.linalg.norm(pos[a] - pos[b])) for a, b in combinations(pos, 2)}


def new_model_marker_dists(dets, poses, K):
    """New (RANSAC) logic: identical except robust_triangulate replaces the single DLT."""
    pos = {}
    for code, d in dets.items():
        dd = [(c, xy) for c, xy in d if c in poses]
        if len(dd) >= 2:
            X = robust_triangulate(dd, poses, K)
            if X is not None:
                pos[code] = X
    return {frozenset((a, b)): float(np.linalg.norm(pos[a] - pos[b])) for a, b in combinations(pos, 2)}


def main():
    markers, poses, K, wh = build_truth()
    gt = true_dists(markers)
    clean = make_detections(markers, poses, K, wh)

    print("SHAPE error (cm vs known-true marker distances) that each ruler reports, under outliers")
    print("(these are the ACTUAL metric numbers -- what we ranked methods by)\n")
    print(f"   {'outlier frac':>12} {'OLD ruler (DLT)':>18} {'NEW ruler (RANSAC)':>20}")
    for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        old_vals, new_vals = [], []
        for _ in range(5):
            dets = {code: list(v) for code, v in clean.items()}
            for code, obs in dets.items():
                k = int(round(frac * len(obs)))
                idx = RNG.choice(len(obs), size=k, replace=False) if k else []
                for ii in idx:                        # canopy false positive: random pixel somewhere
                    name = obs[ii][0]
                    obs[ii] = (name, (RNG.uniform(0, wh[0]), RNG.uniform(0, wh[1])))
            ro, _ = bestfit_resid(old_model_marker_dists(dets, poses, K), gt)
            rn, _ = bestfit_resid(new_model_marker_dists(dets, poses, K), gt)
            if ro is not None:
                old_vals.append(ro)
            if rn is not None:
                new_vals.append(rn)
        mo = float(np.median(old_vals)) if old_vals else float("nan")
        mn = float(np.median(new_vals)) if new_vals else float("nan")
        print(f"   {frac:>12.1f} {mo:>18.3f} {mn:>20.3f}")

    print("\nReading: the physical-GT floor (tape-vs-survey) is ~0.66 cm. The OLD ruler crosses that within")
    print("a handful of outlier detections -- so on real sessions (which DO have canopy false positives)")
    print("its reported 'shape error' was mostly triangulation contamination, not real pose/geometry error.")
    print("That is why it disagreed with compare_to_agisoft and flipped the keypoint conclusion.")


if __name__ == "__main__":
    main()
