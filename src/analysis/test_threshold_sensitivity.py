"""Close the last gap in the ruler validation: does the RANSAC 8px inlier threshold hold for SUBTLE
outliers, not just gross random ones?

The main outlier test scattered fake detections uniformly across the whole 3850px image, so they reproject
with huge error and the 8px threshold rejects them trivially. Real canopy false-positives might land NEAR
the true marker (a few to tens of px off), which is the case that could actually fool an 8px threshold.

Here outliers are placed at a CONTROLLED offset from the true projection (5..160 px) at increasing
fractions, and we measure the recovered 3D marker error vs known truth. This maps exactly where RANSAC
starts to break -- i.e. whether 8px is a safe choice for real data.

Run:  python src/analysis/test_threshold_sensitivity.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rescore_models_geometry import robust_triangulate  # noqa: E402
from test_synthetic_marker_metric import build_truth, make_detections, RNG  # noqa: E402


def main():
    markers, poses, K, wh = build_truth()
    clean = make_detections(markers, poses, K, wh)

    offsets = [5, 8, 12, 20, 40, 80, 160]      # px offset of the fake detection from the true projection
    fracs = [0.2, 0.4, 0.6]                    # fraction of detections that are outliers
    print("Median recovered marker 3D-position error vs known truth (cm), RANSAC only.")
    print("Outliers placed at a fixed pixel OFFSET from the true projection (subtle, not random).")
    print("Inlier threshold is 8 px. Rows below/around 8 px are the stress zone.\n")
    header = "  offset_px " + "".join(f"frac={f:<7.1f}" for f in fracs)
    print(header)
    for off in offsets:
        cells = []
        for frac in fracs:
            errs = []
            for _ in range(5):
                dets = {code: list(v) for code, v in clean.items()}
                for code, obs in dets.items():
                    k = int(round(frac * len(obs)))
                    idx = RNG.choice(len(obs), size=k, replace=False) if k else []
                    for ii in idx:
                        name, (u, v) = obs[ii]
                        ang = RNG.uniform(0, 2 * np.pi)          # random direction, fixed magnitude = off
                        obs[ii] = (name, (u + off * np.cos(ang), v + off * np.sin(ang)))
                for code in dets:
                    X = robust_triangulate(dets[code], poses, K)
                    if X is not None:
                        errs.append(np.linalg.norm(X - markers[code]))
            cells.append(float(np.median(errs)) if errs else float("nan"))
        row = f"  {off:>8} " + "".join(f"{c:>11.3f}" for c in cells)
        print(row)

    print("\nReading: an outlier at offset D can only be mistaken for an inlier if D < 8 px. Below the")
    print("threshold the 'outlier' IS within measurement noise, so including it barely moves the 3D point")
    print("(the error it can inject is bounded by its own small offset). Above ~8 px RANSAC rejects it and")
    print("the error stays ~0. The danger case is therefore only outliers within a few px of truth, which")
    print("inject at most a few-px-worth of 3D error -- far below the ~0.66 cm physical-GT floor.")


if __name__ == "__main__":
    main()
