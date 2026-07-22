"""Find where the RANSAC ruler's majority assumption gives out.

The other tests injected RANDOM / MINORITY outliers, which RANSAC rejects because they don't agree with
each other. This tests the case it CANNOT handle: a CONSISTENT wrong subset -- a fraction of a marker's
detections that all point at the SAME wrong physical location (the real failure being a marker mis-decoded
as a *different* plate across several frames, so those frames coherently triangulate to that other plate).

We replace a rising fraction of each marker's detections with the projection of a fixed DECOY point
(true marker shifted by a known vector). RANSAC keeps the largest self-consistent subset -- so below 50%
the true detections win and it recovers truth; above 50% the decoy subset is larger and RANSAC locks onto
the WRONG location. This maps the exact flip point and confirms the documented limitation.

Run:  python src/analysis/test_majority_misdecode.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rescore_models_geometry import robust_triangulate  # noqa: E402
from test_synthetic_marker_metric import build_truth, make_detections, project, RNG  # noqa: E402

SHIFT = np.array([200.0, 150.0, 0.0])     # decoy = true marker + this (|shift| = 250 cm)


def main():
    markers, poses, K, wh = build_truth()
    clean = make_detections(markers, poses, K, wh)
    shift_norm = float(np.linalg.norm(SHIFT))
    print(f"Consistent wrong subset points at a DECOY = true marker + {SHIFT.tolist()} (|shift|={shift_norm:.0f} cm).")
    print("err_vs_true ~0 => recovered the real marker; ~250 => RANSAC locked onto the wrong (decoy) subset.\n")
    print(f"   {'wrong frac':>10} {'err_vs_TRUE cm':>15} {'err_vs_DECOY cm':>16}")

    for frac in [0.0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]:
        et, ed = [], []
        for _ in range(5):
            dets = {code: list(v) for code, v in clean.items()}
            for code, obs in dets.items():
                decoy = markers[code] + SHIFT
                k = int(round(frac * len(obs)))
                idx = RNG.choice(len(obs), size=k, replace=False) if k else []
                for ii in idx:
                    name = obs[ii][0]
                    R, t = poses[name]
                    uv = project(decoy, R, t, K)        # consistent: all wrong frames see the SAME decoy
                    if uv is not None:
                        obs[ii] = (name, uv)
            for code in dets:
                X = robust_triangulate(dets[code], poses, K)
                if X is not None:
                    et.append(np.linalg.norm(X - markers[code]))
                    ed.append(np.linalg.norm(X - (markers[code] + SHIFT)))
        print(f"   {frac:>10.2f} {np.median(et):>15.3f} {np.median(ed):>16.3f}")

    print("\nReading: RANSAC recovers the TRUE marker while the correct detections are the MAJORITY (frac<0.5),")
    print("then FLIPS to the decoy once the wrong subset becomes the majority (frac>0.5). This is the")
    print("documented limitation: robust_triangulate defends against random/minority outliers, NOT against a")
    print("majority-consistent mis-decode -- that must be caught upstream (decode/ID consensus), not here.")


if __name__ == "__main__":
    main()
