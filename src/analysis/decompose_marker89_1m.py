"""Decompose the fD/0706 marker-89 '~1 m' precisely: how much was OLD-triangulation-method artifact
(bad detections averaged in) vs REAL pose-driven error? Hold the other 5 markers fixed (robust), then
triangulate marker 89 through the SAME baseline poses two ways — OLD one-pass DLT (all detections) vs
NEW RANSAC — and report marker 89's distance error to the other 5 vs survey. Then do NEW on hloc poses.
"""

import os
import sys
from itertools import combinations

import numpy as np
import pycolmap as pc

sys.path.insert(0, os.path.dirname(__file__))
from rescore_models_geometry import load_survey_cm, load_dets, triangulate, robust_triangulate, _reproj, REPO  # noqa: E402

FIELD, PLOT = "field_D", "20250706"
OTHERS = [77, 85, 101, 105, 113]   # the 5 non-89 markers, held fixed
M89 = 89


def load_model(model_dir):
    rec = pc.Reconstruction(model_dir)
    poses = {im.name: (im.cam_from_world().rotation.matrix(), np.array(im.cam_from_world().translation))
             for im in rec.images.values()}
    cam = list(rec.cameras.values())[0]
    K = np.array([[cam.params[0], 0, cam.params[1]], [0, cam.params[0], cam.params[2]], [0, 0, 1.0]])
    return poses, K


def m89_error(X89, others_pos, survey, scale):
    """median/max abs error (cm) of marker-89's distances to the 5 fixed others vs survey."""
    errs = []
    for o in OTHERS:
        if o in others_pos:
            d = np.linalg.norm(X89 - others_pos[o]) * scale
            errs.append(abs(d - np.linalg.norm(survey[M89] - survey[o])))
    return (float(np.median(errs)), float(max(errs))) if errs else (None, None)


def main():
    sess = os.path.join(REPO, "input_plots", "phone", FIELD, PLOT)
    survey = load_survey_cm(FIELD)
    dets = load_dets(sess)

    for tag, model in [("BASELINE (COLMAP-ALIKED)", "sparse/0"), ("hloc-ALIKED", "sparse_aliked_r2048/0")]:
        poses, K = load_model(os.path.join(sess, model))
        # fix the 5 other markers robustly + a scale from their known survey distances
        others_pos = {}
        for o in OTHERS:
            dd = [(c, xy) for c, xy in dets[o] if c in poses]
            X = robust_triangulate(dd, poses, K) if len(dd) >= 2 else None
            if X is not None:
                others_pos[o] = X
        sd = {frozenset((a, b)): np.linalg.norm(survey[a] - survey[b]) for a, b in combinations(OTHERS, 2)}
        od = {frozenset((a, b)): np.linalg.norm(others_pos[a] - others_pos[b])
              for a, b in combinations(others_pos, 2)}
        scale = float(np.median([sd[k] / od[k] for k in (set(sd) & set(od))]))

        dd89 = [(c, xy) for c, xy in dets[M89] if c in poses]
        X_old = triangulate(dd89, poses, K)               # OLD: one-pass DLT over ALL detections
        X_new = robust_triangulate(dd89, poses, K)         # NEW: RANSAC
        inl = sum(1 for c, xy in dd89 if _reproj(c, xy, X_new, poses, K) < 8.0) if X_new is not None else 0

        print(f"\n=== {tag} ({model}) ===")
        print(f"  marker 89 has {len(dd89)} detections through these poses; RANSAC keeps {inl} as inliers")
        if X_old is not None:
            mo, xo = m89_error(X_old, others_pos, survey, scale)
            print(f"  OLD one-pass DLT (all {len(dd89)} dets):  median {mo:8.2f} cm | max {xo:8.2f} cm")
        if X_new is not None:
            mn, xn = m89_error(X_new, others_pos, survey, scale)
            print(f"  NEW RANSAC ({inl} inliers):             median {mn:8.2f} cm | max {xn:8.2f} cm")


if __name__ == "__main__":
    main()
