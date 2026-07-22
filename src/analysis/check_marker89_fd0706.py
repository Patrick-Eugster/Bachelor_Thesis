"""Direct marker-89 before/after on field_D/20250706 — the symptom that started the SfM investigation.
The rescore MEDIAN (8 cm) can hide one catastrophic marker, so this breaks the error down PER MARKER:
triangulate each marker (robust) through a model's poses, then report the median/max abs error of that
marker's pairwise distances to the other markers vs SURVEY (global best-fit scale, alignment-free).
Marker code 89 = target 3 (the ~1 m drift case). Compares baseline (COLMAP-ALIKED) vs hloc-ALIKED.

Run:  python src/analysis/check_marker89_fd0706.py
"""

import os
import sys
from itertools import combinations

import numpy as np
import pycolmap as pc

sys.path.insert(0, os.path.dirname(__file__))
from rescore_models_geometry import load_survey_cm, load_dets, robust_triangulate, REPO  # noqa: E402

FIELD, PLOT = "field_D", "20250706"
MODELS = ["sparse/0", "sparse_aliked_r2048/0"]


def positions(model_dir, dets):
    """Robust-triangulate every marker through this model's poses. Returns {code: XYZ}."""
    rec = pc.Reconstruction(model_dir)
    poses = {}
    for im in rec.images.values():
        T = im.cam_from_world()
        poses[im.name] = (T.rotation.matrix(), np.array(T.translation))
    cam = list(rec.cameras.values())[0]
    f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    pos = {}
    for code, d in dets.items():
        dd = [(c, xy) for c, xy in d if c in poses]
        if len(dd) >= 2:
            X = robust_triangulate(dd, poses, K)
            if X is not None:
                pos[code] = X
    return pos


def main():
    sess = os.path.join(REPO, "input_plots", "phone", FIELD, PLOT)
    survey = load_survey_cm(FIELD)
    dets = load_dets(sess)
    sd = {frozenset((a, b)): float(np.linalg.norm(survey[a] - survey[b])) for a, b in combinations(survey, 2)}

    print(f"\n{FIELD}/{PLOT} — per-marker geometry error vs SURVEY (cm). Marker 89 = the ~1 m drift case.\n")
    for m in MODELS:
        pos = positions(os.path.join(sess, m), dets)
        od = {frozenset((a, b)): float(np.linalg.norm(pos[a] - pos[b])) for a, b in combinations(pos, 2)}
        keys = set(od) & set(sd)
        s = float(np.median([sd[k] / od[k] for k in keys])) if keys else 1.0
        print(f"{m}  (best-fit scale={s:.4f}, markers triangulated={len(pos)}):")
        for c in sorted(pos):
            errs = [abs(od[frozenset((c, o))] * s - sd[frozenset((c, o))])
                    for o in pos if o != c and frozenset((c, o)) in sd]
            if errs:
                flag = "   <-- MARKER 89" if c == 89 else ""
                print(f"    marker {c}: median {np.median(errs):7.2f} cm | max {max(errs):8.2f} cm{flag}")
        print()


if __name__ == "__main__":
    main()
