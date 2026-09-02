"""C3: detection-free marker geometry on a session with verified 2D markers (field_D/20250627).

Triangulates the 6 markers from the HAND-VERIFIED 2D positions (input_plots/.../agisoft/marker_projections.csv,
Pinned==True) through OUR camera poses (sparse/0), then compares the pairwise marker distances to the physical
SURVEY and TAPE (best-fit scale per source, cm). This removes our marker detector from the chain: the only
inputs are the verified marker pixels and our reconstruction's poses, so the residual is pure reconstruction
geometry error, not detection error. Prints the detection-based number (our detector) beside it for contrast.
"""
import argparse
import csv
import os
import sys
from itertools import combinations

import numpy as np
import pycolmap as pc

sys.path.insert(0, os.path.dirname(__file__))
from rescore_models_geometry import (load_survey_cm, load_tape_cm, robust_triangulate,
                                      bestfit_resid, TARGET_TO_CODE)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_verified_2d(field, plot):
    """{code: [(image_stem, (x,y)), ...]} from the hand-verified Pinned==True projections."""
    p = os.path.join(REPO, "input_plots", "phone", field, plot, "agisoft", "marker_projections.csv")
    out = {}
    for r in csv.DictReader(open(p)):
        if str(r.get("Pinned")).strip().lower() != "true":
            continue
        code = TARGET_TO_CODE.get(int(r["Marker"].split()[-1]))
        try:
            out.setdefault(code, []).append((os.path.splitext(r["Camera"])[0], (float(r["X"]), float(r["Y"]))))
        except ValueError:
            pass
    return out


def load_detected_2d(field, plot):
    """{code: [(image_stem, (x,y))]} from OUR detector (marker_triangulation.json, src=='detected')."""
    import json
    tri = json.load(open(os.path.join(REPO, "input_plots", "phone", field, plot,
                                       "logs", "marker_triangulation.json")))
    out = {}
    for code, obs in tri.items():
        for o in obs:
            if o.get("src") == "detected":
                out.setdefault(int(code), []).append((os.path.splitext(o["cam"])[0], tuple(o["xy"])))
    return out


def model_poses_K(model_dir):
    """Return ({image_stem: (R,t)}, K 3x3) from a COLMAP model (shared single camera)."""
    rec = pc.Reconstruction(model_dir)
    poses = {}
    for im in rec.images.values():
        T = im.cam_from_world()
        poses[os.path.splitext(im.name)[0]] = (T.rotation.matrix(), np.array(T.translation))
    cam = list(rec.cameras.values())[0]
    f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
    return poses, np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def marker_dists(points2d, poses, K):
    """Triangulate each marker from its (stem,xy) observations + poses, return {frozenset(a,b): dist}."""
    pos = {}
    for code, obs in points2d.items():
        dd = [(poses[s], xy) for s, xy in obs if s in poses]
        if len(dd) >= 2:
            # robust_triangulate wants [(cam_name,xy)] + poses-by-name; adapt to a local pose list
            cams = {i: p for i, (p, _) in enumerate(dd)}
            dets = [(i, xy) for i, (_, xy) in enumerate(dd)]
            X = robust_triangulate(dets, cams, K)
            if X is not None:
                pos[code] = X
    return ({frozenset((a, b)): float(np.linalg.norm(pos[a] - pos[b])) for a, b in combinations(pos, 2)},
            len(pos))


def report(tag, d_our, survey, tape):
    d_sur = {frozenset((a, b)): float(np.linalg.norm(survey[a] - survey[b]))
             for a, b in combinations(survey, 2)} if len(survey) >= 3 else {}
    rs, ns = bestfit_resid(d_our, d_sur)
    rt, nt = bestfit_resid(d_our, tape)
    fs = f"{rs:.2f} cm (n{ns})" if rs is not None else "-"
    ft = f"{rt:.2f} cm (n{nt})" if rt is not None else "-"
    print(f"  {tag:<26} vs SURVEY {fs:>14}   vs TAPE {ft:>14}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="field_D")
    ap.add_argument("--plot", default="20250627")
    ap.add_argument("--model", default="sparse/0")
    args = ap.parse_args()
    base = os.path.join(REPO, "input_plots", "phone", args.field, args.plot)
    poses, K = model_poses_K(os.path.join(base, args.model))
    survey = load_survey_cm(args.field)
    try:
        tape = load_tape_cm(args.field)
    except Exception:
        tape = {}

    print(f"\n=== C3: marker geometry SHAPE error, {args.field}/{args.plot} (cm, best-fit scale) ===")
    print(f"    poses from {args.model}; distances vs physical survey and tape\n")
    gt2d = load_verified_2d(args.field, args.plot)
    det2d = load_detected_2d(args.field, args.plot)
    d_gt, n_gt = marker_dists(gt2d, poses, K)
    d_det, n_det = marker_dists(det2d, poses, K)
    report(f"VERIFIED GT ({n_gt} mk)", d_gt, survey, tape)
    report(f"our detector ({n_det} mk)", d_det, survey, tape)
    print("\n  (VERIFIED GT row = detection-free: only verified pixels + our poses, no detector in the loop.)")


if __name__ == "__main__":
    main()
