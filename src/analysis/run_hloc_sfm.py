"""Run an hloc-based SfM on a phone session and write a COLMAP model, so we can A/B a
detector-free / better-matching front-end against our default ALIKED+LightGlue `sparse/0`.

WHY: on repetitive wheat our ALIKED matcher emits false matches -> pose drift. GLOMAP (a *mapper*) reused those bad matches
and was worse. This script swaps the *matcher* (stage 1) instead: SuperPoint+LightGlue (still
detector-based, a sanity swap) or LoFTR (detector-free / semi-dense, the real test), then feeds the
better matches into COLMAP's own incremental mapper (via hloc/pycolmap) -> `sparse_<matcher>/0`.

The output model is forced to SINGLE-camera SIMPLE_PINHOLE (same as our baseline sparse/0) so that
a downstream marker-geometry comparison can read f/cx/cy from params[0:3] and compare fairly.

Runs in the isolated hloc venv (NOT the wheat3dgs CUDA env):
  source <your-hloc-venv>/bin/activate
  python src/analysis/run_hloc_sfm.py --field field_A --plot 20250715 --matcher splg
  python src/analysis/run_hloc_sfm.py --field field_A --plot 20250715 --matcher loftr

Then compare the resulting sparse_<matcher>/0 against the baseline sparse/0 with your own
marker-geometry / pose check.
"""

import os
import shutil
import argparse
from pathlib import Path

import pycolmap as pc
from hloc import (extract_features, match_features, match_dense,
                  pairs_from_exhaustive, reconstruction)

REPO = Path(__file__).resolve().parents[2]

# detector-based sparse front-ends: (extractor conf, matcher conf)
SPARSE = {
    "splg":   ("superpoint_max", "superpoint+lightglue"),
    "aliked": ("aliked-n16",     "aliked+lightglue"),   # same family as our COLMAP baseline
}
# detector-free / semi-dense front-ends (hloc match_dense)
DENSE = {"loftr": "loftr"}


def run(field, plot, matcher, image_subdir, out_name, keep_work, resize_max=None, max_kpts=None):
    """Extract+match with the chosen front-end, then incremental-map to a SIMPLE_PINHOLE model."""
    sess = REPO / "input_plots" / "phone" / field / plot
    images = sess / image_subdir
    assert images.is_dir(), f"missing images dir {images}"
    img_list = sorted(p.name for p in images.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"[{field}/{plot}] {matcher}: {len(img_list)} images from {image_subdir} "
          f"(resize_max={resize_max}, max_kpts={max_kpts})")

    # work dir keyed by matcher+resolution+kpts so different-resolution runs don't reuse stale h5
    tag = matcher + (f"_r{resize_max}" if resize_max else "") + (f"_k{max_kpts}" if max_kpts else "")
    work = sess / "hloc_work" / tag
    work.mkdir(parents=True, exist_ok=True)
    pairs = work / "pairs-exhaustive.txt"
    pairs_from_exhaustive.main(pairs, image_list=img_list)   # all-pairs = free loop closure

    # 1. matching (stage 1) -> feature + match h5 files
    if matcher in SPARSE:
        fconf = dict(extract_features.confs[SPARSE[matcher][0]])
        fconf["preprocessing"] = dict(fconf.get("preprocessing", {}))
        if resize_max:                      # detector-based: CAN run near-native (sparse keypoints)
            fconf["preprocessing"]["resize_max"] = resize_max
        if max_kpts:                        # more keypoints to exploit the extra resolution
            # the keypoint-cap key differs per extractor: SuperPoint uses max_keypoints,
            # ALIKED uses max_num_keypoints
            kp_key = "max_num_keypoints" if matcher == "aliked" else "max_keypoints"
            fconf["model"] = dict(fconf["model"], **{kp_key: max_kpts})
        mconf = match_features.confs[SPARSE[matcher][1]]
        feats = extract_features.main(fconf, images, work, image_list=img_list)
        matches = match_features.main(mconf, pairs, fconf["output"], work)
    elif matcher in DENSE:
        conf = dict(match_dense.confs[DENSE[matcher]])
        if resize_max:                      # LoFTR default is 1024 (4x downscale from 4032) -> too coarse
            conf["preprocessing"] = dict(conf["preprocessing"], resize_max=resize_max)
        feats = work / f"{matcher}_feats.h5"
        matches = work / f"{matcher}_matches.h5"
        # match_dense writes pseudo-keypoints (feats) + matches for the pairs
        match_dense.main(conf, pairs, images, export_dir=work,
                         features=feats, matches=matches)
    else:
        raise SystemExit(f"unknown matcher {matcher}")

    # 2. incremental mapper (stage 2) -> COLMAP model, forced single SIMPLE_PINHOLE (== baseline)
    sfm_dir = sess / out_name / "0"
    if sfm_dir.exists():
        shutil.rmtree(sfm_dir)
    sfm_dir.mkdir(parents=True, exist_ok=True)
    model = reconstruction.main(
        sfm_dir, images, pairs, feats, matches,
        camera_mode=pc.CameraMode.SINGLE,
        image_options={"camera_model": "SIMPLE_PINHOLE"},
        image_list=img_list,
        # match our baseline run_colmap.py mapper (tight BA tol) so the A/B isolates the MATCHER,
        # not the mapper convergence
        mapper_options={"ba_global_function_tolerance": 1e-6},
    )
    print(f"  -> {out_name}/0 : {model.num_reg_images()}/{len(img_list)} registered, "
          f"{model.num_points3D()} points")
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)   # h5 feats/matches are large intermediates
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True, help="e.g. field_A")
    ap.add_argument("--plot", required=True, help="session, e.g. 20250715")
    ap.add_argument("--matcher", default="splg",
                    choices=list(SPARSE) + list(DENSE))
    ap.add_argument("--image_subdir", default="input_uniform",
                    help="same images our baseline sparse/0 used")
    ap.add_argument("--out", default=None, help="output model dir name (default sparse_<matcher>)")
    ap.add_argument("--keep_work", action="store_true",
                    help="keep the hloc_work/ h5 features+matches (large) for reuse")
    ap.add_argument("--resize_max", type=int, default=None,
                    help="override matching resolution; detector-based can go near-native 4032, "
                         "dense (LoFTR) is memory-capped ~1024-1600")
    ap.add_argument("--max_kpts", type=int, default=None,
                    help="detector-based only: keypoint budget (default 4096); raise with resolution")
    args = ap.parse_args()
    suffix = (f"{args.matcher}" + (f"_r{args.resize_max}" if args.resize_max else "")
              + (f"_k{args.max_kpts}" if args.max_kpts else ""))
    out_name = args.out or f"sparse_{suffix}"
    run(args.field, args.plot, args.matcher, args.image_subdir, out_name, args.keep_work,
        resize_max=args.resize_max, max_kpts=args.max_kpts)


if __name__ == "__main__":
    main()
