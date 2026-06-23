"""Flavour 2 of Level B: GCP-constrained bundle adjustment with the markers (via pycolmap).

The COLMAP CLI does not expose ground-control-point BA, and pycolmap 4.0.4 has no dedicated GCP class
either — but its general BundleAdjuster + BundleAdjustmentConfig let us build it: we add the 6 markers
as 3D points AT THE SURVEYED positions, attach their real 2D image observations, mark those points
CONSTANT (= the GCP constraint), and run BA so the camera poses (and scene points) re-optimise to honour
the survey. Unlike Flavour 1 (a 7-DOF similarity applied post-hoc), here the markers are baked into the
optimiser — the real §11b goal.

Start point = the Flavour-1 metric model (`sparse_metric/`), already ~18 mm from the survey, so BA only
has to polish. Markers/survey are in the SAME local metric frame (origin from metric_frame.json).

What we measure (the honest comparison vs Flavour 1's ~18 mm):
  * marker reprojection error BEFORE (survey points through the ORIGINAL poses) vs AFTER (refined poses)
    — does the imagery SUPPORT pinning the markers to the survey, or fight it?
  * scene reprojection error before/after — did anchoring hurt the rest of the reconstruction?
  * camera-centre shift — how far BA had to move the cameras to satisfy the markers.
Markers are constant, so their 3D "fit" is 0 by construction; the meaningful signal is whether their
2D reprojection drops to scene level (~1-2 px) = metric anchoring consistent with the images.

Output: a refined `sparse_metric_gcp/` model + `logs/metric_gcp_ba.json`. Intrinsics fixed by default
(scale is pinned by the GCPs; refining focal would just add a depth/scale wobble) — toggle via config.

Usage:
    python src/preprocessing/marker_gcp_ba.py field=field_A plot=20250609
"""

import json
import os
import sys

import hydra
import numpy as np
import pycolmap as pc
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(__file__))
import marker_scale  # noqa: E402  (load_survey / field_letter)


def marker_reproj_px(rec, marker_pids, survey_local, obs_by_code):
    """Mean/median reprojection error (px) of the fixed survey points through the current poses."""
    errs = []
    for code, pid in marker_pids.items():
        X = survey_local[code]
        for ob in obs_by_code[code]:
            img = rec.find_image_with_name(ob["cam"])
            if img is None:
                continue
            uv = img.project_point(X)
            if uv is None:
                continue
            errs.append(float(np.linalg.norm(uv - np.array(ob["xy"]))))
    errs = np.array(errs)
    return errs


def camera_centers(rec):
    """{image_name: projection centre} for the registered images."""
    return {rec.image(i).name: rec.image(i).projection_center() for i in rec.reg_image_ids()}


@hydra.main(version_base=None, config_path="../../configs/preprocessing", config_name="marker_gcp_ba")
def main(cfg: DictConfig):
    """Run marker GCP-constrained bundle adjustment on the metric model."""
    src = cfg.source_path
    letter = marker_scale.field_letter(cfg.field)
    in_dir = os.path.join(src, cfg.metric_model)
    out_dir = os.path.join(src, cfg.output_dir)
    print(OmegaConf.to_yaml(cfg))

    # --- survey in the SAME local metric frame as the Flavour-1 model ---
    meta = json.load(open(os.path.join(src, cfg.metric_frame_json)))
    origin = np.array(meta["survey_origin_ch1903_lv95"], dtype=np.float64)
    survey_abs = marker_scale.load_survey(cfg.survey_file.replace("<L>", letter))
    survey_local = {c: survey_abs[c] - origin for c in survey_abs}

    # --- marker 2D observations (real only: detected/snapped inliers, NOT reprojected) ---
    tri = json.load(open(os.path.join(src, cfg.triangulation_json)))
    obs_by_code = {}
    for code_str, obs_list in tri.items():
        code = int(code_str)
        if code not in survey_local:
            continue
        kept = [o for o in obs_list
                if o.get("src") != "reprojected" and (o.get("inlier", True) or cfg.use_outliers)]
        if len(kept) >= cfg.min_views:
            obs_by_code[code] = kept

    rec = pc.Reconstruction(in_dir)
    print(f"loaded metric model: {rec.num_images()} imgs, {rec.num_points3D()} pts; "
          f"markers with >= {cfg.min_views} obs: {sorted(obs_by_code)}")

    scene_reproj_before = rec.compute_mean_reprojection_error()
    centers_before = camera_centers(rec)

    # --- add the 6 markers as 3D points at the survey positions, with their observations ---
    marker_pids = {}
    for code, obs in obs_by_code.items():
        elems = []
        for ob in obs:
            img = rec.find_image_with_name(ob["cam"])
            if img is None:
                continue
            img.points2D.append(pc.Point2D(np.array(ob["xy"], dtype=np.float64)))
            idx = img.num_points2D() - 1
            elems.append(pc.TrackElement(img.image_id, idx))
        pid = rec.add_point3D(survey_local[code], pc.Track(elems), np.array([255, 0, 0], np.uint8))
        for el in elems:                                  # link the new 2D obs to the marker point
            rec.image(el.image_id).set_point3D_for_point2D(el.point2D_idx, pid)
        marker_pids[code] = pid
    print(f"added {len(marker_pids)} marker GCP points: {marker_pids}")

    mk_before = marker_reproj_px(rec, marker_pids, survey_local, obs_by_code)

    # --- BA config: all images; markers CONSTANT (GCP), every other point variable ---
    ba_cfg = pc.BundleAdjustmentConfig()
    for img_id in rec.reg_image_ids():
        ba_cfg.add_image(img_id)
    marker_pid_set = set(marker_pids.values())
    for pid in rec.point3D_ids():
        if pid in marker_pid_set:
            ba_cfg.add_constant_point(pid)
        else:
            ba_cfg.add_variable_point(pid)

    opts = pc.BundleAdjustmentOptions()
    opts.refine_focal_length = cfg.refine_focal_length
    opts.refine_principal_point = cfg.refine_principal_point
    opts.refine_extra_params = cfg.refine_extra_params
    opts.refine_points3D = True
    opts.refine_rig_from_world = True            # the camera poses (what we want the GCPs to correct)
    opts.print_summary = True

    ba = pc.create_default_bundle_adjuster(opts, ba_cfg, rec)
    summary = ba.solve()
    print("BA termination:", summary.termination_type if hasattr(summary, "termination_type") else summary)

    scene_reproj_after = rec.compute_mean_reprojection_error()
    mk_after = marker_reproj_px(rec, marker_pids, survey_local, obs_by_code)
    centers_after = camera_centers(rec)
    shifts_mm = np.array([np.linalg.norm(centers_after[n] - centers_before[n]) * 1000
                          for n in centers_before if n in centers_after])

    if os.path.isdir(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    rec.write(out_dir)

    result = {
        "field": cfg.field, "plot": cfg.plot,
        "markers": sorted(obs_by_code), "n_marker_obs": int(sum(len(v) for v in obs_by_code.values())),
        "refine_intrinsics": bool(cfg.refine_focal_length or cfg.refine_principal_point),
        "scene_reproj_px_before": round(scene_reproj_before, 4),
        "scene_reproj_px_after": round(scene_reproj_after, 4),
        "marker_reproj_px_before": {"mean": round(float(mk_before.mean()), 3),
                                    "median": round(float(np.median(mk_before)), 3),
                                    "max": round(float(mk_before.max()), 3)},
        "marker_reproj_px_after": {"mean": round(float(mk_after.mean()), 3),
                                   "median": round(float(np.median(mk_after)), 3),
                                   "max": round(float(mk_after.max()), 3)},
        "camera_shift_mm": {"mean": round(float(shifts_mm.mean()), 2),
                            "median": round(float(np.median(shifts_mm)), 2),
                            "max": round(float(shifts_mm.max()), 2)},
        "output_model": out_dir,
    }
    out_json = os.path.join(src, cfg.output_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(result, open(out_json, "w"), indent=1)

    print("\n" + "=" * 68)
    print(f"  GCP-CONSTRAINED BA  {cfg.field}/{cfg.plot}   (markers fixed at survey)")
    print("=" * 68)
    print(f"  scene reproj (px)      : {scene_reproj_before:.3f}  ->  {scene_reproj_after:.3f}")
    print(f"  marker reproj (px) mean: {mk_before.mean():.3f}  ->  {mk_after.mean():.3f}   "
          f"(median {np.median(mk_before):.2f} -> {np.median(mk_after):.2f})")
    print(f"  camera shift (mm)      : mean {shifts_mm.mean():.2f}  median {np.median(shifts_mm):.2f}  "
          f"max {shifts_mm.max():.2f}")
    print(f"  output model           : {out_dir}")
    print("=" * 68)
    print("READ: marker reproj dropping toward scene reproj = imagery SUPPORTS the metric anchoring;")
    print("      staying high = survey & images disagree (the ~18mm was real, BA can't absorb it).")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
