"""Arm A experiment: does integrating markers INTO COLMAP (with intrinsics refined) beat the post-hoc
approach? Evaluated with leave-one-marker-out (LOMO) so we never train and test on the same point.

Plan + rationale: docs/preprocessing/markers/MARKER_COLMAP_RERUN_EXPERIMENT.md. This is a DIAGNOSTIC script — it does not
touch the dataset or the production sparse_metric/ models; it only reads them and writes a JSON report.

The honest test (LOMO): for each of the 6 markers, hold it OUT, anchor the other 5 as constant GCPs at
their surveyed positions, run bundle adjustment, then PREDICT the held-out marker's 3D position by
triangulating its own 2D observations through the (now refined) cameras, and measure the error in mm vs
its surveyed value. The held-out marker never constrained the solve, so its error is an unbiased measure
of whether the camera model actually got better. We compare four methods per fold:

  * baseline_noBA  — no anchoring at all; triangulate the held-out marker through the ORIGINAL metric
                     model cameras. This is the post-hoc / Flavour-1 prediction (model unchanged).
  * frozen         — anchor 5, refine poses + scene points, intrinsics FROZEN (= current Flavour 2).
  * focal          — anchor 5, also refine the focal length.
  * focal_pp       — anchor 5, also refine focal length + principal point.

If 'focal'/'focal_pp' give LOWER held-out error than baseline/frozen (and scene reproj doesn't get
worse), markers genuinely improve calibration → in-BA integration is worth it. If flat/worse, it
confirms 6 coplanar markers can't improve internal geometry and Flavour 1 stays sufficient.

Generic over sessions: pass any field=/plot= that has sparse_metric/ + logs/metric_frame.json +
logs/marker_triangulation.json + a survey file. Add more dates later and just re-run.

Usage:
    python src/preprocessing/marker_gcp_lomo.py field=field_A plot=20250609
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

# the four methods we compare: (name, refine_focal, refine_principal_point). baseline_noBA is handled
# specially (no anchoring / no BA at all).
METHODS = [
    ("baseline_noBA", None, None),
    ("frozen",        False, False),
    ("focal",         True,  False),
    ("focal_pp",      True,  True),
]


def held_out_rays(rec, obs):
    """Build (camera-centre C, unit world ray d) for each 2D observation of one marker, using the
    current (possibly refined) camera poses + intrinsics. Used to triangulate the held-out marker."""
    rays = []
    for ob in obs:
        img = rec.find_image_with_name(ob["cam"])
        if img is None:
            continue
        cam = rec.camera(img.camera_id)
        x, y = cam.cam_from_img(np.asarray(ob["xy"], dtype=np.float64))  # normalised (undistorted) coords
        d_cam = np.array([x, y, 1.0])
        R_wc = img.cam_from_world().rotation.matrix()                    # world->cam rotation
        d_world = R_wc.T @ d_cam                                         # cam->world direction
        d_world /= np.linalg.norm(d_world)
        rays.append((np.asarray(img.projection_center(), dtype=np.float64), d_world))
    return rays


def triangulate_lines(rays):
    """Least-squares closest point to a bundle of 3D lines (C_i, d_i): min_X sum ||(I - d d^T)(X-C)||^2.
    Returns None if fewer than 2 rays."""
    if len(rays) < 2:
        return None
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for C, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ C
    return np.linalg.solve(A, b)


def intrinsics(rec):
    """(focal, cx, cy) of the (single) camera — phone runs with single_camera so all images share it."""
    cam = rec.cameras[list(rec.cameras.keys())[0]]
    return cam.focal_length_x, cam.principal_point_x, cam.principal_point_y


def run_anchor_ba(in_dir, survey_local, obs_by_code, anchors, refine_focal, refine_pp):
    """Fresh metric model → add the `anchors` markers as CONSTANT GCPs (survey pos + their 2D obs) →
    bundle-adjust (poses + scene points always; intrinsics per the flags). Returns the refined
    reconstruction plus before/after scene reproj and intrinsics."""
    rec = pc.Reconstruction(in_dir)
    scene_before = rec.compute_mean_reprojection_error()
    foc_b, cx_b, cy_b = intrinsics(rec)

    marker_pids = []
    for code in anchors:
        elems = []
        for ob in obs_by_code[code]:
            img = rec.find_image_with_name(ob["cam"])
            if img is None:
                continue
            img.points2D.append(pc.Point2D(np.asarray(ob["xy"], dtype=np.float64)))
            idx = img.num_points2D() - 1
            elems.append(pc.TrackElement(img.image_id, idx))
        pid = rec.add_point3D(survey_local[code], pc.Track(elems), np.array([255, 0, 0], np.uint8))
        for el in elems:
            rec.image(el.image_id).set_point3D_for_point2D(el.point2D_idx, pid)
        marker_pids.append(pid)

    ba_cfg = pc.BundleAdjustmentConfig()
    for iid in rec.reg_image_ids():
        ba_cfg.add_image(iid)
    mset = set(marker_pids)
    for pid in rec.point3D_ids():
        (ba_cfg.add_constant_point if pid in mset else ba_cfg.add_variable_point)(pid)

    opts = pc.BundleAdjustmentOptions()
    opts.refine_focal_length = refine_focal
    opts.refine_principal_point = refine_pp
    opts.refine_extra_params = False          # SIMPLE_PINHOLE has none anyway
    opts.refine_points3D = True
    opts.refine_rig_from_world = True         # the camera poses (what the GCPs should correct)
    opts.print_summary = False
    pc.create_default_bundle_adjuster(opts, ba_cfg, rec).solve()

    scene_after = rec.compute_mean_reprojection_error()
    foc_a, cx_a, cy_a = intrinsics(rec)
    return rec, {
        "scene_reproj_before": scene_before, "scene_reproj_after": scene_after,
        "focal_before": foc_b, "focal_after": foc_a,
        "cx_before": cx_b, "cx_after": cx_a, "cy_before": cy_b, "cy_after": cy_a,
    }


@hydra.main(version_base=None, config_path="../../configs/preprocessing", config_name="marker_gcp_lomo")
def main(cfg: DictConfig):
    """Run the leave-one-marker-out experiment over the four methods and write logs/metric_gcp_lomo.json."""
    src = cfg.source_path
    letter = marker_scale.field_letter(cfg.field)
    in_dir = os.path.join(src, cfg.metric_model)
    print(OmegaConf.to_yaml(cfg))

    # survey in the SAME local metric frame as the Flavour-1 model
    meta = json.load(open(os.path.join(src, cfg.metric_frame_json)))
    origin = np.array(meta["survey_origin_ch1903_lv95"], dtype=np.float64)
    survey_abs = marker_scale.load_survey(cfg.survey_file.replace("<L>", letter))
    survey_local = {c: survey_abs[c] - origin for c in survey_abs}

    # real marker observations only (detected/snapped inliers, NOT reprojected), >= min_views
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
    markers = sorted(obs_by_code)
    print(f"usable markers ({len(markers)}): {markers}")
    if len(markers) < 3:
        print("ERROR: need >= 3 usable markers for a LOMO experiment.")
        return

    # results[method] = list of per-fold dicts
    results = {name: [] for name, _, _ in METHODS}
    for held in markers:
        anchors = [m for m in markers if m != held]
        for name, rfoc, rpp in METHODS:
            if name == "baseline_noBA":
                rec = pc.Reconstruction(in_dir)         # original model, no anchoring/BA
                stats = None
            else:
                rec, stats = run_anchor_ba(in_dir, survey_local, obs_by_code, anchors, rfoc, rpp)
            X = triangulate_lines(held_out_rays(rec, obs_by_code[held]))
            if X is None:
                continue
            err_mm = float(np.linalg.norm(X - survey_local[held]) * 1000.0)
            fold = {"held": held, "err_mm": err_mm}
            if stats is not None:
                fold["scene_reproj_after"] = round(stats["scene_reproj_after"], 4)
                fold["dfocal"] = round(stats["focal_after"] - stats["focal_before"], 3)
                fold["dcx"] = round(stats["cx_after"] - stats["cx_before"], 3)
                fold["dcy"] = round(stats["cy_after"] - stats["cy_before"], 3)
            results[name].append(fold)
        print(f"  held {held:>4}: " + "  ".join(
            f"{name}={[f['err_mm'] for f in results[name] if f['held']==held][0]:.1f}mm"
            for name, _, _ in METHODS))

    # per-method summary
    summary = {}
    for name, _, _ in METHODS:
        errs = np.array([f["err_mm"] for f in results[name]])
        s = {"held_out_mm_mean": round(float(errs.mean()), 2),
             "held_out_mm_median": round(float(np.median(errs)), 2),
             "held_out_mm_max": round(float(errs.max()), 2)}
        bafolds = [f for f in results[name] if "scene_reproj_after" in f]
        if bafolds:
            s["scene_reproj_after_mean"] = round(float(np.mean([f["scene_reproj_after"] for f in bafolds])), 4)
            s["dfocal_mean"] = round(float(np.mean([f["dfocal"] for f in bafolds])), 3)
            s["dcx_mean"] = round(float(np.mean([f["dcx"] for f in bafolds])), 3)
            s["dcy_mean"] = round(float(np.mean([f["dcy"] for f in bafolds])), 3)
        summary[name] = s

    out = {"field": cfg.field, "plot": cfg.plot, "markers": markers,
           "n_folds": len(markers), "summary": summary, "folds": results}
    out_path = os.path.join(src, cfg.output_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)

    base = summary["baseline_noBA"]["held_out_mm_mean"]
    print("\n" + "=" * 70)
    print(f"  LOMO EXPERIMENT  {cfg.field}/{cfg.plot}   ({len(markers)} folds)")
    print("=" * 70)
    print(f"  {'method':<16}{'held-out mm':>14}{'scene px':>11}{'Δfocal':>10}{'Δcx/Δcy':>14}")
    for name, _, _ in METHODS:
        s = summary[name]
        scene = f"{s.get('scene_reproj_after_mean', '-')}"
        dfoc = f"{s.get('dfocal_mean', '-')}"
        dpp = (f"{s.get('dcx_mean','-')}/{s.get('dcy_mean','-')}"
               if "dcx_mean" in s else "-")
        delta = s["held_out_mm_mean"] - base
        print(f"  {name:<16}{s['held_out_mm_mean']:>10.2f} mm{scene:>11}{dfoc:>10}{dpp:>14}"
              f"   ({delta:+.2f} vs baseline)")
    print("=" * 70)
    print("READ: lower held-out mm = better camera model. 'focal'/'focal_pp' beating baseline AND not")
    print("      raising scene px = markers genuinely improve calibration. Flat/worse = Flavour 1 stays.")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
