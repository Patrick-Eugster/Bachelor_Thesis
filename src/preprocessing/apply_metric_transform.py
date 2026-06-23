"""Flavour 1 of Level B: rewrite the COLMAP model into a METRIC frame using the marker transform.

Step 3 (marker_scale.py) found the similarity transform (scale + R + t) that maps our COLMAP world
onto the surveyed marker XYZ. This script applies that SAME transform to the whole `sparse/0/` model
and writes a metric `sparse_metric/`. The reconstruction geometry is unchanged (rigid+scale map, no
re-optimisation) — it just gets real metres, which is what phenotyping (length/width/volume in mm) needs.

Why a hand-written rewrite (not `colmap model_transformer`): the tool's `--transform_path` matrix
convention did not match a plain `[sR|t]` 4x4 (it produced a scrambled rotation + 30 m translation,
caught by the built-in check), so we transform the COLMAP TEXT model ourselves — only the numeric
coordinates change; tracks, 2D observations and IDs are copied verbatim. World similarity X' = s R X + t
implies, for a world->cam pose (R_wc, t_wc): R_wc' = R_wc R^T,  t_wc' = s t_wc - R_wc R^T t (so the
camera centre maps like any world point, C' = s R C + t — verified). COLMAP 4.1 stores the pose in BOTH
images.txt and frames.txt (RIG_FROM_WORLD), so both are transformed; cameras/rigs are unchanged.

LOCAL origin (important): the survey is CH1903+/LV95 where X ~ 2.69e6 m. Placing the model there would
wreck 3DGS's float32 position precision (~0.25 m ulp at 2.7e6). So the metric frame uses the survey
CENTROID as origin (coords near 0, mm meaningful); the CH1903 origin is stored in metric_frame.json so
absolute georeferencing is recoverable (just add it back). Text-only output (no stale .bin).

Usage:
    python src/preprocessing/apply_metric_transform.py field=field_A plot=20250609
"""

import json
import os
import shutil
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "gaussians", "scene"))
from colmap_loader import qvec2rotmat, rotmat2qvec  # noqa: E402
import marker_scale  # noqa: E402  (reuse load_survey / load_ours / umeyama — single source of truth)


def _f(v):
    """Format a float with enough precision to be lossless for poses/coords."""
    return f"{v:.15g}"


def transform_pose(qvec, tvec, s, R, t):
    """Map a COLMAP world->cam pose under world similarity X' = sRX + t. Returns (qvec', tvec')."""
    R_wc = qvec2rotmat(np.asarray(qvec, dtype=np.float64))
    t_wc = np.asarray(tvec, dtype=np.float64)
    R_new = R_wc @ R.T
    t_new = s * t_wc - R_wc @ R.T @ t
    return rotmat2qvec(R_new), t_new


def rewrite_images(in_path, out_path, s, R, t):
    """Copy images.txt transforming each pose line; the POINTS2D line is copied unchanged."""
    expecting_pose = True
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith("#") or not line.strip():
                fout.write(line)
                continue
            if expecting_pose:
                p = line.split()
                qvec = list(map(float, p[1:5]))
                tvec = list(map(float, p[5:8]))
                q, tt = transform_pose(qvec, tvec, s, R, t)
                rest = " ".join(p[8:])  # CAMERA_ID + NAME (name may contain spaces — fine, it's the tail)
                fout.write(f"{p[0]} {_f(q[0])} {_f(q[1])} {_f(q[2])} {_f(q[3])} "
                           f"{_f(tt[0])} {_f(tt[1])} {_f(tt[2])} {rest}\n")
                expecting_pose = False
            else:
                fout.write(line)            # 2D observations: pixel coords don't change
                expecting_pose = True


def rewrite_frames(in_path, out_path, s, R, t):
    """Copy frames.txt transforming each RIG_FROM_WORLD pose (cols 2..8); rest copied."""
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith("#") or not line.strip():
                fout.write(line)
                continue
            p = line.split()
            qvec = list(map(float, p[2:6]))
            tvec = list(map(float, p[6:9]))
            q, tt = transform_pose(qvec, tvec, s, R, t)
            rest = " ".join(p[9:])
            fout.write(f"{p[0]} {p[1]} {_f(q[0])} {_f(q[1])} {_f(q[2])} {_f(q[3])} "
                       f"{_f(tt[0])} {_f(tt[1])} {_f(tt[2])} {rest}\n")


def rewrite_points3D(in_path, out_path, s, R, t):
    """Copy points3D.txt transforming each XYZ (cols 1..3); colour/error/track copied unchanged."""
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith("#") or not line.strip():
                fout.write(line)
                continue
            p = line.split()
            X = np.array(list(map(float, p[1:4])))
            Xn = s * (R @ X) + t
            rest = " ".join(p[4:])
            fout.write(f"{p[0]} {_f(Xn[0])} {_f(Xn[1])} {_f(Xn[2])} {rest}\n")


def camera_centers_from_images(path):
    """{name: C} read straight from an images.txt (pose lines only) for verification."""
    out = {}
    expecting_pose = True
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            if expecting_pose:
                p = line.split()
                R = qvec2rotmat(np.array(list(map(float, p[1:5]))))
                tt = np.array(list(map(float, p[5:8])))
                out[" ".join(p[8:]).split(maxsplit=1)[-1]] = -R.T @ tt
                expecting_pose = False
            else:
                expecting_pose = True
    return out


@hydra.main(version_base=None, config_path="../../configs/preprocessing", config_name="marker_metric")
def main(cfg: DictConfig):
    """Apply the marker similarity transform to sparse/0 -> a metric sparse_metric/ text model."""
    src = cfg.source_path
    letter = marker_scale.field_letter(cfg.field)
    in_dir = os.path.join(src, cfg.sparse_dir)
    out_dir = os.path.join(src, cfg.output_dir)
    print(OmegaConf.to_yaml(cfg))

    for fn in ("images.txt", "points3D.txt", "cameras.txt"):
        if not os.path.exists(os.path.join(in_dir, fn)):
            raise SystemExit(f"need a TEXT COLMAP model; missing {fn} in {in_dir} "
                             f"(re-run run_colmap.py with export_text=true)")

    # --- same transform as Step 3, into a LOCAL metric frame (origin = survey centroid) ---
    ours = marker_scale.load_ours(os.path.join(src, cfg.points_json))
    survey = marker_scale.load_survey(cfg.survey_file.replace("<L>", letter))
    codes = sorted(set(ours) & set(survey))
    if len(codes) < 3:
        raise SystemExit(f"need >= 3 shared markers, have {codes}")
    P_ours = np.array([ours[c] for c in codes])
    P_survey_abs = np.array([survey[c] for c in codes])
    origin = P_survey_abs.mean(0)
    P_survey_local = P_survey_abs - origin
    s, R, t = marker_scale.umeyama(P_ours, P_survey_local)

    # --- rewrite the model (text) ---
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    rewrite_images(os.path.join(in_dir, "images.txt"), os.path.join(out_dir, "images.txt"), s, R, t)
    rewrite_points3D(os.path.join(in_dir, "points3D.txt"), os.path.join(out_dir, "points3D.txt"), s, R, t)
    shutil.copy(os.path.join(in_dir, "cameras.txt"), os.path.join(out_dir, "cameras.txt"))
    for opt in ("rigs.txt",):  # unchanged under a world similarity
        if os.path.exists(os.path.join(in_dir, opt)):
            shutil.copy(os.path.join(in_dir, opt), os.path.join(out_dir, opt))
    if os.path.exists(os.path.join(in_dir, "frames.txt")):
        rewrite_frames(os.path.join(in_dir, "frames.txt"), os.path.join(out_dir, "frames.txt"), s, R, t)

    # --- VERIFY 1: written camera centres obey C' = sRC + t (pose formula + I/O correct) ---
    Cin = camera_centers_from_images(os.path.join(in_dir, "images.txt"))
    Cout = camera_centers_from_images(os.path.join(out_dir, "images.txt"))
    pose_err = [np.linalg.norm((s * (R @ Cin[n]) + t) - Cout[n]) for n in Cin if n in Cout]
    pose_max_mm = float(np.max(pose_err) * 1000) if pose_err else float("nan")

    # --- VERIFY 2: our markers land on the (local) survey within the Step-3 residual ---
    fitted = (s * (R @ P_ours.T)).T + t
    resid_mm = np.linalg.norm(fitted - P_survey_local, axis=1) * 1000
    rms_mm = float(np.sqrt((resid_mm ** 2).mean()))

    meta = {
        "field": cfg.field, "plot": cfg.plot, "shared_markers": codes,
        "scale": s, "frame": "local metres (origin = survey centroid)",
        "survey_origin_ch1903_lv95": origin.tolist(),
        "note": "add survey_origin_ch1903_lv95 back to get absolute CH1903+/LV95 coordinates",
        "umeyama_rms_mm": round(rms_mm, 2),
        "pose_writeback_max_mm": round(pose_max_mm, 6),
        "input_model": in_dir, "output_model": out_dir,
    }
    meta_path = os.path.join(src, cfg.output_meta)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    json.dump(meta, open(meta_path, "w"), indent=1)

    print("\n" + "=" * 66)
    print(f"  METRIC MODEL  {cfg.field}/{cfg.plot}")
    print("=" * 66)
    print(f"  scale applied          : {s:.6f}  m / colmap-unit")
    print(f"  output model           : {out_dir}  (text)")
    print(f"  CH1903+ origin (subtr.): {[round(float(o),3) for o in origin]}")
    print(f"  marker fit RMS         : {rms_mm:.2f} mm  (per-marker: "
          + ", ".join(f'{c}:{r:.1f}' for c, r in zip(codes, resid_mm)) + ")")
    print(f"  pose write-back check  : {pose_max_mm:.4g} mm max  "
          f"{'OK' if pose_max_mm < 0.001 else 'CHECK!'}")
    print("=" * 66)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
