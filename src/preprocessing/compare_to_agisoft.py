"""Compare our COLMAP sparse/ to the supervisor's Agisoft agisoft/sparse/ for the same plot.

What it does:
  1. Loads both reconstructions' camera poses (extrinsics).
  2. Finds images present in both (matched by filename).
  3. Umeyama-aligns our camera centers onto Agisoft's (recovers s, R, t).
  4. Reports per-camera translation error (in meters, since Agisoft is metric)
     and optional rotation error (in degrees).

Output: prints summary table to stdout, writes per-camera JSON to logs/.

Run with:  python src/preprocessing/compare_to_agisoft.py field=field_D plot=20250523
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Reuse the same COLMAP readers 3DGS uses everywhere else, but load the file directly
# instead of going through gaussians.scene.__init__ — that pulls in torch (and 3DGS deps)
# which this script genuinely doesn't need.
_loader_path = Path(__file__).resolve().parents[1] / "gaussians" / "scene" / "colmap_loader.py"
_spec = importlib.util.spec_from_file_location("_colmap_loader", _loader_path)
_colmap_loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_colmap_loader)
qvec2rotmat = _colmap_loader.qvec2rotmat
read_extrinsics_binary = _colmap_loader.read_extrinsics_binary
read_extrinsics_text = _colmap_loader.read_extrinsics_text
# NOTE: we don't reuse colmap_loader's read_intrinsics_text — it asserts model == "PINHOLE"
# (a downstream 3DGS constraint), which would reject Agisoft's SIMPLE_PINHOLE exports.
# We only need (model_name, width, height) for the summary, not full intrinsic parsing.


def load_extrinsics(sparse_dir: str) -> dict:
    """Load images.bin or images.txt from a COLMAP sparse dir.
    Returns the dict that colmap_loader produces (id → Image namedtuple)."""
    bin_path = os.path.join(sparse_dir, "images.bin")
    txt_path = os.path.join(sparse_dir, "images.txt")
    if os.path.exists(bin_path):
        return read_extrinsics_binary(bin_path)
    if os.path.exists(txt_path):
        return read_extrinsics_text(txt_path)
    raise FileNotFoundError(f"No images.bin or images.txt in {sparse_dir}")


def load_intrinsics_summary(sparse_dir: str) -> list:
    """Minimal cameras.bin/txt reader — only returns (id, model_name, width, height) per camera
    for the summary printout. Doesn't parse the focal/distortion params because we don't need them
    here, and the 3DGS-style loader asserts PINHOLE-only which would reject Agisoft's SIMPLE_PINHOLE."""
    import struct
    txt_path = os.path.join(sparse_dir, "cameras.txt")
    bin_path = os.path.join(sparse_dir, "cameras.bin")
    out = []
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
                out.append((int(parts[0]), parts[1], int(parts[2]), int(parts[3])))
        return out
    if os.path.exists(bin_path):
        # binary format: num_cameras (uint64) then for each: camera_id (uint32),
        # model_id (int32), width (uint64), height (uint64), num_params doubles
        model_id_to_name = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL",
                            4: "OPENCV", 5: "OPENCV_FISHEYE", 6: "FULL_OPENCV"}
        model_num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12}
        with open(bin_path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
                np_ = model_num_params.get(model_id, 0)
                f.read(8 * np_)  # skip params
                out.append((cam_id, model_id_to_name.get(model_id, f"id{model_id}"), w, h))
        return out
    raise FileNotFoundError(f"No cameras.bin or cameras.txt in {sparse_dir}")


def extract_poses(extrinsics: dict) -> tuple:
    """For each image, compute the camera center (in world coords) and the rotation matrix.
    COLMAP convention: X_cam = R * X_world + t, so camera center C = -R.T @ t.
    Returns: (centers, rotations) — both dicts keyed by image filename."""
    centers = {}
    rotations = {}
    for img in extrinsics.values():
        R = qvec2rotmat(img.qvec)         # world -> camera
        C = -R.T @ img.tvec               # camera center in world coords
        centers[img.name] = C
        rotations[img.name] = R
    return centers, rotations


def umeyama(X: np.ndarray, Y: np.ndarray) -> tuple:
    """Closed-form similarity transform (Umeyama 1991).
    Find s, R, t such that  s * R @ x + t  ≈  y  for paired points x ↔ y.
    Returns: scale (float), rotation (3×3), translation (3,)."""
    n = X.shape[0]
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)
    Xc = X - mu_X
    Yc = Y - mu_Y
    # cross-covariance, then SVD
    H = Xc.T @ Yc / n
    U, D, Vt = np.linalg.svd(H)
    # ensure right-handed rotation (no reflection) via the determinant trick
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    S = np.diag([1.0, 1.0, d])
    R = Vt.T @ S @ U.T
    var_X = (Xc ** 2).sum() / n
    s = np.trace(np.diag(D) @ S) / var_X
    t = mu_Y - s * R @ mu_X
    return float(s), R, t


def rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angle in degrees between two 3D rotations (R1 vs R2).
    Uses arccos((trace(R1.T @ R2) - 1) / 2). Clipped to handle FP rounding."""
    cos_theta = (np.trace(R1.T @ R2) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def summarize(arr: np.ndarray, unit: str, fmt: str = ".4f") -> dict:
    """Return mean/median/min/max/std of an array as a dict — used for both translation and rotation summaries."""
    return {
        f"mean_{unit}": float(format(arr.mean(), fmt)),
        f"median_{unit}": float(format(np.median(arr), fmt)),
        f"min_{unit}": float(format(arr.min(), fmt)),
        f"max_{unit}": float(format(arr.max(), fmt)),
        f"std_{unit}": float(format(arr.std(), fmt)),
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing/compare_to_agisoft")
def main(cfg: DictConfig):
    print("--- compare_to_agisoft config ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------------")

    ours_dir = os.path.join(cfg.source_path, cfg.ours_sparse_dir)
    ref_dir = os.path.join(cfg.source_path, cfg.ref_sparse_dir)

    if not os.path.isdir(ours_dir):
        print(f"ERROR: ours sparse dir not found: {ours_dir}")
        print("Did you run src/preprocessing/convert.py yet?")
        sys.exit(1)
    if not os.path.isdir(ref_dir):
        print(f"ERROR: reference (agisoft) sparse dir not found: {ref_dir}")
        sys.exit(1)

    # 1. Load both reconstructions
    print(f"Loading our COLMAP from: {ours_dir}")
    ext_ours = load_extrinsics(ours_dir)
    int_ours = load_intrinsics_summary(ours_dir)
    centers_ours, rots_ours = extract_poses(ext_ours)

    print(f"Loading Agisoft reference from: {ref_dir}")
    ext_ref = load_extrinsics(ref_dir)
    int_ref = load_intrinsics_summary(ref_dir)
    centers_ref, rots_ref = extract_poses(ext_ref)

    # 2. Match by filename — only cameras present in both can be compared
    common = sorted(set(centers_ours.keys()) & set(centers_ref.keys()))
    only_ours = sorted(set(centers_ours.keys()) - set(centers_ref.keys()))
    only_ref = sorted(set(centers_ref.keys()) - set(centers_ours.keys()))
    print(f"\nCamera counts:")
    print(f"  ours:    {len(centers_ours)}")
    print(f"  agisoft: {len(centers_ref)}")
    print(f"  common:  {len(common)}")
    if only_ours:
        print(f"  only in ours: {len(only_ours)}  (first 3: {only_ours[:3]})")
    if only_ref:
        print(f"  only in agisoft: {len(only_ref)}  (first 3: {only_ref[:3]})")

    if len(common) < 3:
        print(f"\nERROR: only {len(common)} common cameras — need ≥3 for Umeyama alignment.")
        sys.exit(1)

    # 3. Umeyama alignment: ours → agisoft
    X = np.array([centers_ours[n] for n in common])   # our centers
    Y = np.array([centers_ref[n] for n in common])    # agisoft centers (real meters)
    s, R_align, t_align = umeyama(X, Y)
    X_aligned = (s * (R_align @ X.T)).T + t_align

    # 4. Per-camera translation error (in meters, because agisoft IS metric)
    trans_err = np.linalg.norm(X_aligned - Y, axis=1)

    # 5. Per-camera rotation error (optional)
    rot_err = None
    if cfg.include_rotation:
        # our world rotated by R_align matches agisoft world → camera rotations transform accordingly
        # If x_cam = R_ours @ x_world_ours and x_world_ref = R_align @ x_world_ours (ignoring scale/translation),
        # then x_cam = R_ours @ R_align.T @ x_world_ref, so our "aligned" world->camera rotation is R_ours @ R_align.T.
        rot_err = np.array([
            rotation_angle_deg(rots_ours[n] @ R_align.T, rots_ref[n])
            for n in common
        ])

    # 6. Report
    print(f"\n=== Umeyama alignment ===")
    print(f"  recovered scale: {s:.6f}  (our reconstruction is {s:.3f}× metric)")
    print(f"  translation:     [{t_align[0]:.3f}, {t_align[1]:.3f}, {t_align[2]:.3f}] m")
    print(f"  rotation det:    {np.linalg.det(R_align):+.4f}  (should be +1)")

    print(f"\n=== Translation error vs Agisoft (after alignment) ===")
    print(f"  mean:    {trans_err.mean()*1000:.2f} mm")
    print(f"  median:  {np.median(trans_err)*1000:.2f} mm")
    print(f"  min:     {trans_err.min()*1000:.2f} mm")
    print(f"  max:     {trans_err.max()*1000:.2f} mm")
    print(f"  std:     {trans_err.std()*1000:.2f} mm")

    if rot_err is not None:
        print(f"\n=== Rotation error vs Agisoft ===")
        print(f"  mean:    {rot_err.mean():.3f}°")
        print(f"  median:  {np.median(rot_err):.3f}°")
        print(f"  min:     {rot_err.min():.3f}°")
        print(f"  max:     {rot_err.max():.3f}°")
        print(f"  std:     {rot_err.std():.3f}°")

    # camera model summary — useful to know what both sides actually used
    def cam_summary(cams_list):
        return [f"{model} ({w}x{h}) id={cid}" for (cid, model, w, h) in cams_list]
    print(f"\n=== Camera models ===")
    print(f"  ours:    {cam_summary(int_ours)}")
    print(f"  agisoft: {cam_summary(int_ref)}")

    # 7. Save full per-camera report as JSON
    out_path = os.path.join(cfg.source_path, cfg.output_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    report = {
        "field": cfg.field,
        "plot": cfg.plot,
        "n_ours": len(centers_ours),
        "n_agisoft": len(centers_ref),
        "n_common": len(common),
        "only_ours": only_ours,
        "only_agisoft": only_ref,
        "alignment": {
            "scale": s,
            "translation": t_align.tolist(),
            "rotation_matrix": R_align.tolist(),
        },
        "translation_error_m": summarize(trans_err, "m"),
        "rotation_error_deg": summarize(rot_err, "deg", ".4f") if rot_err is not None else None,
        "per_camera": [
            {
                "name": n,
                "trans_error_m": float(trans_err[i]),
                "rot_error_deg": float(rot_err[i]) if rot_err is not None else None,
            }
            for i, n in enumerate(common)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
