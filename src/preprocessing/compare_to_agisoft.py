"""Compare our COLMAP sparse/ to the supervisor's Agisoft agisoft/sparse/ for the same plot.

What it does:
  1. Loads both reconstructions' camera poses (extrinsics).
  2. Finds images present in both (matched by filename).
  3. Umeyama-aligns our camera centers onto Agisoft's (recovers s, R, t).
  4. Reports per-camera translation error (in meters, since Agisoft is metric)
     and optional rotation error (in degrees).

Three optional extra comparisons (toggles in configs/preprocessing/compare_to_agisoft.yaml):
  - compare_intrinsics: focal length (resolution-normalized f/W) + horizontal FOV, ours vs Agisoft.
  - compare_points:     cloud-to-cloud nearest-neighbour (Chamfer) distance between our sparse
                        points (after the same Umeyama transform) and Agisoft's metric cloud.
  - compare_reproj:     reprojection error in pixels, RECOMPUTED from scratch the same way for
                        both reconstructions (an internal self-consistency number per model, not
                        an ours-vs-Agisoft distance). Agisoft's exported ERROR column is all zeros,
                        so it must be recomputed; ours is cross-checked against its stored column.

Output: prints summary table to stdout, writes per-camera JSON to logs/.

Run with:  python src/preprocessing/compare_to_agisoft.py field=field_D plot=20250523
"""

import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


# Agisoft renames images on ingestion: "IMG_20250523_102855.jpg" -> "IMG_20250523_102855_3.jpg".
# We strip the trailing "_<digits>" right before the extension so names can be matched against ours.
_AGISOFT_SUFFIX_RE = re.compile(r"_(\d+)(\.[A-Za-z]+)$")


def _strip_agisoft_suffix(name: str) -> str:
    """Remove a trailing '_<digits>' before the file extension, e.g. 'IMG_..._3.jpg' -> 'IMG_....jpg'.
    Leaves names without that pattern unchanged. Lets us match Agisoft's renamed files against ours."""
    return _AGISOFT_SUFFIX_RE.sub(r"\2", name)

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


def extract_poses(extrinsics: dict, name_normalizer=None) -> tuple:
    """For each image, compute the camera center (in world coords) and the rotation matrix.
    COLMAP convention: X_cam = R * X_world + t, so camera center C = -R.T @ t.
    Returns: (centers, rotations) — both dicts keyed by image filename. If name_normalizer
    is given, dict keys are passed through it first (used to strip Agisoft's '_N' suffix)."""
    centers = {}
    rotations = {}
    for img in extrinsics.values():
        R = qvec2rotmat(img.qvec)         # world -> camera
        C = -R.T @ img.tvec               # camera center in world coords
        key = name_normalizer(img.name) if name_normalizer else img.name
        centers[key] = C
        rotations[key] = R
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


# ----------------------------------------------------------------------------------------------
# Extra comparisons: intrinsics, point cloud, reprojection error
# ----------------------------------------------------------------------------------------------

_MODEL_ID_TO_NAME = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL",
                     4: "OPENCV", 5: "OPENCV_FISHEYE", 6: "FULL_OPENCV"}
_MODEL_NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12}


def load_intrinsics_full(sparse_dir: str) -> dict:
    """Read cameras.txt/bin returning {camera_id: {"model", "w", "h", "params"}}.
    Unlike load_intrinsics_summary this KEEPS the params list (focal, principal point, distortion)
    — needed both for the intrinsics comparison and to project points for the reprojection error."""
    import struct
    txt_path = os.path.join(sparse_dir, "cameras.txt")
    bin_path = os.path.join(sparse_dir, "cameras.bin")
    out = {}
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = line.split()
                out[int(p[0])] = {"model": p[1], "w": int(p[2]), "h": int(p[3]),
                                  "params": list(map(float, p[4:]))}
        return out
    if os.path.exists(bin_path):
        with open(bin_path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
                npar = _MODEL_NUM_PARAMS.get(model_id, 0)
                params = list(struct.unpack("<" + "d" * npar, f.read(8 * npar)))
                out[cam_id] = {"model": _MODEL_ID_TO_NAME.get(model_id, f"id{model_id}"),
                               "w": w, "h": h, "params": params}
        return out
    raise FileNotFoundError(f"No cameras.bin or cameras.txt in {sparse_dir}")


def cam_fxfycxcy(cam: dict) -> tuple:
    """Pull (fx, fy, cx, cy) out of a camera's params for the common pinhole-ish models.
    Distortion coeffs are ignored on purpose — both our and Agisoft exports are undistorted
    pinholes, so we only need the linear projection part for FOV and reprojection."""
    m, p = cam["model"], cam["params"]
    if m == "PINHOLE":                                    # fx, fy, cx, cy
        return p[0], p[1], p[2], p[3]
    if m in ("OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):  # fx, fy, cx, cy, dist...
        return p[0], p[1], p[2], p[3]
    # SIMPLE_PINHOLE / SIMPLE_RADIAL / RADIAL and fallback: f, cx, cy[, dist...]
    return p[0], p[0], p[1], p[2]


def compare_intrinsics(int_ours: dict, int_ref: dict) -> dict:
    """Compare focal length + horizontal FOV between our camera(s) and Agisoft's.
    Focal is normalized as f/W so different undistorted resolutions are comparable.
    Principal point is reported as an offset from image center (both are usually centered).
    Prints a small table and returns the numbers as a dict."""
    def describe(cid, cam):
        fx, fy, cx, cy = cam_fxfycxcy(cam)
        w, h = cam["w"], cam["h"]
        return {
            "camera_id": cid, "model": cam["model"], "w": w, "h": h,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "f_over_w": fx / w,
            "fov_x_deg": math.degrees(2 * math.atan(w / (2 * fx))),
            "cx_offset_px": cx - w / 2.0, "cy_offset_px": cy - h / 2.0,
        }
    ours = [describe(cid, cam) for cid, cam in sorted(int_ours.items())]
    ref = [describe(cid, cam) for cid, cam in sorted(int_ref.items())]
    our_fw = float(np.mean([d["f_over_w"] for d in ours]))
    ref_fw = float(np.mean([d["f_over_w"] for d in ref]))
    focal_diff_pct = (our_fw - ref_fw) / ref_fw * 100.0
    our_fov = float(np.mean([d["fov_x_deg"] for d in ours]))
    ref_fov = float(np.mean([d["fov_x_deg"] for d in ref]))

    print(f"\n=== Intrinsics: ours vs Agisoft ===")
    print(f"  {'side':<8} {'cam':>4} {'model':<15} {'WxH':>11} {'focal':>9} {'f/W':>8} {'FOVx':>8} {'cx_off':>8} {'cy_off':>8}")
    for tag, lst in (("ours", ours), ("agisoft", ref)):
        for d in lst:
            print(f"  {tag:<8} {d['camera_id']:>4} {d['model']:<15} {d['w']}x{d['h']:<5} "
                  f"{d['fx']:>9.2f} {d['f_over_w']:>8.4f} {d['fov_x_deg']:>7.2f}° "
                  f"{d['cx_offset_px']:>8.1f} {d['cy_offset_px']:>8.1f}")
    print(f"  --> our focal is {focal_diff_pct:+.2f}% vs Agisoft (f/W {our_fw:.4f} vs {ref_fw:.4f}); "
          f"FOVx {our_fov:.2f}° vs {ref_fov:.2f}°")
    return {"ours": ours, "agisoft": ref,
            "our_mean_f_over_w": our_fw, "agisoft_mean_f_over_w": ref_fw,
            "focal_diff_pct": focal_diff_pct,
            "our_mean_fov_x_deg": our_fov, "agisoft_mean_fov_x_deg": ref_fov}


def load_points3d(sparse_dir: str) -> tuple:
    """Read points3D.txt/bin → (xyz [N,3], id_to_xyz dict, stored_error [N]).
    Prefers the .txt because it carries the POINT3D_ID we need for the id→xyz map (the shared
    colmap_loader reader drops the id). id_to_xyz lets the reprojection step look up a 3D point
    from an image's 2D observation."""
    import struct
    txt = os.path.join(sparse_dir, "points3D.txt")
    binp = os.path.join(sparse_dir, "points3D.bin")
    ids, xyz, errs = [], [], []
    if os.path.exists(txt):
        with open(txt) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                e = line.split()
                ids.append(int(e[0]))
                xyz.append([float(e[1]), float(e[2]), float(e[3])])
                errs.append(float(e[7]))
    elif os.path.exists(binp):
        with open(binp, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                pid, x, y, z, r, g, b, err = struct.unpack("<QdddBBBd", f.read(43))
                tl = struct.unpack("<Q", f.read(8))[0]
                f.read(8 * tl)  # skip the track (image_id, point2d_idx) pairs
                ids.append(pid)
                xyz.append([x, y, z])
                errs.append(err)
    else:
        raise FileNotFoundError(f"No points3D.txt or points3D.bin in {sparse_dir}")
    xyz = np.array(xyz)
    id_to_xyz = {pid: xyz[i] for i, pid in enumerate(ids)}
    return xyz, id_to_xyz, np.array(errs)


def compare_point_clouds(pts_ours: np.ndarray, s: float, R_align: np.ndarray,
                         t_align: np.ndarray, pts_ref: np.ndarray, trim_pct: float) -> dict:
    """Cloud-to-cloud nearest-neighbour (Chamfer) distance between our sparse points (after the
    same Umeyama transform used for the cameras) and Agisoft's metric cloud. The two clouds are
    independent triangulations with no point correspondence, so NN distance is the standard
    correspondence-free way to measure geometry agreement. Reports both directions:
    ours→agisoft (is each of our points near a real one?) and agisoft→ours (do we cover theirs?).
    Robust stats + a trimmed mean because sparse clouds have outlier triangulations."""
    from scipy.spatial import cKDTree
    P = (s * (R_align @ pts_ours.T)).T + t_align   # our cloud into Agisoft's metric frame
    d_o2a, _ = cKDTree(pts_ref).query(P, k=1)       # each of ours -> nearest agisoft
    d_a2o, _ = cKDTree(P).query(pts_ref, k=1)       # each agisoft -> nearest ours

    def stats(d):
        d_mm = d * 1000.0   # agisoft frame is metric (meters) -> mm
        cut = np.percentile(d_mm, 100 - trim_pct)   # drop the worst trim_pct% as outliers
        trimmed = d_mm[d_mm <= cut]
        return {"median_mm": float(np.median(d_mm)), "mean_mm": float(d_mm.mean()),
                "trimmed_mean_mm": float(trimmed.mean()),
                "p90_mm": float(np.percentile(d_mm, 90)),
                "p95_mm": float(np.percentile(d_mm, 95)), "max_mm": float(d_mm.max())}
    so, sa = stats(d_o2a), stats(d_a2o)
    chamfer = (so["median_mm"] + sa["median_mm"]) / 2.0

    print(f"\n=== Point cloud agreement (Chamfer NN distance, after alignment) ===")
    print(f"  points:   ours {len(P)}   agisoft {len(pts_ref)}")
    print(f"  ours->agisoft : median {so['median_mm']:.1f} mm  trimmed-mean {so['trimmed_mean_mm']:.1f} mm  p95 {so['p95_mm']:.1f} mm")
    print(f"  agisoft->ours : median {sa['median_mm']:.1f} mm  trimmed-mean {sa['trimmed_mean_mm']:.1f} mm  p95 {sa['p95_mm']:.1f} mm")
    print(f"  --> symmetric Chamfer (median): {chamfer:.1f} mm")
    return {"n_ours": len(P), "n_agisoft": len(pts_ref),
            "ours_to_agisoft": so, "agisoft_to_ours": sa,
            "symmetric_chamfer_median_mm": chamfer}


def compute_reproj_errors(ext: dict, intr: dict, id_to_xyz: dict) -> np.ndarray:
    """Recompute reprojection error from scratch: for every 2D keypoint linked to a 3D point,
    project that 3D point into the image with its camera intrinsics + pose and measure the pixel
    distance to the keypoint. Identical math for both reconstructions so the numbers are directly
    comparable. Vectorized per image. Returns the array of per-observation errors in pixels."""
    errs = []
    for img in ext.values():
        cam = intr.get(img.camera_id)
        if cam is None:
            continue
        fx, fy, cx, cy = cam_fxfycxcy(cam)
        R = qvec2rotmat(img.qvec)          # world -> camera
        t = img.tvec
        ids = img.point3D_ids
        xys = img.xys
        keep = ids >= 0
        if not keep.any():
            continue
        ids_k, xys_k = ids[keep], xys[keep]
        present = np.array([pid in id_to_xyz for pid in ids_k])
        ids_k, xys_k = ids_k[present], xys_k[present]
        if len(ids_k) == 0:
            continue
        Xw = np.array([id_to_xyz[pid] for pid in ids_k])     # [M,3] world points
        Xc = (R @ Xw.T).T + t                                # [M,3] camera coords
        front = Xc[:, 2] > 0                                 # drop points behind the camera
        Xc, xys_k = Xc[front], xys_k[front]
        if len(Xc) == 0:
            continue
        u = fx * Xc[:, 0] / Xc[:, 2] + cx
        v = fy * Xc[:, 1] / Xc[:, 2] + cy
        errs.append(np.hypot(u - xys_k[:, 0], v - xys_k[:, 1]))
    return np.concatenate(errs) if errs else np.array([])


def reproj_summary(err: np.ndarray) -> dict:
    """mean/median/p90/p95 of a reprojection-error array (px). Empty-safe."""
    if err.size == 0:
        return {"n_obs": 0}
    return {"n_obs": int(err.size), "mean_px": float(err.mean()),
            "median_px": float(np.median(err)), "p90_px": float(np.percentile(err, 90)),
            "p95_px": float(np.percentile(err, 95)), "max_px": float(err.max())}


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing/compare_to_agisoft")
def main(cfg: DictConfig):
    print("--- compare_to_agisoft config ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------------")
    t_start = time.time()

    ours_dir = os.path.join(cfg.source_path, cfg.ours_sparse_dir)
    ref_dir = os.path.join(cfg.source_path, cfg.ref_sparse_dir)

    if not os.path.isdir(ours_dir):
        print(f"ERROR: ours sparse dir not found: {ours_dir}")
        print("Did you run src/preprocessing/run_colmap.py yet?")
        sys.exit(1)
    if not os.path.isdir(ref_dir):
        print(f"ERROR: reference (agisoft) sparse dir not found: {ref_dir}")
        sys.exit(1)

    # 1. Load both reconstructions. Agisoft renames images with an "_<index>" suffix on ingestion,
    # so we normalize the agisoft side back to the original filenames for matching.
    print(f"Loading our COLMAP from: {ours_dir}")
    ext_ours = load_extrinsics(ours_dir)
    int_ours = load_intrinsics_summary(ours_dir)
    centers_ours, rots_ours = extract_poses(ext_ours)

    print(f"Loading Agisoft reference from: {ref_dir}")
    ext_ref = load_extrinsics(ref_dir)
    int_ref = load_intrinsics_summary(ref_dir)
    centers_ref, rots_ref = extract_poses(ext_ref, name_normalizer=_strip_agisoft_suffix)
    n_renamed = sum(1 for img in ext_ref.values() if _strip_agisoft_suffix(img.name) != img.name)
    if n_renamed > 0:
        print(f"  (normalized {n_renamed}/{len(ext_ref)} Agisoft filenames by stripping '_<N>' suffix before .jpg)")

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

    # 6b. Optional extra comparisons (intrinsics / point cloud / reprojection error)
    intrinsics_report = None
    points_report = None
    reproj_report = None
    id2xyz_ours = id2xyz_ref = None

    if cfg.get("compare_intrinsics", True):
        intr_ours_full = load_intrinsics_full(ours_dir)
        intr_ref_full = load_intrinsics_full(ref_dir)
        intrinsics_report = compare_intrinsics(intr_ours_full, intr_ref_full)

    if cfg.get("compare_points", True):
        xyz_ours, id2xyz_ours, _ = load_points3d(ours_dir)
        xyz_ref, id2xyz_ref, _ = load_points3d(ref_dir)
        points_report = compare_point_clouds(xyz_ours, s, R_align, t_align, xyz_ref,
                                             cfg.get("points_trim_pct", 5.0))

    if cfg.get("compare_reproj", True):
        # need the full intrinsics + the id→xyz maps; reuse if a prior block already loaded them
        if intrinsics_report is None:
            intr_ours_full = load_intrinsics_full(ours_dir)
            intr_ref_full = load_intrinsics_full(ref_dir)
        if id2xyz_ours is None:
            _, id2xyz_ours, stored_err_ours = load_points3d(ours_dir)
            _, id2xyz_ref, _ = load_points3d(ref_dir)
        else:
            _, _, stored_err_ours = load_points3d(ours_dir)
        err_ours = compute_reproj_errors(ext_ours, intr_ours_full, id2xyz_ours)
        err_ref = compute_reproj_errors(ext_ref, intr_ref_full, id2xyz_ref)
        sum_ours, sum_ref = reproj_summary(err_ours), reproj_summary(err_ref)
        # cross-check our recompute against COLMAP's stored per-point ERROR column (different
        # granularity — per-point mean vs per-observation — so we expect "close", not identical)
        stored_mean = float(stored_err_ours.mean()) if stored_err_ours.size else None
        reproj_report = {"ours_recomputed": sum_ours, "agisoft_recomputed": sum_ref,
                         "ours_stored_mean_px": stored_mean}
        print(f"\n=== Reprojection error (recomputed, same math both sides) ===")
        print(f"  ours    : mean {sum_ours.get('mean_px', float('nan')):.3f} px  "
              f"median {sum_ours.get('median_px', float('nan')):.3f} px  ({sum_ours['n_obs']} obs)")
        print(f"  agisoft : mean {sum_ref.get('mean_px', float('nan')):.3f} px  "
              f"median {sum_ref.get('median_px', float('nan')):.3f} px  ({sum_ref['n_obs']} obs)")
        if stored_mean is not None:
            print(f"  (cross-check: our recompute mean {sum_ours.get('mean_px', float('nan')):.3f} px "
                  f"vs COLMAP stored ERROR mean {stored_mean:.3f} px)")
        print(f"  NOTE: this is each model's INTERNAL self-consistency, not an ours-vs-Agisoft distance; "
              f"px not directly comparable across different resolutions.")

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
        "intrinsics": intrinsics_report,
        "point_cloud": points_report,
        "reprojection": reproj_report,
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

    elapsed = time.time() - t_start
    minutes, seconds = divmod(int(elapsed), 60)
    print("\n" + "="*50)
    print("      COMPARE_TO_AGISOFT SUMMARY")
    print("="*50)
    print(f"{'Plot:':<28} {cfg.field}/{cfg.plot}")
    print(f"{'Cameras ours:':<28} {len(centers_ours)}")
    print(f"{'Cameras agisoft:':<28} {len(centers_ref)}")
    print(f"{'Common (matched):':<28} {len(common)}")
    print(f"{'Agisoft renamed:':<28} {n_renamed}  (suffix '_<N>' stripped)")
    print("-" * 50)
    print(f"{'Umeyama scale (our→m):':<28} {s:.6f}")
    print(f"{'Mean translation err:':<28} {trans_err.mean()*1000:.2f} mm")
    print(f"{'Median translation err:':<28} {np.median(trans_err)*1000:.2f} mm")
    if rot_err is not None:
        print(f"{'Mean rotation err:':<28} {rot_err.mean():.3f}°")
        print(f"{'Median rotation err:':<28} {np.median(rot_err):.3f}°")
    if intrinsics_report is not None:
        print(f"{'Focal diff (f/W):':<28} {intrinsics_report['focal_diff_pct']:+.2f}%")
    if points_report is not None:
        print(f"{'Point cloud Chamfer:':<28} {points_report['symmetric_chamfer_median_mm']:.1f} mm (median)")
    if reproj_report is not None:
        print(f"{'Reproj err ours/agisoft:':<28} "
              f"{reproj_report['ours_recomputed'].get('mean_px', float('nan')):.2f} / "
              f"{reproj_report['agisoft_recomputed'].get('mean_px', float('nan')):.2f} px (mean)")
    print("-" * 50)
    print(f"{'TOTAL TIME:':<28} {minutes}m {seconds}s  ({elapsed:.1f}s)")
    print("="*50 + "\n")

    summary = {
        "step": "compare",
        "field": cfg.field,
        "plot": cfg.plot,
        "n_ours": len(centers_ours),
        "n_agisoft": len(centers_ref),
        "n_common": len(common),
        "scale": s,
        "mean_trans_mm": float(trans_err.mean() * 1000),
        "median_trans_mm": float(np.median(trans_err) * 1000),
        "mean_rot_deg": float(rot_err.mean()) if rot_err is not None else None,
        "median_rot_deg": float(np.median(rot_err)) if rot_err is not None else None,
        "focal_diff_pct": intrinsics_report["focal_diff_pct"] if intrinsics_report else None,
        "chamfer_median_mm": points_report["symmetric_chamfer_median_mm"] if points_report else None,
        "reproj_mean_px_ours": reproj_report["ours_recomputed"].get("mean_px") if reproj_report else None,
        "reproj_mean_px_agisoft": reproj_report["agisoft_recomputed"].get("mean_px") if reproj_report else None,
        "elapsed_s": elapsed,
    }
    summary_path = os.path.join(cfg.source_path, "logs", "compare_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    main()
