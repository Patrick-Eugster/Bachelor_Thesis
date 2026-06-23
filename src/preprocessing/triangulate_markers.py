"""Triangulate the per-image marker detections into ONE 3D point per marker, using COLMAP poses.

Modular + detector-agnostic: reads a detections JSON (any detector version — default the v8 manifest
run) + a COLMAP `sparse/0/` model. For each manifest marker ID it gathers every view that saw it,
back-projects the pixels into rays with the known camera poses, and finds the 3D point closest to all
the rays (DLT + RANSAC outlier rejection + least-squares refine). Then it:
  * SNAPS near-neighbour misreads back to the right marker by LOCATION (e.g. 117 -> 85 at target 5's
    3D spot — the thing code/Hamming alone can't decide because 117 is Hamming-1 from 85/101/113),
  * REPROJECTS each 3D point into every registered image to recover detections that were missed
    (glare/blur frames) — the same "Pinned" trick Agisoft uses.

Outputs (READ-ONLY on the data, writes into the plot's logs/ + a vis folder):
  logs/marker_points3d.json         the 6 3D points + per-marker quality (reproj err, parallax, #views)
  logs/marker_triangulation.json    every observation per marker (detected / snapped / reprojected)
  marker_vis_v8manifest_triangulated/*.png  overlays: detected (green) / snapped (orange) / reprojected (blue)

Theory + the manifest/Hamming background: docs/MARKER_CODE_STRUCTURE.md, docs/MARKER_INTEGRATION_PLAN.md.

Usage:
    python src/preprocessing/triangulate_markers.py field=field_A plot=20250609
    python src/preprocessing/triangulate_markers.py field=field_D plot=20250523 write_overlays=false
"""

import json
import os
import shutil
import sys
import time

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import least_squares

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "gaussians", "scene"))
from colmap_loader import qvec2rotmat, read_extrinsics_text, read_intrinsics_text  # noqa: E402
import marker_codes  # noqa: E402


def build_camera(cam):
    """Intrinsic matrix K from a COLMAP camera (SIMPLE_PINHOLE: f,cx,cy / PINHOLE: fx,fy,cx,cy)."""
    p = cam.params
    if cam.model == "SIMPLE_PINHOLE":
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    else:  # PINHOLE
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K, cam.width, cam.height


def projection_matrix(K, qvec, tvec):
    """3x4 projection P = K[R|t] (COLMAP world->cam: X_cam = R X_world + t) plus R,t for cheirality."""
    R = qvec2rotmat(qvec)
    t = np.asarray(tvec, dtype=np.float64)
    P = K @ np.hstack([R, t.reshape(3, 1)])
    return P, R, t


def project(P, X):
    """Project a 3D point to pixel coords via a 3x4 matrix."""
    xh = P @ np.append(X, 1.0)
    return xh[:2] / xh[2]


def dlt_triangulate(obs):
    """Linear (DLT) triangulation from a list of (P, (x,y)). Returns the 3D point (SVD null-space)."""
    A = []
    for P, (x, y) in obs:
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def reproj_errors(X, obs):
    """Per-view reprojection error (px) of a 3D point against its observations."""
    return np.array([np.linalg.norm(project(P, X) - np.array(xy)) for P, xy in obs])


def refine(X, obs):
    """Nonlinear refine: minimise total reprojection residual over the (inlier) observations."""
    def resid(x):
        r = []
        for P, xy in obs:
            r.extend(project(P, x) - np.array(xy))
        return r
    sol = least_squares(resid, X, method="lm")
    return sol.x


def ransac_triangulate(obs, thresh, iters, rng):
    """Robust triangulation: sample minimal pairs, keep the 3D point with the most inliers (reproj <
    thresh), then refine on the inliers. Falls back to plain DLT for 2 views. Returns (X, inlier_mask)."""
    n = len(obs)
    if n <= 2:
        X = dlt_triangulate(obs)
        return X, np.ones(n, dtype=bool)
    best_inl, best_X = None, None
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(pairs)
    for (i, j) in pairs[:iters]:
        try:
            X = dlt_triangulate([obs[i], obs[j]])
        except np.linalg.LinAlgError:
            continue
        err = reproj_errors(X, obs)
        inl = err < thresh
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best_X = inl, X
    if best_inl is None or best_inl.sum() < 2:
        X = dlt_triangulate(obs)
        return X, np.ones(n, dtype=bool)
    inl_obs = [obs[k] for k in range(n) if best_inl[k]]
    X = refine(dlt_triangulate(inl_obs), inl_obs)
    # re-evaluate inliers around the refined point
    inl = reproj_errors(X, obs) < thresh
    if inl.sum() >= 2:
        inl_obs = [obs[k] for k in range(n) if inl[k]]
        X = refine(dlt_triangulate(inl_obs), inl_obs)
    return X, inl


def cam_center(R, t):
    """Camera centre in world coords."""
    return -R.T @ t


def max_parallax_deg(X, cams):
    """Largest angle (deg) between the rays from the inlier cameras to the point = triangulation
    angle. Low parallax = depth poorly constrained (rays nearly parallel)."""
    dirs = []
    for (R, t) in cams:
        d = X - cam_center(R, t)
        n = np.linalg.norm(d)
        if n > 1e-9:
            dirs.append(d / n)
    best = 0.0
    for a in range(len(dirs)):
        for b in range(a + 1, len(dirs)):
            c = np.clip(np.dot(dirs[a], dirs[b]), -1.0, 1.0)
            best = max(best, np.degrees(np.arccos(c)))
    return best


def solve_marker(ob, views, cfg, rng):
    """Triangulate one marker from its observations (list of {cam,xy,...}); fills inlier/reproj_px on
    each obs and returns the points3d entry (3D point + quality), or None if too few views."""
    if len(ob) < cfg.min_views:
        return None
    tri_obs = [(views[o["cam"]]["P"], o["xy"]) for o in ob]
    X, inl = ransac_triangulate(tri_obs, cfg.reproj_thresh_px, cfg.ransac_iters, rng)
    for o, good in zip(ob, inl):
        o["inlier"] = bool(good)
        o["reproj_px"] = float(np.linalg.norm(project(views[o["cam"]]["P"], X) - np.array(o["xy"])))
    if inl.sum() < cfg.min_views:
        return None
    inl_cams = [(views[o["cam"]]["R"], views[o["cam"]]["t"]) for o in ob if o["inlier"]]
    errs = np.array([o["reproj_px"] for o in ob if o["inlier"]])
    return {
        "xyz": [float(v) for v in X],
        "n_views": len(ob), "n_inliers": int(inl.sum()),
        "mean_reproj_px": float(errs.mean()), "median_reproj_px": float(np.median(errs)),
        "max_reproj_px": float(errs.max()), "parallax_deg": float(max_parallax_deg(X, inl_cams)),
    }


def load_detections(path, key):
    """Read a detections JSON's `per_image` (or `per_image_dropped`) into {image_name: [(x,y,id)]}."""
    data = json.load(open(path))
    out = {}
    for cam, lst in data.get(key, {}).items():
        out[cam] = [(float(d["center"][0]), float(d["center"][1]), int(d["id"])) for d in lst]
    return out, data


@hydra.main(version_base=None, config_path="../../configs",
            config_name="preprocessing/triangulate_markers")
def main(cfg: DictConfig):
    """Triangulate manifest markers from a detections JSON + COLMAP poses; snap misreads + recover
    missed frames by reprojection; write 3D points + quality + overlays."""
    print("--- triangulate_markers config ---")
    print(OmegaConf.to_yaml(cfg))
    print("----------------------------------")
    t0 = time.time()

    sparse = os.path.join(cfg.source_path, cfg.sparse_dir)
    cams_raw = read_intrinsics_text(os.path.join(sparse, "cameras.txt"))
    imgs_raw = read_extrinsics_text(os.path.join(sparse, "images.txt"))
    # per registered image name -> (P, R, t, W, H)
    views = {}
    for im in imgs_raw.values():
        K, W, H = build_camera(cams_raw[im.camera_id])
        P, R, t = projection_matrix(K, im.qvec, im.tvec)
        views[im.name] = {"P": P, "R": R, "t": t, "W": W, "H": H}
    print(f"COLMAP model: {len(views)} registered images, {len(cams_raw)} camera(s).")

    det_path = os.path.join(cfg.source_path, cfg.detections_json)
    kept, data = load_detections(det_path, "per_image")
    dropped, _ = load_detections(det_path, "per_image_dropped")
    manifest = [int(x) for x in (data.get("id_filter", {}).get("manifest")
                                 or list(cfg.plot_manifest))]
    print(f"Detections: {det_path}\nManifest: {sorted(manifest)}")

    # 1. group kept manifest detections by id -> observations in registered views only
    obs_by_id = {m: [] for m in manifest}
    for cam, dets in kept.items():
        if cam not in views:
            continue
        for (x, y, i) in dets:
            if i in obs_by_id:
                obs_by_id[i].append({"cam": cam, "xy": (x, y), "src": "detected"})

    rng = np.random.default_rng(0)
    points3d, observations = {}, {}

    # 2. triangulate each marker (RANSAC + refine)
    for m in manifest:
        ob = obs_by_id[m]
        points3d[m] = solve_marker(ob, views, cfg, rng)
        observations[m] = ob

    n_snapped = 0
    used = set()                       # dropped dets (cam, x, y) already consumed, so step 3 won't reuse

    def hamming_ok(code, m):
        """Optional code guard. Default OFF (snap_hamming_max<=0) → assignment is PURELY by location
        (robust to multi-bit occlusion misreads that Hamming would wrongly exclude). RANSAC + the
        reproj threshold are the real safeguards. Set snap_hamming_max>0 to also require the decoded
        code to be within N bits of the marker."""
        return cfg.snap_hamming_max <= 0 or marker_codes.hamming(code, m) <= cfg.snap_hamming_max

    # 2b. SEED an under-covered marker from its dropped misreads (e.g. 85 from the 117s): triangulate
    # the Hamming-near dropped detections that don't already belong to a solved marker's 3D point.
    if cfg.snap_enabled:
        for m in manifest:
            if points3d[m] is not None:
                continue
            cand = []
            for cam, dets in dropped.items():
                if cam not in views:
                    continue
                V = views[cam]
                for (x, y, code) in dets:
                    if not hamming_ok(code, m):
                        continue
                    # skip if it lands on an already-solved marker (it belongs to that one) — pure geometry
                    claimed = any(points3d[sm] and
                                  np.linalg.norm(project(V["P"], np.array(points3d[sm]["xyz"]))
                                                 - np.array([x, y])) < cfg.snap_tol_px
                                  for sm in manifest if sm != m)
                    if not claimed:
                        cand.append({"cam": cam, "xy": (x, y), "src": "snapped", "from_code": int(code)})
            entry = solve_marker(cand, views, cfg, rng)
            if entry is not None:
                points3d[m] = entry
                seeded = [o for o in cand if o.get("inlier")]
                observations[m] = seeded
                for o in seeded:
                    used.add((o["cam"], round(o["xy"][0], 1), round(o["xy"][1], 1)))
                n_snapped += len(seeded)

    # 3. SNAP near-neighbour misreads (dropped) back to a marker by LOCATION + Hamming guard
    if cfg.snap_enabled:
        for cam, dets in dropped.items():
            if cam not in views:
                continue
            V = views[cam]
            for (x, y, code) in dets:
                if (cam, round(x, 1), round(y, 1)) in used:   # already used as a seed
                    continue
                best_m, best_d = None, cfg.snap_tol_px
                for m in manifest:
                    if points3d[m] is None:
                        continue
                    if not hamming_ok(code, m):
                        continue
                    proj = project(V["P"], np.array(points3d[m]["xyz"]))
                    d = float(np.linalg.norm(proj - np.array([x, y])))
                    if d < best_d:
                        best_d, best_m = d, m
                if best_m is not None:
                    observations[best_m].append({"cam": cam, "xy": (x, y), "src": "snapped",
                                                 "from_code": int(code), "reproj_px": best_d})
                    n_snapped += 1

    # 4. REPROJECT each 3D point into every registered image to recover missed frames
    n_recovered = 0
    margin = cfg.image_margin_px
    detected_cams = {m: {o["cam"] for o in observations[m]} for m in manifest}
    for m in manifest:
        if points3d[m] is None:
            continue
        X = np.array(points3d[m]["xyz"])
        for cam, V in views.items():
            if cam in detected_cams[m]:
                continue
            depth = (V["R"] @ X + V["t"])[2]
            if depth <= 0:
                continue
            px, py = project(V["P"], X)
            if -margin <= px <= V["W"] + margin and -margin <= py <= V["H"] + margin:
                observations[m].append({"cam": cam, "xy": [float(px), float(py)],
                                        "src": "reprojected"})
                n_recovered += 1

    # ---- write outputs ----
    logs = os.path.join(cfg.source_path, "logs")
    os.makedirs(logs, exist_ok=True)
    pts_out = {str(m): points3d[m] for m in manifest}
    with open(os.path.join(logs, cfg.points_json), "w") as f:
        json.dump({"field": cfg.field, "plot": cfg.plot, "manifest": sorted(manifest),
                   "n_registered_images": len(views), "points3d": pts_out,
                   "n_snapped": n_snapped, "n_reprojected": n_recovered}, f, indent=2)
    with open(os.path.join(logs, cfg.obs_json), "w") as f:
        json.dump({str(m): observations[m] for m in manifest}, f, indent=2)

    if cfg.write_overlays:
        vis_dir = os.path.join(cfg.source_path, cfg.output_vis_dir)
        # wipe first so a re-run on a smaller image set can't leave stale overlays from a previous run
        shutil.rmtree(vis_dir, ignore_errors=True)
        os.makedirs(vis_dir, exist_ok=True)
        col = {"detected": (0, 200, 0), "snapped": (0, 165, 255), "reprojected": (255, 100, 0)}
        by_cam = {}
        for m in manifest:
            for o in observations[m]:
                by_cam.setdefault(o["cam"], []).append((m, o["xy"], o["src"]))
        img_dir = os.path.join(cfg.source_path, "images")
        for cam, items in by_cam.items():
            img = cv2.imread(os.path.join(img_dir, cam))
            if img is None:
                continue
            r = max(18, int(0.012 * max(img.shape[:2])))
            for m, (x, y), src in items:
                p = (int(round(x)), int(round(y)))
                cv2.circle(img, p, r, col[src], 4)
                cv2.putText(img, str(m), (p[0] + r, p[1]), cv2.FONT_HERSHEY_SIMPLEX,
                            1.4, col[src], 4)
            W = img.shape[1]
            if cfg.overlay_max_width and W > cfg.overlay_max_width:
                s = cfg.overlay_max_width / W
                img = cv2.resize(img, (int(W * s), int(img.shape[0] * s)))
            cv2.imwrite(os.path.join(vis_dir, cam), img)

    # ---- summary ----
    elapsed = time.time() - t0
    print("\n" + "=" * 64)
    print("      MARKER TRIANGULATION SUMMARY")
    print("=" * 64)
    print(f"{'Plot:':<26} {cfg.field}/{cfg.plot}")
    print(f"{'Registered images:':<26} {len(views)}")
    print("-" * 64)
    print(f"{'id':>5} {'views':>6} {'inl':>4} {'snap':>5} {'reproj':>7} "
          f"{'med_px':>7} {'max_px':>7} {'parallax°':>9}")
    for m in manifest:
        p = points3d[m]
        if p is None:
            print(f"{m:>5} {len(observations[m]):>6}  ---  (too few views to triangulate)")
            continue
        ns = sum(1 for o in observations[m] if o["src"] == "snapped")
        nr = sum(1 for o in observations[m] if o["src"] == "reprojected")
        print(f"{m:>5} {p['n_views']:>6} {p['n_inliers']:>4} {ns:>5} {nr:>7} "
              f"{p['median_reproj_px']:>7.2f} {p['max_reproj_px']:>7.1f} {p['parallax_deg']:>9.1f}")
    print("-" * 64)
    print(f"snapped (misreads recovered): {n_snapped}   reprojected (missed frames): {n_recovered}")
    mm, ss = divmod(int(elapsed), 60)
    print(f"{'TOTAL TIME:':<26} {mm}m {ss}s")
    print(f"3D points  -> {os.path.join('logs', cfg.points_json)}")
    print(f"observations -> {os.path.join('logs', cfg.obs_json)}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
