"""Validate the marker-geometry RULER (rescore_models_geometry.py) on SYNTHETIC ground truth.

Why this exists: every method verdict (GLOMAP-worse, keypoints-help, ...) came out of
rescore_models_geometry.py, but we never checked that the ruler itself reads true. This builds a fake
plot where we KNOW the answer (we typed the marker 3D positions + camera poses), projects the markers to
perfect 2D pixels by pinhole math, and then checks that the real audited functions
(triangulate / robust_triangulate / bestfit_resid, imported straight from the module under test)
recover the truth. No field, no images, no wheat -- the ruler only ever touches the 6 markers.

Three tests, on data whose true answer we know:
  1. ZERO         -- true poses + perfect detections must score ~0 cm error (else the ruler is broken).
  2. MONOTONIC    -- perturbing poses by a known, growing amount must grow the error (else it's blind).
  3. OUTLIER      -- inject fake canopy detections; the RANSAC fix must recover truth while the old
                     one-pass DLT blows up. This is what separates "correct" (recovers a KNOWN answer)
                     from "tuned" (only happens to match our real 14 sessions).

Run:  python src/analysis/test_synthetic_marker_metric.py
"""

import os
import sys
from itertools import combinations

import numpy as np

# import the REAL functions under audit -- we test the shipping code, not a copy
sys.path.insert(0, os.path.dirname(__file__))
from rescore_models_geometry import triangulate, robust_triangulate, bestfit_resid  # noqa: E402

RNG = np.random.default_rng(20260721)     # fixed seed -> reproducible (no wall-clock randomness)

# --- fake but realistic geometry (all in cm) ---
# 6 markers laid out like the real plot: a flat-ish rectangle on the ground, tiny height jitter so they
# are NOT perfectly coplanar (mirrors the real ~cm height differences).
CODES = [113, 105, 89, 101, 85, 77]
MARKERS_XY = np.array([[0, 0], [250, 0], [500, 0], [0, 350], [250, 350], [500, 350]], float)


def build_truth():
    """Invent the ground truth: 6 marker 3D positions + 90 camera poses on a circle looking inward.
    Returns (markers dict code->XYZ, poses dict name->(R,t world->cam), K, image size)."""
    markers = {}
    for code, (x, y) in zip(CODES, MARKERS_XY):
        z = RNG.normal(0, 2.0)                       # +-2 cm height jitter -> near but not perfectly flat
        markers[code] = np.array([x, y, z], float)
    center = np.mean([m for m in markers.values()], axis=0)

    # intrinsics roughly matching our undistorted phone SIMPLE_PINHOLE cameras
    W, H, f = 3850, 2878, 3000.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])

    poses = {}
    n_cams = 90
    radius, height = 450.0, 180.0                    # ~4.5 m out, ~1.8 m up -> plot fills the frame
    for i in range(n_cams):
        ang = 2 * np.pi * i / n_cams
        C = center + np.array([radius * np.cos(ang), radius * np.sin(ang), height])
        R = look_at(C, center)                       # world->cam rotation
        t = -R @ C
        poses[f"cam_{i:03d}"] = (R, t)
    return markers, poses, K, (W, H)


def look_at(C, target, world_up=np.array([0, 0, 1.0])):
    """Standard look-at: build a world->camera rotation for a camera at C aimed at target.
    Camera convention +z forward, +x right, +y down (matches the pinhole projection used everywhere here)."""
    fwd = target - C
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.vstack([right, down, fwd])             # rows = cam axes in world -> world->cam rotation


def project(X, R, t, K):
    """Pinhole-project a 3D world point through one camera. Returns (u,v) or None if behind the camera."""
    xc = R @ X + t
    if xc[2] <= 0:
        return None
    uv = K @ xc
    return (uv[0] / uv[2], uv[1] / uv[2])


def make_detections(markers, poses, K, wh, noise_px=0.0):
    """Project every marker into every camera that sees it (in front + inside the image). Optional pixel
    noise. Returns dets: code -> list of (camname, (u,v)), same shape as a real marker_triangulation.json."""
    W, H = wh
    dets = {code: [] for code in markers}
    for name, (R, t) in poses.items():
        for code, X in markers.items():
            uv = project(X, R, t, K)
            if uv is None:
                continue
            u, v = uv
            if 0 <= u < W and 0 <= v < H:
                if noise_px:
                    u += RNG.normal(0, noise_px)
                    v += RNG.normal(0, noise_px)
                dets[code].append((name, (u, v)))
    return dets


def rodrigues(axis, angle_rad):
    """Small rotation matrix about a unit axis by angle (Rodrigues formula). Used to perturb poses."""
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)


def perturb_poses(poses, rot_deg, trans_cm):
    """Independently jitter each camera's rotation + center by the given magnitude. Independent (non-rigid)
    on purpose: a rigid shift of ALL cameras leaves marker distances unchanged, so only per-camera
    inconsistency distorts the triangulated SHAPE -- which is exactly how real accumulated drift behaves."""
    out = {}
    for name, (R, t) in poses.items():
        C = -R.T @ t
        axis = RNG.normal(size=3)
        dR = rodrigues(axis, np.deg2rad(RNG.normal(0, rot_deg))) if rot_deg else np.eye(3)
        dC = RNG.normal(0, trans_cm, size=3) if trans_cm else np.zeros(3)
        Rn = dR @ R
        Cn = C + dC
        out[name] = (Rn, -Rn @ Cn)
    return out


def dists_from_dets(dets, poses, K, robust=True):
    """Triangulate each marker from its detections through the given poses, then all pairwise distances.
    robust=True uses the audited RANSAC robust_triangulate; False uses the old one-pass DLT triangulate."""
    pos = {}
    for code, d in dets.items():
        if len(d) >= 2:
            X = robust_triangulate(d, poses, K) if robust else triangulate(d, poses, K)
            if X is not None:
                pos[code] = X
    return {frozenset((a, b)): float(np.linalg.norm(pos[a] - pos[b])) for a, b in combinations(pos, 2)}, pos


def true_dists(markers):
    """The exact pairwise marker distances (our synthetic 'survey')."""
    return {frozenset((a, b)): float(np.linalg.norm(markers[a] - markers[b]))
            for a, b in combinations(markers, 2)}


def main():
    markers, poses, K, wh = build_truth()
    gt = true_dists(markers)
    print(f"Synthetic plot: {len(markers)} markers, {len(poses)} cameras. "
          f"True marker distances span {min(gt.values()):.0f}-{max(gt.values()):.0f} cm.\n")

    # ---------------- TEST 1: ZERO ----------------
    dets = make_detections(markers, poses, K, wh)                 # perfect, noise-free
    d_robust, _ = dists_from_dets(dets, poses, K, robust=True)
    d_plain, _ = dists_from_dets(dets, poses, K, robust=False)
    r_rob, n = bestfit_resid(d_robust, gt)
    r_pln, _ = bestfit_resid(d_plain, gt)
    ok1 = r_rob < 0.1
    print("TEST 1  ZERO (true poses + perfect detections -> must be ~0 cm)")
    print(f"   robust: {r_rob:.4f} cm   plain: {r_pln:.4f} cm   (over n={n} shared pairs)")
    print(f"   -> {'PASS' if ok1 else 'FAIL'}: ruler reads ~0 on truth\n")

    # ---------------- TEST 2: MONOTONIC ----------------
    print("TEST 2  MONOTONIC (perturb poses by a known growing amount -> error must grow)")
    levels = [0.0, 1.0, 2.0, 5.0, 10.0]              # rot(deg) == trans(cm) magnitude per level
    errs = []
    for m in levels:
        trials = []
        for _ in range(5):                            # average a few trials to smooth the random jitter
            pert = perturb_poses(poses, rot_deg=m * 0.1, trans_cm=m)   # 10 cm drift ~ 1 deg tilt
            d_p, _ = dists_from_dets(dets, pert, K, robust=True)
            r, _ = bestfit_resid(d_p, gt)
            if r is not None:
                trials.append(r)
        e = float(np.mean(trials))
        errs.append(e)
        print(f"   perturb {m:5.1f} cm / {m*0.1:.1f} deg  ->  shape error {e:7.3f} cm")
    ok2 = all(errs[i] < errs[i + 1] for i in range(len(errs) - 1))
    print(f"   -> {'PASS' if ok2 else 'FAIL'}: error increases monotonically with perturbation\n")

    # ---------------- TEST 3: OUTLIER (the prize) ----------------
    print("TEST 3  OUTLIER (inject fake canopy detections -> RANSAC must recover truth, old DLT must not)")
    print("   Median marker 3D-position error vs known truth (cm):")
    print(f"   {'outlier frac':>12} {'RANSAC (fix)':>14} {'old DLT':>12}")
    ok3 = True
    for frac in [0.0, 0.2, 0.4, 0.6]:
        rob_errs, pln_errs = [], []
        for _ in range(5):
            noisy = {code: list(v) for code, v in dets.items()}
            for code, obs in noisy.items():
                k = int(round(frac * len(obs)))
                idx = RNG.choice(len(obs), size=k, replace=False) if k else []
                for ii in idx:                        # replace real detection with a random pixel (canopy FP)
                    name = obs[ii][0]
                    obs[ii] = (name, (RNG.uniform(0, wh[0]), RNG.uniform(0, wh[1])))
            for code in noisy:
                Xr = robust_triangulate(noisy[code], poses, K)
                Xp = triangulate(noisy[code], poses, K)
                if Xr is not None:
                    rob_errs.append(np.linalg.norm(Xr - markers[code]))
                if Xp is not None:
                    pln_errs.append(np.linalg.norm(Xp - markers[code]))
        mr = float(np.median(rob_errs)) if rob_errs else float("nan")
        mp = float(np.median(pln_errs)) if pln_errs else float("nan")
        print(f"   {frac:>12.1f} {mr:>14.3f} {mp:>12.3f}")
        if frac >= 0.2 and not (mr < 5.0 and mp > 5 * max(mr, 0.1)):
            ok3 = False
    print(f"   -> {'PASS' if ok3 else 'FAIL'}: RANSAC stays near truth under outliers while DLT diverges\n")

    print("=" * 60)
    verdict = "RULER VALIDATED" if (ok1 and ok2 and ok3) else "RULER PROBLEM -- see failing test"
    print(f"OVERALL: {'PASS' if (ok1 and ok2 and ok3) else 'FAIL'}  ({verdict})")
    print("=" * 60)


if __name__ == "__main__":
    main()
