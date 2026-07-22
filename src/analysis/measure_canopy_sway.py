"""Measure wheat-canopy SWAY (wind) directly from the still SfM images, using the rigid markers as a
built-in control. Motivation: on a windy day the wheat heads move between photos, breaking SfM's
static-scene assumption -> pose drift. The stills are NOT the video frames, so we can't use the video;
we test the actual SfM inputs instead.

Two signals (both per session, written to logs/canopy_sway.json):

  SIGNAL 1 - marker vs canopy reprojection error.
    Markers are bolted to stakes = rigid; we localise them sub-pixel. Scene (canopy) points are wheat.
    Compare median per-observation reprojection error: canopy / marker. >1 means the wheat fits the
    rigid model worse than the markers. (CONFOUND: canopy points are also just harder to localise -
    thin, repetitive - so a high ratio alone is not proof of motion. Signal 2 is the clincher.)

  SIGNAL 2 - THE WIND TEST: pairwise-triangulation deviation vs TIME GAP, parallax-controlled.
    For a physical point seen in views i and j, triangulate it from just that pair. If the point is
    rigid, every pair agrees regardless of when the two photos were taken. If wind moves it over time,
    pairs far apart IN TIME triangulate to more-different positions. We only use pairs whose parallax
    angle is in a fixed band, so baseline/parallax is held ~constant and the ONLY thing varying is the
    time gap between the two photos (from the HHMMSS in the filename). We bin pairs by time gap and
    report the median deviation-from-center per bin, separately for markers (rigid control, should be
    FLAT) and canopy. Canopy deviation RISING with time gap = wind; markers flat = the control holds.

Deviations are normalised by the session's median camera-baseline (scene scale) so sessions compare.

Usage (no GPU):
  python src/analysis/measure_canopy_sway.py --field field_D --plot 20250706
  python src/analysis/measure_canopy_sway.py            # all 3 benchmark sessions
"""

import os
import re
import json
import argparse
from itertools import combinations

import numpy as np
import pycolmap as pc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TARGET_TO_CODE = {1: 113, 2: 105, 3: 89, 4: 101, 5: 85, 6: 77}
TIME_BINS = [(0, 10), (10, 30), (30, 60), (60, 9999)]   # seconds
PARALLAX_BAND = (8.0, 45.0)                              # deg: hold baseline ~constant across pairs


def img_time_seconds(name):
    """Parse capture time from a filename like IMG_20250706_123040~2.jpg -> seconds-of-day (12:30:40)."""
    m = re.search(r"_(\d{8})_(\d{2})(\d{2})(\d{2})", name)
    if not m:
        return None
    h, mm, s = int(m.group(2)), int(m.group(3)), int(m.group(4))
    return h * 3600 + mm * 60 + s


def cam_center(R, t):
    """Camera center in world coords from world->cam (R,t)."""
    return -R.T @ t


def parallax_deg(Ca, Cb, X):
    """Angle at X between the two camera viewing directions (deg)."""
    a, b = Ca - X, Cb - X
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return np.degrees(np.arccos(np.clip((a @ b) / (na * nb), -1, 1)))


def triangulate_pair(P1, x1, P2, x2):
    """DLT triangulation of one point from two 3x4 projection matrices + pixel coords."""
    A = np.array([x1[0] * P1[2] - P1[0], x1[1] * P1[2] - P1[1],
                  x2[0] * P2[2] - P2[0], x2[1] * P2[2] - P2[1]])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def timegap_profile(observations, K_of, P_of, C_of, scene_scale):
    """observations = [(view_key, time_s, xy), ...] for ONE physical point.
    Triangulate every parallax-banded pair, measure each pair's deviation from the point's robust
    center, and bucket (time_gap -> deviation/scene_scale). Returns list of (time_gap, norm_dev)."""
    pts, meta = [], []
    for (ka, ta, xa), (kb, tb, xb) in combinations(observations, 2):
        X = triangulate_pair(P_of[ka], xa, P_of[kb], xb)
        if parallax_deg(C_of[ka], C_of[kb], X) < PARALLAX_BAND[0] or \
           parallax_deg(C_of[ka], C_of[kb], X) > PARALLAX_BAND[1]:
            continue
        pts.append(X)
        meta.append(abs(ta - tb))
    if len(pts) < 3:
        return []
    center = np.median(np.array(pts), axis=0)              # robust center of this point's pair-estimates
    out = []
    for X, gap in zip(pts, meta):
        out.append((gap, float(np.linalg.norm(X - center)) / scene_scale))
    return out


def bin_by_time(pairs):
    """pairs=[(time_gap, norm_dev),...] -> {bin_label: median_dev} + slope (dev vs gap)."""
    res = {}
    for lo, hi in TIME_BINS:
        d = [dev for g, dev in pairs if lo <= g < hi]
        res[f"{lo}-{hi if hi < 9999 else '+'}s"] = (round(float(np.median(d)), 4) if d else None, len(d))
    gaps = np.array([g for g, _ in pairs]); devs = np.array([d for _, d in pairs])
    slope = float(np.polyfit(gaps, devs, 1)[0]) if len(gaps) >= 5 and np.ptp(gaps) > 0 else None
    return res, slope


def process(field, plot, model, n_canopy, seed):
    sess = os.path.join(REPO, "input_plots", "phone", field, plot)
    mdir = os.path.join(sess, model)
    if not os.path.isdir(mdir):
        return None
    rec = pc.Reconstruction(mdir)

    # per-image pose, projection matrix, center, time
    K_of, P_of, C_of, T_of = {}, {}, {}, {}
    for im in rec.images.values():
        cam = rec.camera(im.camera_id)
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        Tw = im.cam_from_world()
        R, t = Tw.rotation.matrix(), np.array(Tw.translation)
        key = im.name
        K_of[key] = K
        P_of[key] = K @ np.hstack([R, t.reshape(3, 1)])
        C_of[key] = cam_center(R, t)
        T_of[key] = img_time_seconds(im.name)
    if any(v is None for v in T_of.values()):
        missing = [n for n, v in T_of.items() if v is None][:3]
        print(f"  {field}/{plot}: WARN no timestamp on some images e.g. {missing}")

    # scene scale = median camera-camera distance (to normalise deviations across sessions)
    centers = np.array(list(C_of.values()))
    idx = np.random.default_rng(0).choice(len(centers), min(len(centers), 40), replace=False)
    dists = [np.linalg.norm(centers[a] - centers[b]) for a, b in combinations(idx, 2)]
    scene_scale = float(np.median(dists)) if dists else 1.0

    span = [t for t in T_of.values() if t is not None]
    sweep_s = (max(span) - min(span)) if span else 0

    # ---- CANOPY: sample scene points with enough observations ----
    rng = np.random.default_rng(seed)
    pt_ids = list(rec.points3D.keys())
    rng.shuffle(pt_ids)
    canopy_pairs, canopy_reproj = [], []
    used = 0
    for pid in pt_ids:
        if used >= n_canopy:
            break
        p3 = rec.point3D(pid)
        obs = []
        for el in p3.track.elements:
            im = rec.images[el.image_id]
            key = im.name
            if T_of[key] is None:
                continue
            xy = np.array(im.points2D[el.point2D_idx].xy)
            obs.append((key, T_of[key], xy))
            # per-observation reprojection error vs the model's stored 3D point
            uv = P_of[key] @ np.append(p3.xyz, 1.0)
            if uv[2] > 0:
                canopy_reproj.append(np.hypot(uv[0] / uv[2] - xy[0], uv[1] / uv[2] - xy[1]))
        if len(obs) < 5:
            continue
        prof = timegap_profile(obs, K_of, P_of, C_of, scene_scale)
        if prof:
            canopy_pairs.extend(prof)
            used += 1

    # ---- MARKERS: rigid control, from our detections ----
    marker_pairs, marker_reproj = [], []
    tri_path = os.path.join(sess, "logs", "marker_triangulation.json")
    if os.path.exists(tri_path):
        tri = json.load(open(tri_path))
        for code in TARGET_TO_CODE.values():
            dets = [o for o in tri.get(str(code), [])
                    if o.get("src") == "detected" and o["cam"] in P_of and T_of[o["cam"]] is not None]
            if len(dets) < 5:
                continue
            obs = [(o["cam"], T_of[o["cam"]], np.array(o["xy"])) for o in dets]
            # global triangulation for reproj baseline
            allX = [triangulate_pair(P_of[a[0]], a[2], P_of[b[0]], b[2]) for a, b in combinations(obs, 2)]
            Xg = np.median(np.array(allX), axis=0)
            for k, _, xy in obs:
                uv = P_of[k] @ np.append(Xg, 1.0)
                if uv[2] > 0:
                    marker_reproj.append(np.hypot(uv[0] / uv[2] - xy[0], uv[1] / uv[2] - xy[1]))
            prof = timegap_profile(obs, K_of, P_of, C_of, scene_scale)
            marker_pairs.extend(prof)

    canopy_bins, canopy_slope = bin_by_time(canopy_pairs) if canopy_pairs else ({}, None)
    marker_bins, marker_slope = bin_by_time(marker_pairs) if marker_pairs else ({}, None)
    return {
        "field": field, "plot": plot, "model": model,
        "n_images": rec.num_images(), "sweep_span_s": sweep_s,
        "scene_scale": round(scene_scale, 4),
        "signal1_reproj_px": {
            "marker_median": round(float(np.median(marker_reproj)), 3) if marker_reproj else None,
            "canopy_median": round(float(np.median(canopy_reproj)), 3) if canopy_reproj else None,
            "canopy_over_marker": (round(float(np.median(canopy_reproj) / np.median(marker_reproj)), 2)
                                   if marker_reproj and canopy_reproj else None),
        },
        "signal2_timegap_deviation_normalised": {
            "canopy_bins": canopy_bins, "canopy_slope_per_s": canopy_slope,
            "marker_bins": marker_bins, "marker_slope_per_s": marker_slope,
            "n_canopy_points": used, "n_canopy_pairs": len(canopy_pairs),
            "n_marker_pairs": len(marker_pairs),
        },
    }


def print_report(r):
    print(f"\n=== {r['field']}/{r['plot']}  ({r['model']}, {r['n_images']} imgs, "
          f"sweep {r['sweep_span_s']}s) ===")
    s1 = r["signal1_reproj_px"]
    print(f"  SIGNAL 1 reproj: marker {s1['marker_median']}px  canopy {s1['canopy_median']}px  "
          f"ratio canopy/marker = {s1['canopy_over_marker']}")
    s2 = r["signal2_timegap_deviation_normalised"]
    print(f"  SIGNAL 2 deviation-vs-timegap (normalised by scene scale; RISING canopy = wind):")
    print(f"    {'time-gap bin':<12} {'canopy':>12} {'marker (ctrl)':>14}")
    for b in [f"{lo}-{hi if hi<9999 else '+'}s" for lo, hi in TIME_BINS]:
        cv = s2["canopy_bins"].get(b, (None, 0)); mv = s2["marker_bins"].get(b, (None, 0))
        print(f"    {b:<12} {str(cv[0]):>8} (n={cv[1]:<4}) {str(mv[0]):>8} (n={mv[1]})")
    print(f"    slope/s: canopy {s2['canopy_slope_per_s']}  marker {s2['marker_slope_per_s']}  "
          f"[{s2['n_canopy_points']} canopy pts]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default=None)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--model", default="sparse/0")
    ap.add_argument("--n_canopy", type=int, default=400, help="canopy points to sample per session")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.field and args.plot:
        sessions = [(args.field, args.plot)]
    else:
        sessions = [("field_D", "20250706"), ("field_A", "20250715"), ("field_D", "20250722")]

    all_res = []
    for field, plot in sessions:
        r = process(field, plot, args.model, args.n_canopy, args.seed)
        if r is None:
            print(f"  {field}/{plot}: skipped (no {args.model})")
            continue
        print_report(r)
        out = os.path.join(REPO, "input_plots", "phone", field, plot, "logs", "canopy_sway.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(r, open(out, "w"), indent=2)
        all_res.append(r)

    if len(all_res) > 1:
        print("\n=== CROSS-SESSION SUMMARY (does sway rank match pose-drift rank?) ===")
        print(f"  {'session':<22} {'canopy/marker reproj':>20} {'canopy timegap-slope/s':>24}")
        for r in all_res:
            print(f"  {r['field']+'/'+r['plot']:<22} "
                  f"{str(r['signal1_reproj_px']['canopy_over_marker']):>20} "
                  f"{str(r['signal2_timegap_deviation_normalised']['canopy_slope_per_s']):>24}")
        print("  (known pose-drift vs Agisoft: 0706=79mm worst, 0722=24mm, 0715=9mm best)")


if __name__ == "__main__":
    main()
