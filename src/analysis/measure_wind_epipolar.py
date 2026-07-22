"""Wind test v2 — escape the survivorship bias of measure_canopy_sway.py.

Idea: measure the epipolar (Sampson) residual of the MATCHES in the COLMAP database (before BA discards
non-rigid points) against the GLOBAL bundle-adjusted poses, and see whether that residual grows with the
TIME GAP between the two photos of a pair.

  - For a rigid, static scene: a correct match satisfies the epipolar constraint of the global poses
    exactly, so Sampson distance ~ noise, INDEPENDENT of when the two photos were taken.
  - If wind moved a wheat head between the two photos, the matched point is at two different physical
    places -> it violates the single rigid geometry -> higher Sampson distance, and the effect GROWS the
    longer the time gap (more time for the head to move). Markers (rigid) stay flat = built-in control.

We use the geometrically-verified inlier matches (two_view_geometries) but recompute Sampson against the
GLOBAL model poses, NOT the per-pair RANSAC F. Confounds (honest): verified matches are already somewhat
filtered (points moving a lot were dropped by two-view RANSAC), and a point moving ALONG its epipolar line
raises nothing -> this is a LOWER BOUND on wind. Sampson is baseline-invariant for correct rigid matches,
so wider-baseline (=longer-time) pairs do NOT inflate it on their own.

Usage (no GPU):
  python src/analysis/measure_wind_epipolar.py --field field_D --plot 20250706
  python src/analysis/measure_wind_epipolar.py            # 0706 / 0715 / 0722
"""

import os
import re
import json
import sqlite3
import argparse
from itertools import combinations

import numpy as np
import pycolmap as pc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAX_IMAGE_ID = 2147483647
TARGET_TO_CODE = {1: 113, 2: 105, 3: 89, 4: 101, 5: 85, 6: 77}
TIME_BINS = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 9999)]
MARKER_RADIUS = 45.0     # px: a keypoint within this of a marker detection is a "marker" keypoint


def img_time_seconds(name):
    """Capture time in seconds-of-day from IMG_20250706_123040~2.jpg (12:30:40)."""
    m = re.search(r"_(\d{8})_(\d{2})(\d{2})(\d{2})", name)
    return int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4)) if m else None


def pair_id_to_images(pid):
    i2 = pid % MAX_IMAGE_ID
    return (pid - i2) // MAX_IMAGE_ID, i2


def sampson(F, x1, x2):
    """Sampson distances (px) for arrays of correspondences x1,x2 (N x 2), given fundamental F."""
    x1h = np.hstack([x1, np.ones((len(x1), 1))])
    x2h = np.hstack([x2, np.ones((len(x2), 1))])
    Fx1 = (F @ x1h.T).T            # N x 3
    Ftx2 = (F.T @ x2h.T).T
    num = np.sum(x2h * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    den = np.where(den < 1e-12, np.nan, den)
    return np.sqrt(num / den)


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def load_db(db_path):
    """Return keypoints{image_id: Nx2}, matches{(id1,id2): Mx2 idx}, names{image_id:name}."""
    con = sqlite3.connect(db_path)
    names = {i: n for i, n in con.execute("SELECT image_id, name FROM images")}
    kps = {}
    for iid, rows, cols, data in con.execute("SELECT image_id, rows, cols, data FROM keypoints"):
        if data is None or rows == 0:
            continue
        kps[iid] = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)[:, :2].copy()
    matches = {}
    for pid, rows, cols, data in con.execute(
            "SELECT pair_id, rows, cols, data FROM two_view_geometries"):
        if data is None or rows == 0:
            continue
        m = np.frombuffer(data, dtype=np.uint32).reshape(rows, cols)[:, :2]
        matches[pair_id_to_images(pid)] = m
    con.close()
    return kps, matches, names


def marker_kp_mask(kps_xy, marker_xys):
    """Boolean mask: which keypoints lie within MARKER_RADIUS of any marker detection in this image."""
    if not marker_xys:
        return np.zeros(len(kps_xy), dtype=bool)
    M = np.array(marker_xys)
    mask = np.zeros(len(kps_xy), dtype=bool)
    for mx in M:
        mask |= np.hypot(kps_xy[:, 0] - mx[0], kps_xy[:, 1] - mx[1]) <= MARKER_RADIUS
    return mask


def process(field, plot, model):
    sess = os.path.join(REPO, "input_plots", "phone", field, plot)
    mdir = os.path.join(sess, model)
    db = os.path.join(sess, "distorted", "database.db")
    if not (os.path.isdir(mdir) and os.path.exists(db)):
        return None
    rec = pc.Reconstruction(mdir)
    cam = list(rec.cameras.values())[0]
    f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    Kinv = np.linalg.inv(K)

    pose = {}                                  # image name -> (R, t, time)
    for im in rec.images.values():
        T = im.cam_from_world()
        pose[im.name] = (T.rotation.matrix(), np.array(T.translation), img_time_seconds(im.name))

    # marker detections per image name (rigid control)
    markers_by_name = {}
    tri_path = os.path.join(sess, "logs", "marker_triangulation.json")
    if os.path.exists(tri_path):
        tri = json.load(open(tri_path))
        for code in TARGET_TO_CODE.values():
            for o in tri.get(str(code), []):
                if o.get("src") == "detected":
                    markers_by_name.setdefault(o["cam"], []).append(o["xy"])

    kps, matches, names = load_db(db)
    canopy_bins = {b: [] for b in TIME_BINS}
    marker_bins = {b: [] for b in TIME_BINS}

    for (id1, id2), m in matches.items():
        n1, n2 = names.get(id1), names.get(id2)
        if n1 not in pose or n2 not in pose or id1 not in kps or id2 not in kps:
            continue
        t1, t2 = pose[n1][2], pose[n2][2]
        if t1 is None or t2 is None:
            continue
        gap = abs(t1 - t2)
        # relative pose cam2_from_cam1 -> F from GLOBAL poses
        R1, tr1, _ = pose[n1]; R2, tr2, _ = pose[n2]
        Rrel = R2 @ R1.T
        trel = tr2 - Rrel @ tr1
        F = Kinv.T @ skew(trel) @ Rrel @ Kinv
        x1 = kps[id1][m[:, 0]]
        x2 = kps[id2][m[:, 1]]
        d = sampson(F, x1, x2)
        # classify each match: marker (both endpoints near a marker) vs canopy (the rest)
        mk1 = marker_kp_mask(kps[id1], markers_by_name.get(n1, []))[m[:, 0]]
        mk2 = marker_kp_mask(kps[id2], markers_by_name.get(n2, []))[m[:, 1]]
        is_marker = mk1 & mk2
        for lo, hi in TIME_BINS:
            if lo <= gap < hi:
                canopy_bins[(lo, hi)].append(d[~is_marker & np.isfinite(d)])
                marker_bins[(lo, hi)].append(d[is_marker & np.isfinite(d)])
                break

    TRUE_CEIL = 30.0     # px: a match above this is a FALSE correspondence (wrong head), not sway

    def summarize(bins):
        """Per time-gap bin: median Sampson of the TRUE-ish matches (<30px) = wind signal,
        and the fraction of FALSE matches (>=30px) = repetitive-texture confusion."""
        out = {}
        for (lo, hi), chunks in bins.items():
            arr = np.concatenate(chunks) if chunks else np.array([])
            lbl = f"{lo}-{hi if hi < 9999 else '+'}s"
            true = arr[arr < TRUE_CEIL]
            out[lbl] = {
                "true_med_px": round(float(np.median(true)), 3) if len(true) else None,
                "false_frac": round(float(np.mean(arr >= TRUE_CEIL)), 3) if len(arr) else None,
                "n": int(len(arr)),
            }
        return out

    return {"field": field, "plot": plot, "model": model,
            "canopy_sampson_px": summarize(canopy_bins),
            "marker_sampson_px": summarize(marker_bins)}


def print_report(r):
    print(f"\n=== {r['field']}/{r['plot']}  ({r['model']}) ===")
    print("  TRUE-match residual (<30px) RISING with time-gap = WIND; false_frac = repetitive-texture mismatch")
    print(f"  {'time-gap':<9} {'canopy true_med':>16} {'false%':>8} {'n':>9}   {'marker true_med':>16} {'n':>6}")
    for b in [f"{lo}-{hi if hi<9999 else '+'}s" for lo, hi in TIME_BINS]:
        c = r["canopy_sampson_px"][b]; m = r["marker_sampson_px"][b]
        fp = f"{c['false_frac']*100:.0f}%" if c['false_frac'] is not None else "-"
        print(f"  {b:<9} {str(c['true_med_px'])+' px':>16} {fp:>8} {c['n']:>9}   "
              f"{str(m['true_med_px'])+' px':>16} {m['n']:>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default=None)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--model", default="sparse/0")
    args = ap.parse_args()
    sessions = ([(args.field, args.plot)] if args.field and args.plot
                else [("field_D", "20250706"), ("field_A", "20250715"), ("field_D", "20250722")])
    results = []
    for field, plot in sessions:
        r = process(field, plot, args.model)
        if r is None:
            print(f"  {field}/{plot}: skipped (no model/db)")
            continue
        print_report(r)
        out = os.path.join(REPO, "input_plots", "phone", field, plot, "logs", "wind_epipolar.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(r, open(out, "w"), indent=2)
        results.append(r)
    if len(results) > 1:
        print("\n=== CROSS-SESSION: TRUE-match residual near vs far in time (rise = wind) ===")
        for r in results:
            c = r["canopy_sampson_px"]
            near = c.get("0-5s", {}).get("true_med_px")
            mid = c.get("15-30s", {}).get("true_med_px")
            far = c.get("60-+s", {}).get("true_med_px")
            print(f"  {r['field']+'/'+r['plot']:<22} 0-5s={near}px  15-30s={mid}px  60+s={far}px")
        print("  (known pose-drift: 0706=79mm worst, 0722=24mm, 0715=9mm best)")


if __name__ == "__main__":
    main()
