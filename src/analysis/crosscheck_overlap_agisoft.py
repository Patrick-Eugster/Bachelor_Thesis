"""Cross-check our 'which image pairs overlap' classification against TWO independent references, to break
the circularity of judging our reconstruction with our own reconstruction:

  (A) Agisoft's SEPARATE reconstruction (agisoft/sparse/0) — a different SfM method. co-visibility =
      #3D points two images share. Independent of our poses. (Agisoft is NOT ground truth, just a 2nd
      opinion: if BOTH methods say a pair doesn't overlap, that's robust; if they disagree, it's ambiguous.)
  (B) The decoded coded MARKERS (logs/marker_triangulation.json) — markers are identified from pixels, no
      reconstruction involved. If two images both detect the SAME coded marker, they provably overlap there.

For every geometrically-verified match pair in our database we report our co-visibility, Agisoft's, whether
they share a decoded marker, and our epipolar false-match fraction. If the pairs we call 'no overlap'
(where ~all matches look false) are ALSO no-overlap by Agisoft AND share no marker, then those matches are
necessarily false (no shared scene to match) — confirmed without trusting our own poses.

Usage:  python src/analysis/crosscheck_overlap_agisoft.py --field field_D --plot 20250706
"""

import os
import re
import json
import sqlite3
import argparse

import numpy as np
import pycolmap as pc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAX_IMAGE_ID = 2147483647
TARGET_TO_CODE = {1: 113, 2: 105, 3: 89, 4: 101, 5: 85, 6: 77}


def canon(full):
    """Canonical key IMG_<date>_<time> to bridge our names to Agisoft's renamed ones (drops any trailing
    _<seq> or ~N). Used ONLY for the our<->Agisoft mapping, never for our-internal co-visibility."""
    stem = os.path.splitext(os.path.basename(full))[0]
    m = re.match(r"(IMG_\d{8}_\d{6})", stem)
    return m.group(1) if m else stem


def our_covis_map(model_dir):
    """Points seen per image, keyed by FULL image name (our names are all distinct)."""
    rec = pc.Reconstruction(model_dir)
    seen = {im.name: set(p.point3D_id for p in im.points2D if p.has_point3D())
            for im in rec.images.values()}
    return rec, seen


def agi_covis_map(model_dir):
    """Points seen per image, keyed by canonical IMG_date_time (Agisoft renamed the files)."""
    rec = pc.Reconstruction(model_dir)
    seen = {}
    for im in rec.images.values():
        seen[canon(im.name)] = set(p.point3D_id for p in im.points2D if p.has_point3D())
    return seen


def epi_false_frac(kps, i1, i2, m, K, Kinv, pose, n1, n2):
    """Fraction of matches with Sampson residual >30px against OUR poses (our false-match estimate)."""
    def skew(t):
        return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    R1, t1 = pose[n1]; R2, t2 = pose[n2]
    Rrel = R2 @ R1.T; trel = t2 - Rrel @ t1
    F = Kinv.T @ skew(trel) @ Rrel @ Kinv
    x1 = np.c_[kps[i1][m[:, 0]], np.ones(len(m))]
    x2 = np.c_[kps[i2][m[:, 1]], np.ones(len(m))]
    Fx1 = (F @ x1.T).T; Ftx2 = (F.T @ x2.T).T
    num = np.sum(x2 * Fx1, 1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    den = np.where(den < 1e-12, np.nan, den)
    d = np.sqrt(num / den); d = d[np.isfinite(d)]
    return float(np.mean(d > 30)) if len(d) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="field_D")
    ap.add_argument("--plot", default="20250706")
    args = ap.parse_args()
    sess = os.path.join(REPO, "input_plots", "phone", args.field, args.plot)

    rec, our_seen = our_covis_map(os.path.join(sess, "sparse", "0"))
    cam = list(rec.cameras.values())[0]; f, cx, cy = cam.params[:3]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]); Kinv = np.linalg.inv(K)
    pose = {}
    for im in rec.images.values():
        T = im.cam_from_world()
        pose[im.name] = (T.rotation.matrix(), np.array(T.translation))

    agi_dir = os.path.join(sess, "agisoft", "sparse", "0")
    agi_seen = agi_covis_map(agi_dir) if os.path.isdir(agi_dir) else None

    # decoded markers per image (FULL our-name) — fully independent overlap signal
    markers_by_name = {}
    tri = os.path.join(sess, "logs", "marker_triangulation.json")
    if os.path.exists(tri):
        t = json.load(open(tri))
        for code in TARGET_TO_CODE.values():
            for o in t.get(str(code), []):
                if o.get("src") == "detected":
                    markers_by_name.setdefault(o["cam"], set()).add(code)

    con = sqlite3.connect(os.path.join(sess, "distorted", "database.db"))
    names = {i: n for i, n in con.execute("SELECT image_id, name FROM images")}
    kps = {}
    for iid, r, c, data in con.execute("SELECT image_id, rows, cols, data FROM keypoints"):
        if data and r:
            kps[iid] = np.frombuffer(data, np.float32).reshape(r, c)[:, :2].copy()

    rows = []
    for pid, r, c, data in con.execute("SELECT pair_id, rows, cols, data FROM two_view_geometries"):
        if not data or not r:
            continue
        i2 = pid % MAX_IMAGE_ID; i1 = (pid - i2) // MAX_IMAGE_ID
        n1, n2 = names.get(i1), names.get(i2)
        if n1 not in pose or n2 not in pose or i1 not in kps or i2 not in kps:
            continue
        our_cov = len(our_seen.get(n1, set()) & our_seen.get(n2, set()))
        c1, c2 = canon(n1), canon(n2)
        agi_cov = (len(agi_seen.get(c1, set()) & agi_seen.get(c2, set()))
                   if agi_seen is not None and c1 in agi_seen and c2 in agi_seen else -1)
        shared_marker = len(markers_by_name.get(n1, set()) & markers_by_name.get(n2, set()))
        m = np.frombuffer(data, np.uint32).reshape(r, c)[:, :2]
        ff = epi_false_frac(kps, i1, i2, m, K, Kinv, pose, n1, n2)
        rows.append((our_cov, agi_cov, shared_marker, r, ff))
    con.close()
    rows = np.array([x for x in rows if x[4] is not None], dtype=float)

    print(f"=== {args.field}/{args.plot}: overlap cross-check ({len(rows)} verified pairs) ===")
    print("Do the pairs WE call no-overlap (high false-match rate) also read no-overlap by "
          "Agisoft + markers?\n")
    print(f"  {'our bucket':<26} {'#pairs':>6} {'our_falseFrac':>13} {'Agisoft covis (med)':>20} "
          f"{'%pairs sharing a marker':>24}")
    for lo, hi, lbl in [(0, 1, "our covis=0 (no overlap)"), (1, 20, "our covis 1-19 (tiny)"),
                        (20, 1e9, "our covis 20+ (real overlap)")]:
        sel = rows[(rows[:, 0] >= lo) & (rows[:, 0] < hi)]
        if not len(sel):
            print(f"  {lbl:<26} 0"); continue
        agi = sel[sel[:, 1] >= 0][:, 1]
        agi_med = f"{np.median(agi):.0f}" if len(agi) else "n/a"
        mark_pct = 100 * np.mean(sel[:, 2] > 0)
        print(f"  {lbl:<26} {len(sel):>6} {np.median(sel[:,4]):>13.2f} {agi_med:>20} {mark_pct:>23.0f}%")

    # agreement between the two independent methods on 'no overlap'
    both = rows[rows[:, 1] >= 0]
    if len(both):
        our_no = both[:, 0] < 1
        agi_no = both[:, 1] < 1
        agree = np.mean(our_no == agi_no)
        print(f"\n  Independent agreement (our covis=0  vs  Agisoft covis=0): {agree*100:.0f}% "
              f"of {len(both)} pairs where Agisoft has both images")
        print(f"  Of pairs WE call no-overlap, Agisoft ALSO says no-overlap: "
              f"{np.mean(agi_no[our_no])*100:.0f}%")
    print("\n(Agisoft is a 2nd independent method, NOT ground truth. Marker-sharing needs no "
          "reconstruction at all.)")


if __name__ == "__main__":
    main()
