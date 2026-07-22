"""Does each phone capture actually CLOSE THE LOOP? i.e. do the first and last photos of the sweep
overlap? Tested with signals that do NOT depend on our final reconstruction's poses (so the answer isn't
circular):

  (1) RAW geometrically-verified matches between the first-K and last-K frames, read straight from
      distorted/database.db (two_view_geometries). These are per-pair RANSAC-verified correspondences —
      computed from the images, not from the global bundle-adjusted poses.
  (2) SHARED DECODED MARKERS: if a first-frame and a last-frame both decode the SAME coded marker
      (from logs/marker_triangulation.json, read from pixels), they PROVABLY overlap — zero reconstruction.

Reference = median verified matches between time-adjacent frames (what "strong overlap" looks like).
A closed loop => end matches and/or a shared end marker. Also prints the pose-based first<->last distance
as a SECONDARY (reconstruction-derived, so circular) cross-check, clearly labelled.

Usage:  python src/analysis/check_loop_closure.py            # all sessions
        python src/analysis/check_loop_closure.py --field field_D --plot 20250706
"""

import os
import re
import glob
import json
import sqlite3
import argparse

import numpy as np
import pycolmap as pc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAX_IMAGE_ID = 2147483647
TARGET_TO_CODE = {1: 113, 2: 105, 3: 89, 4: 101, 5: 85, 6: 77}
K = 5   # how many frames at each end count as "first"/"last"


def img_time(n):
    m = re.search(r"_(\d{8})_(\d{2})(\d{2})(\d{2})", n)
    return int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4)) if m else None


def verified_matches(db_path, names_wanted):
    """{(nameA,nameB): n_verified_matches} from two_view_geometries, only for wanted image names."""
    con = sqlite3.connect(db_path)
    id2name = {i: n for i, n in con.execute("SELECT image_id, name FROM images")}
    out = {}
    for pid, rows, cols, data in con.execute("SELECT pair_id, rows, cols, data FROM two_view_geometries"):
        n = rows if data else 0
        i2 = pid % MAX_IMAGE_ID; i1 = (pid - i2) // MAX_IMAGE_ID
        a, b = id2name.get(i1), id2name.get(i2)
        if a in names_wanted and b in names_wanted:
            out[frozenset((a, b))] = n
    con.close()
    return out


def process(field, plot):
    sess = os.path.join(REPO, "input_plots", "phone", field, plot)
    db = os.path.join(sess, "distorted", "database.db")
    model = os.path.join(sess, "sparse", "0")
    if not (os.path.exists(db) and os.path.isdir(model)):
        return None
    rec = pc.Reconstruction(model)
    named = [(im.name, img_time(im.name)) for im in rec.images.values()]
    if any(t is None for _, t in named):
        return {"field": field, "plot": plot, "skip": "unparsable timestamps"}
    order = [n for n, _ in sorted(named, key=lambda x: x[1])]
    first, last = order[:K], order[-K:]

    vm = verified_matches(db, set(order))
    # end overlap (first x last) vs adjacent (reference)
    end = [vm.get(frozenset((a, b)), 0) for a in first for b in last if a != b]
    adj = [vm.get(frozenset((order[i], order[i + 1])), 0) for i in range(len(order) - 1)]

    # shared decoded markers at the ends (fully independent)
    markers = {}
    tri = os.path.join(sess, "logs", "marker_triangulation.json")
    if os.path.exists(tri):
        t = json.load(open(tri))
        for code in TARGET_TO_CODE.values():
            for o in t.get(str(code), []):
                if o.get("src") == "detected":
                    markers.setdefault(o["cam"], set()).add(code)
    fm = set().union(*[markers.get(n, set()) for n in first]) if first else set()
    lm = set().union(*[markers.get(n, set()) for n in last]) if last else set()
    shared_markers = fm & lm

    # SECONDARY (circular) pose-based first<->last distance
    cen = {}
    for im in rec.images.values():
        T = im.cam_from_world(); cen[im.name] = -T.rotation.matrix().T @ np.array(T.translation)
    C = np.array([cen[n] for n in order])
    extent = np.linalg.norm(C.max(0) - C.min(0))
    dfl = np.linalg.norm(cen[order[0]] - cen[order[-1]])

    return {"field": field, "plot": plot, "n": len(order),
            "end_matches_max": int(max(end)) if end else 0,
            "end_matches_med": int(np.median(end)) if end else 0,
            "adj_matches_med": int(np.median(adj)) if adj else 0,
            "shared_end_markers": sorted(shared_markers),
            "posebased_first_last_pct": round(100 * dfl / extent, 1) if extent else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default=None)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()
    if args.field and args.plot:
        sessions = [(args.field, args.plot)]
    else:
        sessions = []
        for fld in ("field_A", "field_D"):
            for p in sorted(glob.glob(os.path.join(REPO, "input_plots", "phone", fld, "2025*"))):
                sessions.append((fld, os.path.basename(p)))

    print("LOOP CLOSURE per session — do first & last frames overlap? (non-circular: raw matches + markers)\n")
    print(f"  {'session':<28} {'endMatch max/med':>16} {'adjMatch med':>12} {'shared end markers':>20} "
          f"{'pose1<->last':>12}")
    rows = []
    for field, plot in sessions:
        r = process(field, plot)
        if r is None:
            continue
        if r.get("skip"):
            print(f"  {field+'/'+plot:<28} skipped ({r['skip']})"); continue
        closed = "CLOSED" if (r["end_matches_med"] > 0 or r["shared_end_markers"]) else "open?"
        mk = ",".join(map(str, r["shared_end_markers"])) or "-"
        print(f"  {field+'/'+plot:<28} {str(r['end_matches_max'])+'/'+str(r['end_matches_med']):>16} "
              f"{r['adj_matches_med']:>12} {mk:>20} {str(r['posebased_first_last_pct'])+'%':>12}   {closed}")
        rows.append(r)
    print("\n  endMatch = verified matches between first-5 and last-5 frames (0 => ends don't overlap).")
    print("  shared end markers = a coded marker decoded in BOTH a first and a last frame => provable overlap.")
    print("  pose1<->last% = (circular, reconstruction-derived) first-last camera distance as % of trajectory extent; <25% ~ loop.")


if __name__ == "__main__":
    main()
