"""Diagnostic: score how well a COLMAP SfM run used + linked the input images.

Read-only benchmark for comparing feature/matching front-ends (SIFT vs ALIKED vs SuperPoint+LightGlue
etc.) on the SAME images. Prints three blocks for one run:

  A. COVERAGE  — did the photos get used? (from the sparse model)
       sub-models found, images in the largest, total registered across all fragments.
       Fragmentation (many sub-models) is the phone-wheat failure mode — one connected model is the goal.

  B. CONNECTIVITY — how well are the images linked in PAIRS? (from database.db's two_view_geometries)
       verified image pairs, match-graph density, median inliers/pair, strong pairs (>= --strong-inliers),
       and the connected-components of the match graph — the single best predictor of whether SfM will
       fragment (if the strong-pair graph is one piece, SfM *can* build one model).

  C. QUALITY — how good is the model it built? (from the sparse model)
       #3D points, mean track length (images per point), mean reprojection error, mean observations/image.

Database is OPTIONAL (Agisoft's reference model has no COLMAP database → only A + C are shown for it).

Usage:
    python src/analysis/analyze_sfm_connectivity.py --model <sparse_dir> [--database <db>] \
        [--label NAME] [--strong-inliers 30]
    # our SIFT run (4 sub-models) + its database:
    python src/analysis/analyze_sfm_connectivity.py \
        --model input_plots/phone/field_A/20250603/distorted/sparse \
        --database input_plots/phone/field_A/20250603/distorted/database.db --label SIFT
    # Agisoft gold reference (single model, no database):
    python src/analysis/analyze_sfm_connectivity.py \
        --model input_plots/phone/field_A/20250603/agisoft/sparse/0 --label Agisoft
"""

import argparse
import os
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gaussians", "scene"))
from colmap_loader import (read_extrinsics_binary, read_extrinsics_text,  # noqa: E402
                           read_points3D_binary, read_points3D_text)

MAX_IMAGE_ID = 2147483647  # COLMAP's pair_id base (pair_id = id1*MAX + id2, id1 < id2)


def find_submodels(model_dir):
    """Return [(name, path)] of COLMAP models under model_dir. If it has numbered sub-dirs
    (0,1,2,... each with an images.bin/.txt) those are the sub-models; otherwise model_dir itself
    is one model. Sorted biggest-first is done by the caller after loading."""
    subs = []
    for name in sorted(os.listdir(model_dir)):
        p = os.path.join(model_dir, name)
        if os.path.isdir(p) and name.isdigit() and _has_model(p):
            subs.append((name, p))
    if subs:
        return subs
    if _has_model(model_dir):
        return [(os.path.basename(model_dir.rstrip("/")), model_dir)]
    raise SystemExit(f"no COLMAP model (images.bin/.txt) found in {model_dir}")


def _has_model(path):
    """True if a folder holds a COLMAP model in either binary or text form."""
    return (os.path.isfile(os.path.join(path, "images.bin"))
            or os.path.isfile(os.path.join(path, "images.txt")))


def load_model(path):
    """Read one COLMAP model (bin preferred, else txt) → (images_dict, xyzs, errors)."""
    if os.path.isfile(os.path.join(path, "images.bin")):
        images = read_extrinsics_binary(os.path.join(path, "images.bin"))
    else:
        images = read_extrinsics_text(os.path.join(path, "images.txt"))
    pts_bin = os.path.join(path, "points3D.bin")
    if os.path.isfile(pts_bin):
        xyzs, _rgbs, errors = read_points3D_binary(pts_bin)
    elif os.path.isfile(os.path.join(path, "points3D.txt")):
        xyzs, _rgbs, errors = read_points3D_text(os.path.join(path, "points3D.txt"))
    else:
        xyzs, errors = np.empty((0, 3)), np.empty((0, 1))
    return images, xyzs, errors


def coverage_block(model_dir, n_input):
    """Print block A and return (sorted [(name, n_images, path)], largest_path)."""
    subs = find_submodels(model_dir)
    rows = []
    for name, p in subs:
        images, _, _ = load_model(p)
        rows.append((name, len(images), p))
    rows.sort(key=lambda r: r[1], reverse=True)   # biggest first
    largest = rows[0]
    total = sum(r[1] for r in rows)

    print("\n  A. COVERAGE (did the photos get used?)")
    print(f"     sub-models found        : {len(rows)}   "
          f"{'← fragmented!' if len(rows) > 1 else '← one connected model'}")
    for name, n, _ in rows:
        bar = "#" * max(1, round(40 * n / max(1, largest[1])))
        print(f"        model {name:<3} {n:>4} imgs  {bar}")
    denom = f" / {n_input}" if n_input else ""
    print(f"     largest model           : {largest[1]}{denom} images")
    print(f"     total across fragments  : {total}{denom} images")
    return rows, largest[2]


def quality_block(model_path):
    """Print block C for one model (the largest / only one)."""
    images, xyzs, errors = load_model(model_path)
    n_pts = len(xyzs)
    # observations = valid (non -1) 2D-3D links summed over images
    obs = sum(int(np.count_nonzero(im.point3D_ids != -1)) for im in images.values())
    mean_track = obs / n_pts if n_pts else 0.0
    mean_obs_img = obs / len(images) if images else 0.0
    mean_err = float(np.mean(errors)) if n_pts else float("nan")

    print("\n  C. QUALITY (how good is the built model?)")
    print(f"     3D points               : {n_pts}")
    print(f"     mean track length       : {mean_track:.2f} images / point   (higher = better linked)")
    print(f"     mean observations/image : {mean_obs_img:.0f}")
    print(f"     mean reprojection error : {mean_err:.3f} px")


def _decode_pair(pair_id):
    """COLMAP pair_id → (image_id1, image_id2)."""
    id2 = pair_id % MAX_IMAGE_ID
    id1 = (pair_id - id2) // MAX_IMAGE_ID
    return id1, id2


def _largest_component(nodes, edges):
    """Union-find: size of the biggest connected component over `nodes` given `edges` (id pairs)."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    sizes = {}
    for n in nodes:
        r = find(n)
        sizes[r] = sizes.get(r, 0) + 1
    return (max(sizes.values()) if sizes else 0), len(sizes)


def connectivity_block(db_path, strong):
    """Print block B from the match database (two_view_geometries = geometrically-verified pairs)."""
    db = sqlite3.connect(db_path)
    img_ids = [r[0] for r in db.execute("SELECT image_id FROM images")]
    n = len(img_ids)
    possible = n * (n - 1) // 2

    inliers, edges, strong_edges = [], [], []
    for pair_id, rows in db.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0"):
        a, b = _decode_pair(pair_id)
        inliers.append(rows)
        edges.append((a, b))
        if rows >= strong:
            strong_edges.append((a, b))
    db.close()

    inl = np.array(inliers) if inliers else np.array([0])
    big_all, _ = _largest_component(img_ids, edges)
    big_strong, n_comp_strong = _largest_component(img_ids, strong_edges)

    print("\n  B. CONNECTIVITY (how well are images linked in pairs?)")
    print(f"     images in database      : {n}")
    print(f"     verified pairs          : {len(edges)} / {possible} possible   "
          f"(density {100*len(edges)/max(1,possible):.1f}%)")
    print(f"     median inliers / pair   : {np.median(inl):.0f}   (mean {inl.mean():.0f}, max {inl.max()})")
    print(f"     strong pairs (>= {strong})    : {len(strong_edges)}")
    print(f"     match graph (all pairs) : largest connected piece = {big_all} / {n} images")
    print(f"     match graph (strong)    : largest piece = {big_strong} / {n}  "
          f"({n_comp_strong} components)  ← predicts max single-model size")


def main():
    ap = argparse.ArgumentParser(description="Score COLMAP SfM coverage + connectivity + quality.")
    ap.add_argument("--model", required=True, help="sparse dir (may hold sub-models 0,1,... or be one model)")
    ap.add_argument("--database", default=None, help="optional database.db for the connectivity block")
    ap.add_argument("--label", default="run", help="name for the header")
    ap.add_argument("--n-input", type=int, default=0, help="total input images (for the x/total denominator)")
    ap.add_argument("--strong-inliers", type=int, default=30,
                    help="inlier threshold for a 'strong' (registration-capable) pair")
    args = ap.parse_args()

    print("=" * 72)
    print(f"  SfM CONNECTIVITY REPORT — {args.label}")
    print(f"  model: {args.model}")
    if args.database:
        print(f"  database: {args.database}")
    print("=" * 72)

    _rows, largest_path = coverage_block(args.model, args.n_input)
    if args.database and os.path.isfile(args.database):
        connectivity_block(args.database, args.strong_inliers)
    elif args.database:
        print(f"\n  B. CONNECTIVITY — skipped (database not found: {args.database})")
    else:
        print("\n  B. CONNECTIVITY — skipped (no --database given, e.g. Agisoft reference)")
    quality_block(largest_path)
    print("=" * 72)


if __name__ == "__main__":
    main()
