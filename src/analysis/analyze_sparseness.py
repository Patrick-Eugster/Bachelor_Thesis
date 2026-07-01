"""
Diagnostic: measure how "sparse" each COLMAP/Agisoft reconstruction is.

3DGS densification methods behave differently depending on how well-constrained the
input is (see docs/reconstruction/DENSIFICATION_OPTIONS.md). "Sparse / limited views" is not just the
image count — it is mostly about multi-view OVERLAP (how many cameras see each 3D point)
and ANGULAR diversity (do the cameras look from many directions or all from one cone).
This script reads the sparse SfM model and reports those numbers so we can decide, per
dataset, whether MCMC (good on sparse) or AbsGrad (detail on denser data) is the better
fit. Read-only — it never modifies any input.

Run:
    python src/analysis/analyze_sparseness.py                 # auto: all FIP plots + phone sessions
    python src/analysis/analyze_sparseness.py --sparse <dir>  # one specific sparse/0 dir
Output:
    docs/analysis_results/sparseness.json   (raw numbers, git-tracked for the report)
    + a printed summary table
"""

import os
import sys
import glob
import json
import argparse
import numpy as np

# reuse the project's COLMAP reader for the image (extrinsics) side
from gaussians.scene.colmap_loader import read_extrinsics_text, qvec2rotmat

REPO = "/workspace"
OUT_JSON = os.path.join(REPO, "docs", "analysis_results", "sparseness.json")


def parse_track_lengths(points3D_txt):
    """Read points3D.txt and return (track_lengths, reproj_errors) as numpy arrays.

    Each non-comment line is:
        POINT3D_ID X Y Z R G B ERROR  (IMAGE_ID POINT2D_IDX)*
    so the track length = number of (IMAGE_ID, POINT2D_IDX) pairs = (len(elems) - 8) / 2.
    Track length = how many images observe this 3D point — the core sparseness signal.
    The text point reader in colmap_loader drops this, so we parse it ourselves."""
    tracks = []
    errors = []
    with open(points3D_txt, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            elems = line.split()
            errors.append(float(elems[7]))
            tracks.append((len(elems) - 8) // 2)
    return np.array(tracks), np.array(errors)


def camera_angular_spread(images):
    """Mean pairwise angle (degrees) between camera optical axes.

    Small angle = all cameras look from a similar direction (angularly LIMITED, e.g. FIP's
    overhead-only views); large angle = diverse viewpoints. A cheap proxy for 'limited views'
    that the raw image count cannot capture. Optical axis in world = R^T @ [0,0,1]."""
    dirs = []
    for img in images.values():
        R = qvec2rotmat(img.qvec)               # world -> camera rotation
        axis_world = R.T @ np.array([0.0, 0.0, 1.0])   # camera +Z back into world frame
        dirs.append(axis_world / np.linalg.norm(axis_world))
    dirs = np.array(dirs)
    if len(dirs) < 2:
        return float("nan")
    # mean of pairwise angles; clip dot product for numerical safety before arccos
    dots = np.clip(dirs @ dirs.T, -1.0, 1.0)
    iu = np.triu_indices(len(dirs), k=1)         # upper triangle = each pair once
    angles = np.degrees(np.arccos(dots[iu]))
    return float(np.mean(angles))


def analyze_one(sparse_dir, label):
    """Compute the sparseness profile for one sparse/0 directory.
    Returns a dict of metrics (or None if the model files are missing)."""
    images_txt = os.path.join(sparse_dir, "images.txt")
    points_txt = os.path.join(sparse_dir, "points3D.txt")
    if not (os.path.exists(images_txt) and os.path.exists(points_txt)):
        print(f"  [skip] {label}: missing images.txt or points3D.txt")
        return None

    images = read_extrinsics_text(images_txt)
    tracks, errors = parse_track_lengths(points_txt)

    n_images = len(images)
    n_points = len(tracks)
    # observations per image = how many of an image's 2D keypoints matched a 3D point
    obs_per_image = [int(np.sum(np.array(img.point3D_ids) >= 0)) for img in images.values()]

    metrics = {
        "label": label,
        "sparse_dir": sparse_dir,
        "num_registered_images": n_images,
        "num_points3D": n_points,
        "mean_track_length": float(np.mean(tracks)) if n_points else 0.0,
        "median_track_length": float(np.median(tracks)) if n_points else 0.0,
        # fraction of points seen by only 2 images = the weakest-triangulated (most fragile) points
        "frac_track_len_2": float(np.mean(tracks == 2)) if n_points else 0.0,
        "mean_obs_per_image": float(np.mean(obs_per_image)) if obs_per_image else 0.0,
        "mean_reproj_error_px": float(np.mean(errors)) if n_points else 0.0,
        "mean_pairwise_view_angle_deg": camera_angular_spread(images),
    }
    return metrics


def discover_targets():
    """Auto-find all FIP plots and phone sessions that have a sparse/0 model."""
    targets = []
    for d in sorted(glob.glob(os.path.join(REPO, "input_plots", "fip", "*", "sparse", "0"))):
        plot = d.split(os.sep)[-3]
        targets.append((d, f"fip/{plot}"))
    for d in sorted(glob.glob(os.path.join(REPO, "input_plots", "phone", "*", "*", "sparse", "0"))):
        parts = d.split(os.sep)
        targets.append((d, f"phone/{parts[-4]}/{parts[-3]}"))
    return targets


def main():
    parser = argparse.ArgumentParser(description="Measure SfM sparseness (overlap + angular diversity).")
    parser.add_argument("--sparse", action="append", default=None,
                        help="A specific sparse/0 dir (repeatable). If omitted, auto-discovers FIP + phone.")
    args = parser.parse_args()

    if args.sparse:
        targets = [(s, s) for s in args.sparse]
    else:
        targets = discover_targets()

    if not targets:
        print("No sparse models found.")
        sys.exit(1)

    results = []
    print(f"\nAnalyzing {len(targets)} reconstruction(s)...\n")
    for sparse_dir, label in targets:
        m = analyze_one(sparse_dir, label)
        if m:
            results.append(m)

    # ── printed table ──
    print("\n" + "=" * 110)
    print(f"{'dataset':<26}{'imgs':>5}{'pts3D':>9}{'trackL(mean/med)':>18}{'%len2':>7}{'obs/img':>9}{'reproj':>8}{'viewAng°':>10}")
    print("-" * 110)
    for m in results:
        print(f"{m['label']:<26}{m['num_registered_images']:>5}{m['num_points3D']:>9}"
              f"{m['mean_track_length']:>9.2f}/{m['median_track_length']:<8.1f}"
              f"{m['frac_track_len_2']*100:>6.0f}%{m['mean_obs_per_image']:>9.0f}"
              f"{m['mean_reproj_error_px']:>8.2f}{m['mean_pairwise_view_angle_deg']:>10.1f}")
    print("=" * 110)
    print("Higher mean track length + higher view angle + lower %len2 = DENSER / better constrained.")
    print("Lower mean track length + low view angle + high %len2     = SPARSER / limited views.\n")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote raw metrics -> {OUT_JSON}\n")


if __name__ == "__main__":
    main()
