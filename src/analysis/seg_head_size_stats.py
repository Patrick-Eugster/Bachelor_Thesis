"""Per-head Gaussian-count distribution for a 3D-seg run — a structural diagnostic for over/under-
segmentation.

Each head-ID in all_obj_labels.pth (shape (n_heads+1, G) bool; row 0 = background) owns a set of
Gaussians; the row-sum = how many Gaussians that head has. A normal single wheat head is a small tight
clump; a head whose Gaussian count is many times the median is almost certainly SEVERAL physical heads
fused into one ID (under-segmentation / over-merge), while a head with a handful of Gaussians is a
fragment. So the shape of this distribution — median, the heavy upper tail, and the fragment floor — is
a proxy for segmentation cleanliness that needs no ground truth.

Also reports a robust physical size per head: the median distance of the head's Gaussians to their own
centroid (outlier-insensitive, unlike a bbox diagonal that one stray Gaussian inflates).

Give one or more seg dirs (each with gaussians.ply + all_obj_labels.pth). Writes a markdown comparison
table + per-run JSON + a head-size histogram PNG. Read-only on the seg outputs.

Usage:
  python src/analysis/seg_head_size_stats.py \
    --seg NAME=path/to/segmentation_3d/EXP [--seg NAME2=...] \
    --out_dir docs/analysis_results/seg_head_sizes
"""
import argparse
import json
import os

import numpy as np
import torch
from plyfile import PlyData


def _load(seg_dir):
    """Return (per_head_counts (n_heads,), xyz (G,3), labels (n_heads+1,G) bool) for one seg dir."""
    L = torch.load(os.path.join(seg_dir, "all_obj_labels.pth"), map_location="cpu")
    counts = L[1:].sum(dim=1).numpy()                       # Gaussians per head (skip background row 0)
    ply = PlyData.read(os.path.join(seg_dir, "gaussians.ply"))["vertex"]
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype(np.float64)
    return counts, xyz, L


def _robust_radius(mask_row, xyz):
    """Median distance of a head's Gaussians to their centroid (outlier-robust physical size, model units)."""
    idx = mask_row.numpy() if torch.is_tensor(mask_row) else mask_row
    p = xyz[idx]
    if len(p) < 2:
        return 0.0
    return float(np.median(np.linalg.norm(p - p.mean(0), axis=1)))


def _stats(name, seg_dir, tiny_thresh, big_thresh):
    """Compute the distribution stats + double-assignment check + robust sizes for one seg run."""
    counts, xyz, L = _load(seg_dir)
    nonzero = counts[counts > 0]
    med = float(np.median(nonzero)) if len(nonzero) else 0.0
    # robust radius for the heads (all of them — cheap)
    radii = np.array([_robust_radius(L[h], xyz) for h in range(1, L.shape[0]) if counts[h - 1] > 0])
    multi = int((L[1:].sum(dim=0) > 1).sum())              # Gaussians claimed by >1 head (should be 0)
    return {
        "name": name, "seg_dir": seg_dir,
        "n_created": int(len(counts)),                 # total head IDs (= wheat_heads_found), incl empties
        "n_heads": int((counts > 0).sum()),            # IDs that actually own >=1 Gaussian
        "n_empty": int((counts == 0).sum()),           # IDs created in matching then abandoned (all-False row)
        "gaussians_assigned": int(counts.sum()),
        "gaussians_multi_claimed": multi,
        "count_min": int(nonzero.min()) if len(nonzero) else 0,
        "count_median": med,
        "count_mean": float(nonzero.mean()) if len(nonzero) else 0.0,
        "count_p90": float(np.percentile(nonzero, 90)) if len(nonzero) else 0.0,
        "count_p99": float(np.percentile(nonzero, 99)) if len(nonzero) else 0.0,
        "count_max": int(nonzero.max()) if len(nonzero) else 0,
        "max_over_median": (float(nonzero.max()) / med) if med else 0.0,
        "n_tiny": int((nonzero < tiny_thresh).sum()), "tiny_thresh": tiny_thresh,
        "n_big": int((nonzero > big_thresh).sum()), "big_thresh": big_thresh,
        "radius_median_u": float(np.median(radii)) if len(radii) else 0.0,
        "radius_p99_u": float(np.percentile(radii, 99)) if len(radii) else 0.0,
        "_counts": nonzero,   # kept for the histogram, dropped before JSON
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", action="append", required=True, metavar="NAME=DIR",
                    help="a seg run as NAME=path/to/segmentation_3d/EXP (repeatable)")
    ap.add_argument("--tiny_thresh", type=int, default=20, help="heads with < this many Gaussians = fragments")
    ap.add_argument("--big_thresh", type=int, default=800, help="heads with > this many Gaussians = likely merged")
    ap.add_argument("--out_dir", default="docs/analysis_results/seg_head_sizes")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    runs = []
    for spec in a.seg:
        name, path = spec.split("=", 1)
        print(f"analysing {name}: {path}")
        runs.append(_stats(name, path, a.tiny_thresh, a.big_thresh))

    # histogram (log-y) of head sizes, all runs overlaid
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        bins = np.logspace(0, np.log10(max(r["count_max"] for r in runs) + 1), 40)
        for r in runs:
            plt.hist(r["_counts"], bins=bins, histtype="step", linewidth=1.6, label=r["name"])
        plt.axvline(a.big_thresh, color="crimson", ls="--", lw=1, label=f"merge flag (>{a.big_thresh})")
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("Gaussians per head (log)"); plt.ylabel("number of heads (log)")
        plt.title("Per-head Gaussian-count distribution"); plt.legend(); plt.tight_layout()
        hist_path = os.path.join(a.out_dir, "head_size_hist.png")
        plt.savefig(hist_path, dpi=130); plt.close()
        print(f"wrote {hist_path}")
    except Exception as e:
        print(f"(histogram skipped: {e})")

    # markdown table
    for r in runs:
        r.pop("_counts", None)
    cols = [("n_heads", "heads"), ("n_empty", "empty"), ("count_median", "median g/head"), ("count_mean", "mean"),
            ("count_p90", "p90"), ("count_p99", "p99"), ("count_max", "max"),
            ("max_over_median", "max/median"), ("n_big", f">{a.big_thresh}g (merged?)"),
            ("n_tiny", f"<{a.tiny_thresh}g (frag)"), ("gaussians_multi_claimed", "multi-claimed"),
            ("radius_median_u", "robust radius (u)")]
    lines = ["# Per-head Gaussian-count distribution (over/under-seg diagnostic)", "",
             "Row-sum of each head's mask in all_obj_labels.pth = Gaussians per head. A heavy upper tail "
             "(heads far above the median) indicates over-merging (several physical heads fused into one "
             "ID); a large fragment floor (tiny heads) indicates over-splitting. `multi-claimed` should be "
             "0 (no Gaussian in two heads). `robust radius` = median distance of a head's Gaussians to "
             "their centroid (model units), outlier-insensitive.", "",
             "| run | " + " | ".join(h for _, h in cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for r in runs:
        def fmt(k):
            v = r[k]
            return f"{v:.2f}" if isinstance(v, float) else str(v)
        lines.append(f"| {r['name']} | " + " | ".join(fmt(k) for k, _ in cols) + " |")
    md = os.path.join(a.out_dir, "head_size_stats.md")
    open(md, "w").write("\n".join(lines) + "\n")
    json.dump(runs, open(os.path.join(a.out_dir, "head_size_stats.json"), "w"), indent=1)
    print("\n".join(lines))
    print(f"\nwrote {md} + head_size_stats.json")


if __name__ == "__main__":
    main()
