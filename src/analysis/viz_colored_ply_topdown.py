"""Reproduce the '1/2 uncolored, 1/2 messy' symptom directly from the baked colors of gaussians_colored.ply
(what the viewer/360 actually shows). Top-down scatter colored by the ply's baked RGB, phone vs FIP.
Saves PNGs so we can SEE whether half is uncolored and whether it's the colors (seg/export) not the labels.

Run:  python src/analysis/viz_colored_ply_topdown.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plyfile import PlyData

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "docs", "analysis_results")
os.makedirs(OUT, exist_ok=True)

RUNS = [
    ("phone_fA0715", "results/reconstruction/phone/field_A/20250715/vanilla_3dgs/phone_sahi/segmentation_3d/seg_cull_v3/gaussians_colored.ply"),
    ("fip_p461", "results/reconstruction/fip/plot_461/vanilla_3dgs/test_absgrad_v2/segmentation_3d/seg_cull_v3/gaussians_colored.ply"),
]


def load(ply):
    v = PlyData.read(ply)["vertex"]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    names = v.data.dtype.names
    if "red" in names:
        rgb = np.stack([np.asarray(v["red"]), np.asarray(v["green"]), np.asarray(v["blue"])], axis=1) / 255.0
    else:                                   # SH DC coeffs -> RGB
        C0 = 0.28209479177387814
        rgb = 0.5 + C0 * np.stack([np.asarray(v["f_dc_0"]), np.asarray(v["f_dc_1"]), np.asarray(v["f_dc_2"])], axis=1)
        rgb = np.clip(rgb, 0, 1)
    return xyz, rgb


def main():
    for tag, rel in RUNS:
        xyz, rgb = load(os.path.join(REPO, rel))
        spans = xyz.max(0) - xyz.min(0)
        a, b = np.argsort(spans)[::-1][:2]           # two widest axes = top-down plane
        n = xyz.shape[0]
        idx = np.random.default_rng(0).choice(n, size=min(300000, n), replace=False)
        plt.figure(figsize=(10, 8))
        plt.scatter(xyz[idx, a], xyz[idx, b], c=rgb[idx], s=0.4, linewidths=0)
        plt.gca().set_aspect("equal"); plt.title(f"{tag} — baked seg colors (top-down, axes {a},{b})")
        p = os.path.join(OUT, f"segcolor_{tag}.png")
        plt.savefig(p, dpi=110, bbox_inches="tight"); plt.close()
        # also a plain grey render (geometry only) to see if the RECONSTRUCTION itself is half-missing
        plt.figure(figsize=(10, 8))
        plt.scatter(xyz[idx, a], xyz[idx, b], c="0.4", s=0.4, linewidths=0)
        plt.gca().set_aspect("equal"); plt.title(f"{tag} — geometry only (top-down)")
        pg = os.path.join(OUT, f"seggeom_{tag}.png")
        plt.savefig(pg, dpi=110, bbox_inches="tight"); plt.close()
        print(f"{tag}: {n:,} gaussians -> {p}  +  {pg}")


if __name__ == "__main__":
    main()
