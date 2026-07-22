"""(Q4) render the Agisoft-3DGS geometry of the SAME phone session top-down — if it's the same diffuse blob
as our COLMAP one, better SfM doesn't change it (capture-driven, not COLMAP). (Q3) heatmap of phone
real-head coverage over the plot to LOCATE any region that has Gaussians but no head labels ('wheat but
uncolored').  Saves PNGs to docs/analysis_results/.
"""

import os
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plyfile import PlyData

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "docs", "analysis_results")

PHONE_SEG = "results/reconstruction/phone/field_A/20250715/vanilla_3dgs/phone_sahi/segmentation_3d/seg_cull_v3"
AGI_GLOB = "results/reconstruction/phone/field_A/20250715/agisoft/vanilla_3dgs/agisoft_bench/point_cloud/iteration_*/point_cloud.ply"


def xyz_of(ply):
    v = PlyData.read(ply)["vertex"]
    return np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)


def main():
    # ---- Q4: Agisoft geometry top-down ----
    agi = sorted(glob.glob(os.path.join(REPO, AGI_GLOB)))
    if agi:
        xyz = xyz_of(agi[-1])
        a, b = np.argsort(xyz.max(0) - xyz.min(0))[::-1][:2]
        idx = np.random.default_rng(0).choice(xyz.shape[0], min(300000, xyz.shape[0]), replace=False)
        plt.figure(figsize=(10, 8)); plt.scatter(xyz[idx, a], xyz[idx, b], c="0.4", s=0.4, linewidths=0)
        plt.gca().set_aspect("equal"); plt.title(f"AGISOFT-SfM 3DGS geometry (top-down) — {xyz.shape[0]:,} gaussians")
        plt.savefig(os.path.join(OUT, "seggeom_agisoft_fA0715.png"), dpi=110, bbox_inches="tight"); plt.close()
        print(f"Q4: agisoft geometry -> seggeom_agisoft_fA0715.png ({xyz.shape[0]:,} gaussians)")
    else:
        print("Q4: agisoft_bench point_cloud not found")

    # ---- Q3: phone real-head coverage heatmap ----
    lab = torch.load(os.path.join(REPO, PHONE_SEG, "all_obj_labels.pth"), map_location="cpu")   # (H,G)
    G = lab.shape[1]
    per_head = lab.sum(dim=1).numpy()
    real_rows = np.where((per_head > 0) & (per_head <= 0.01 * G))[0]
    real_claimed = lab[torch.as_tensor(real_rows)].any(dim=0).numpy()
    xyz = xyz_of(os.path.join(REPO, PHONE_SEG, "gaussians_colored.ply"))
    a, b = np.argsort(xyz.max(0) - xyz.min(0))[::-1][:2]
    A, B = xyz[:, a], xyz[:, b]
    nb = 40
    ae = np.linspace(A.min(), A.max(), nb + 1); be = np.linspace(B.min(), B.max(), nb + 1)
    dens, _, _ = np.histogram2d(A, B, bins=[ae, be])
    cov, _, _ = np.histogram2d(A, B, bins=[ae, be], weights=real_claimed.astype(float))
    with np.errstate(invalid="ignore"):
        frac = np.where(dens > 0, cov / dens, np.nan)
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    im0 = ax[0].imshow(dens.T, origin="lower", extent=[A.min(), A.max(), B.min(), B.max()], aspect="equal", cmap="viridis")
    ax[0].set_title("phone: gaussian DENSITY (where is there stuff)"); fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(frac.T, origin="lower", extent=[A.min(), A.max(), B.min(), B.max()], aspect="equal", cmap="RdYlGn", vmin=0, vmax=0.5)
    ax[1].set_title("phone: REAL-HEAD coverage (red=has gaussians but NO heads)"); fig.colorbar(im1, ax=ax[1])
    plt.savefig(os.path.join(OUT, "segcoverage_heatmap_phone.png"), dpi=110, bbox_inches="tight"); plt.close()
    print("Q3: phone coverage heatmap -> segcoverage_heatmap_phone.png")


if __name__ == "__main__":
    main()
