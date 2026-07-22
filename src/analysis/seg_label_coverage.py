"""Quantify the '1/2 the field is uncolored' seg symptom, with FIP as a control. all_obj_labels.pth is a
(N_heads x N_gaussians) bool matrix; a Gaussian is 'colored' iff some head claims it. Report coverage,
#heads, and coverage split LEFT/RIGHT of the plot centre. If phone is half-uncovered but FIP (same
seg_cull_v3 method) is fully covered, the failure is phone-specific, not a seg/viewer code bug.

Run:  python src/analysis/seg_label_coverage.py
"""

import os
import numpy as np
import torch
from plyfile import PlyData

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RUNS = [
    ("PHONE field_A/20250715 (seg_cull_v3)",
     "results/reconstruction/phone/field_A/20250715/vanilla_3dgs/phone_sahi/segmentation_3d/seg_cull_v3"),
    ("FIP plot_461 (test_absgrad_v2/seg_cull_v3) [CONTROL]",
     "results/reconstruction/fip/plot_461/vanilla_3dgs/test_absgrad_v2/segmentation_3d/seg_cull_v3"),
]


def load_xyz(ply):
    v = PlyData.read(ply)["vertex"]
    return np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)


def report(name, seg_dir):
    lab_path = os.path.join(REPO, seg_dir, "all_obj_labels.pth")
    ply_path = os.path.join(REPO, seg_dir, "gaussians_colored.ply")
    if not os.path.exists(lab_path):
        print(f"\n{name}: MISSING {lab_path}")
        return
    lab = torch.load(lab_path, map_location="cpu")           # (H, G) bool
    H, G = lab.shape
    per_head = lab.sum(dim=1).numpy()                         # gaussians per head
    big = per_head > 0.01 * G                                 # 'background/oversized' heads (>1% of gaussians)
    real_rows = np.where((per_head > 0) & (~big))[0]          # the actual small wheat heads
    real_claimed = lab[torch.as_tensor(real_rows)].any(dim=0).numpy() if len(real_rows) else np.zeros(G, bool)
    print(f"\n{name}")
    print(f"  Gaussians: {G:,} | heads>0: {int((per_head>0).sum())} | background/oversized heads (>1%): {int(big.sum())} "
          f"(largest {100*per_head.max()/G:.0f}% of all gaussians)")
    print(f"  REAL (small) heads: {len(real_rows)} | covering {100*real_claimed.mean():.1f}% of gaussians")
    if os.path.exists(ply_path):
        xyz = load_xyz(ply_path)
        if xyz.shape[0] == G:
            spans = xyz.max(0) - xyz.min(0)
            ax = int(np.argmax(spans[:2]))
            mid = np.median(xyz[:, ax])
            for side, m in [("LEFT ", xyz[:, ax] < mid), ("RIGHT", xyz[:, ax] >= mid)]:
                rc = 100 * real_claimed[m].mean()
                nheads = len(np.unique(np.where(lab[:, torch.as_tensor(np.where(m)[0])].any(dim=1).numpy() & ~big)[0]))
                print(f"    {side} half (axis {ax}): REAL-head coverage {rc:5.1f}%  |  #real heads present ~{nheads}")
        else:
            print(f"  (ply {xyz.shape[0]:,} verts != {G:,} labels — spatial skipped)")


def main():
    for name, d in RUNS:
        report(name, d)
    print("\nReading: PHONE one half ~low + other ~high, while FIP both halves high => 'half uncolored' is"
          " PHONE-SPECIFIC (reconstruction / seg-matching), NOT a seg/viewer code bug.")


if __name__ == "__main__":
    main()
