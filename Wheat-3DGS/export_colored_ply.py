"""
Export a colored PLY from step 4 segmentation results.
Bakes HSV colors per wheat head into gaussians_colored.ply for viewing in Supersplat.
Usage: python export_colored_ply.py --gaussians_ply <path> --labels_path <path> --output_ply <path> --sh_degree <int>
"""
import os
import sys
import argparse
import colorsys
import torch

# make sure scene/ is importable when called from run_wheat_3dgs.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene.gaussian_model import GaussianModel


def main():
    parser = argparse.ArgumentParser(description="Export colored PLY with per-head HSV colors")
    parser.add_argument("--gaussians_ply", required=True, help="Path to gaussians.ply from step 4")
    parser.add_argument("--labels_path",   required=True, help="Path to all_obj_labels.pth from step 4")
    parser.add_argument("--output_ply",    required=True, help="Output path for gaussians_colored.ply")
    parser.add_argument("--sh_degree", type=int, default=3,
                        help="SH degree used during training (must match, default 3)")
    args = parser.parse_args()

    SH_C0 = 0.28209479177387814  # zeroth-order SH coefficient

    print(f"Loading gaussians from {args.gaussians_ply}...")
    gs = GaussianModel(sh_degree=args.sh_degree)
    gs.load_ply(args.gaussians_ply)
    print(f"Loaded {len(gs.get_xyz)} Gaussians")

    print(f"Loading labels from {args.labels_path}...")
    all_obj_labels = torch.load(args.labels_path)
    n_heads = all_obj_labels.shape[0] - 1  # index 0 is background
    print(f"Baking colors for {n_heads} heads...")

    n_labeled = 0
    for head_id in range(1, n_heads + 1):
        mask = all_obj_labels[head_id].bool()
        # ensure mask is on the same device as the Gaussian features
        if not mask.is_cuda:
            mask = mask.cuda()
        count = mask.sum().item()
        if count == 0:
            continue
        n_labeled += count
        hue = (head_id - 1) / max(n_heads, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        # color must be on cuda — _features_dc is on cuda after load_ply
        color = torch.tensor(
            [(r - 0.5) / SH_C0, (g - 0.5) / SH_C0, (b - 0.5) / SH_C0],
            dtype=torch.float32, device="cuda"
        )
        # use .data to bypass autograd leaf-variable restriction
        gs._features_dc.data[mask] = color.view(1, 1, 3)
        if gs._features_rest is not None and gs._features_rest.shape[1] > 0:
            gs._features_rest.data[mask] = 0.0

    total = len(gs.get_xyz)
    print(f"Colored {n_labeled}/{total} Gaussians ({100 * n_labeled / total:.1f}%)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_ply)), exist_ok=True)
    print(f"Saving to {args.output_ply}...")
    gs.save_ply(args.output_ply)
    print(f"Done. Drag {args.output_ply} into supersplat.dev to view colored wheat heads.")


if __name__ == "__main__":
    main()
