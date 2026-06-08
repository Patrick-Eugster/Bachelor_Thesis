#
# Test A — render-engine comparison (gsplat vs diff-gaussian) on the SAME trained model.
#
# Both engines now place Gaussian *centers* identically (both honor cx/cy when
# use_principal_point trained the model), so the ONLY difference between their outputs is the
# residual scalar-tanfov approximation that diff-gaussian/flashsplat still use for the 2D
# covariance + tile-binning. This script measures that difference directly:
#   - RGB L1 + PSNR between the two renders (how much the image changes)
#   - alpha-mask IoU (alpha>0.5) — this is exactly the `pred_seg` the segmentation matcher uses,
#     so it answers "how much would a head's projected mask move between engines?"
#
# Run AFTER a model is trained. Reads the trained model via the standard ModelParams/Scene path.
#

import os
import json
import torch
from argparse import ArgumentParser
from tqdm import tqdm

from gaussians.scene import Scene
from gaussians.gaussian_renderer import render, render_diffgs, GaussianModel
from gaussians.utils.general_utils import safe_state
from gaussians.arguments import ModelParams, PipelineParams, get_combined_args


def mask_iou(a, b):
    """IoU between two boolean masks (same shape)."""
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / union if union > 0 else 1.0


def compare(dataset: ModelParams, iteration: int, pipeline: PipelineParams):
    """Render every test + train view with both engines and report per-view + mean differences."""
    # Make sure the "gsplat" side is really gsplat, even if WHEAT_RENDERER=diffgs is set
    # in the environment — render_diffgs() is always called explicitly for the diff-gaussian side.
    os.environ.pop("WHEAT_RENDERER", None)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg, dtype=torch.float32, device="cuda")

        results = {}
        for split, views in (("test", scene.getTestCameras()), ("train", scene.getTrainCameras())):
            per_view = []
            for view in tqdm(views, desc=f"Comparing {split}"):
                g = render(view, gaussians, pipeline, background)         # gsplat
                d = render_diffgs(view, gaussians, pipeline, background)  # diff-gaussian
                rgb_g, rgb_d = g["render"].clamp(0, 1), d["render"].clamp(0, 1)

                l1 = torch.abs(rgb_g - rgb_d).mean().item()
                mse = torch.mean((rgb_g - rgb_d) ** 2).item()
                psnr = float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()
                max_px = torch.abs(rgb_g - rgb_d).max().item()

                # alpha mask agreement — the segmentation-relevant number
                a_g = (g["alpha"].squeeze() > 0.5)
                a_d = (d["alpha"].squeeze() > 0.5)
                iou = mask_iou(a_g, a_d)
                disagree_px = (a_g ^ a_d).sum().item()

                per_view.append({"name": view.image_name, "rgb_l1": l1, "rgb_psnr": psnr,
                                 "rgb_max_px": max_px, "alpha_iou": iou, "alpha_disagree_px": disagree_px})

            n = len(per_view)
            if n == 0:
                continue
            mean = {k: sum(v[k] for v in per_view) / n for k in ("rgb_l1", "rgb_psnr", "rgb_max_px", "alpha_iou", "alpha_disagree_px")}
            results[split] = {"mean": mean, "per_view": per_view}

            print(f"\n=== {split} ({n} views) — gsplat vs diff-gaussian ===")
            print(f"  RGB  L1          : {mean['rgb_l1']:.6f}")
            print(f"  RGB  PSNR (g vs d): {mean['rgb_psnr']:.2f} dB   (higher = engines agree more)")
            print(f"  RGB  max pixel   : {mean['rgb_max_px']:.4f}")
            print(f"  alpha mask IoU   : {mean['alpha_iou']:.4f}   (1.0 = identical masks)")
            print(f"  alpha disagree px: {mean['alpha_disagree_px']:.1f} px/view")

        out = os.path.join(dataset.model_path, "renderer_compare.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Compare gsplat vs diff-gaussian renders on a trained model")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Comparing renderers on " + args.model_path)
    safe_state(args.quiet)
    compare(model.extract(args), args.iteration, pipeline.extract(args))
