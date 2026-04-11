#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import glob
import torch
from scene import Scene
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from os import makedirs
from concurrent.futures import ThreadPoolExecutor
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer import flashsplat_render
from utils.image_helper import *

# CHANGED: eval_obj_labels commented out — step 4 (run_3d_seg.py) already saves 2DSeg/*.pt label maps
# per camera, so we load those directly instead of re-running 300 FlashSplat renders per camera.
# Kept here for reference in case the pre-saved maps are unavailable.
# def eval_obj_labels(all_obj_labels, viewpoint_cam, gaussians, pipe, background):
#     """Project 3D head labels to 2D — all ops on GPU, only final pred_mask moved to CPU."""
#     from gaussian_renderer import flashsplat_render
#     render_num = all_obj_labels.size(0)
#     pred_mask = None
#     max_alpha = None
#     min_depth = None
#     for obj_idx in range(render_num):
#         obj_used_mask = (all_obj_labels[obj_idx]).bool()
#         if obj_used_mask.sum().item() == 0 or obj_idx == 0:  # obj 0 is background
#             continue
#         flashsplat_pkg = flashsplat_render(viewpoint_cam, gaussians, pipe, background, used_mask=obj_used_mask.to("cuda"))
#         # keep alpha/depth on GPU — avoids sync stall between renders
#         render_alpha = flashsplat_pkg["alpha"].detach()
#         render_depth = flashsplat_pkg["depth"].detach()
#         if pred_mask is None:
#             pred_mask = torch.zeros_like(render_alpha)
#             max_alpha = torch.zeros_like(render_alpha)
#             min_depth = torch.ones_like(render_alpha)
#         _pix_mask = (render_alpha > 0.5)
#         pix_mask = _pix_mask.clone()
#         overlap_mask = (_pix_mask & (pred_mask > 0))
#         if overlap_mask.sum().item() > 0:
#             if (min_depth[overlap_mask].mean() < render_depth[overlap_mask].mean()):
#                 pix_mask[_pix_mask] = (~(pred_mask[_pix_mask] > 0))
#         pred_mask[pix_mask] = obj_idx
#         min_depth[pix_mask] = render_depth[pix_mask]
#         max_alpha[pix_mask] = render_alpha[pix_mask]
#     if pred_mask is not None:
#         pred_mask = pred_mask.cpu()
#     return pred_mask

# CHANGED: render_set now takes twod_seg_dir instead of all_obj_labels.
# Loads pre-saved 2DSeg/*.pt label maps from step 4 instead of calling eval_obj_labels.
# Old signature: render_set(model_path, name, views, gaussians, pipeline, background, all_obj_labels)
def render_set(model_path, name, views, gaussians, pipeline, background, twod_seg_dir):
    """Render overlay + binary segmentation for all views. Loads pre-saved 2DSeg labels from step 4."""
    render_path = os.path.join(model_path, name, "overlay")
    seg_path = os.path.join(model_path, name, "segmentation")
    makedirs(render_path, exist_ok=True)
    makedirs(seg_path, exist_ok=True)

    def save_outputs(rgb_image, binary_array, image_name):
        """Save overlay PNG and binary segmentation PNG — runs in background thread."""
        torchvision.utils.save_image(rgb_image, os.path.join(render_path, f"{image_name}.png"))
        image = Image.fromarray(binary_array.numpy(), mode="L")
        image.save(os.path.join(seg_path, f"{image_name}.png"))

    with ThreadPoolExecutor(max_workers=2) as executor:
        save_future = None
        for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
            rendering = render(view, gaussians, pipeline, background)["render"].detach().cpu()

            # CHANGED: load pre-saved 2D label map from step 4 instead of calling eval_obj_labels.
            # Was: pred_seg = eval_obj_labels(all_obj_labels, view, gaussians, pipeline, background)
            # Now: instant torch.load — avoids 300 FlashSplat renders per camera
            seg_pt = os.path.join(twod_seg_dir, f"{view.image_name}.pt")
            pred_seg = torch.load(seg_pt).unsqueeze(0)  # (H,W) -> (1,H,W) to match eval_obj_labels shape

            binary_array = (pred_seg.squeeze() != 0).to(torch.uint8) * 255
            rgb_mask = visualize_obj(pred_seg) / 255.0
            rgb_image = overlay_image(rendering, rgb_mask)

            # wait for previous save before submitting new one (backpressure)
            if save_future is not None:
                save_future.result()
            save_future = executor.submit(save_outputs, rgb_image, binary_array, view.image_name)
        if save_future is not None:
            save_future.result()

def render_sets(dataset: ModelParams, pipeline: PipelineParams, exp_name, skip_train, load_counts):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Load fine-tuned gaussians.ply from step 4 — needed for correct RGB overlay render.
        # Step 4 fine-tunes Gaussian positions so iteration_15000 checkpoint won't match.
        ply_dir = os.path.join(dataset.model_path, "wheat-head", exp_name, "ply")
        scene_ply = os.path.join(dataset.model_path, "wheat-head", exp_name, "gaussians.ply")
        twod_seg_dir = os.path.join(dataset.model_path, "wheat-head", exp_name, "2DSeg")

        if os.path.exists(scene_ply):
            print(f"Loading fine-tuned scene model: {scene_ply}")
            gaussians.load_ply(scene_ply)
            print(f"Fine-tuned model: {len(gaussians.get_xyz)} Gaussians")
        else:
            print("WARNING: gaussians.ply not found — using iteration_15000 model (RGB render may look different)")

        # CHANGED: replaced PLY-based all_obj_labels building + eval_obj_labels with direct 2DSeg load.
        # Old approach: load gaussians.ply, build pos_to_idx dict, match per-head PLY positions,
        # build all_obj_labels tensor, then call eval_obj_labels (300 FlashSplat renders per camera).
        # New approach: step 4 already saved these label maps to 2DSeg/*.pt — just load them.
        # Old block kept below for reference:
        # if os.path.exists(scene_ply) and os.path.exists(ply_dir):
        #     fine_xyz_np = gaussians.get_xyz.detach().cpu().numpy()
        #     pos_to_idx = {fine_xyz_np[i].tobytes(): i for i in range(len(fine_xyz_np))}
        #     head_plys = sorted(glob.glob(os.path.join(ply_dir, "wh_*.ply")))
        #     def _head_id(path):
        #         return int(os.path.splitext(os.path.basename(path))[0].split('_')[1])
        #     max_head_id = max(_head_id(p) for p in head_plys)
        #     all_obj_labels = torch.zeros(max_head_id + 1, len(fine_xyz_np), dtype=torch.bool)
        #     head_gs = GaussianModel(dataset.sh_degree)
        #     n_plys = len(head_plys)
        #     for ply_idx, ply_file in enumerate(head_plys):
        #         if ply_idx % 50 == 0:
        #             print(f"Building labels from PLYs: {ply_idx}/{n_plys}")
        #         head_id = _head_id(ply_file)
        #         head_gs.load_ply(ply_file)
        #         head_xyz_np = head_gs.get_xyz.detach().cpu().numpy()
        #         for i in range(len(head_xyz_np)):
        #             idx = pos_to_idx.get(head_xyz_np[i].tobytes(), -1)
        #             if idx >= 0:
        #                 all_obj_labels[head_id, idx] = True
        #     print(f"Building labels from PLYs: {n_plys}/{n_plys} — done")
        #     n_labeled = all_obj_labels[1:].any(dim=0).sum().item()
        #     print(f"Gaussians labeled: {n_labeled}/{len(fine_xyz_np)}")
        # else:
        #     print("WARNING: gaussians.ply or ply/ not found — cannot build labels. Aborting.")
        #     return

        if not os.path.exists(twod_seg_dir):
            print(f"WARNING: 2DSeg/ not found at {twod_seg_dir}. Run step 4 first.")
            return

        n_seg_files = len([f for f in os.listdir(twod_seg_dir) if f.endswith(".pt")])
        print(f"Found {n_seg_files} pre-saved 2DSeg label maps in {twod_seg_dir}")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.getTrainCameras(), gaussians, pipeline, background, twod_seg_dir)
        render_set(dataset.model_path, "test", scene.getTestCameras(), gaussians, pipeline, background, twod_seg_dir)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--exp_name", type=str, help="Exp name")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--load_counts", action="store_true")  # kept for CLI compatibility, no longer used
    args = get_combined_args(parser)
    print(f"Rendering {args.model_path}/{args.exp_name}")
    render_sets(model.extract(args), pipeline.extract(args), args.exp_name, args.skip_train, args.load_counts)
