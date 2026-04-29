
import gc
import glob
import math
import os
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from os import makedirs

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from gaussians.arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussians.gaussian_renderer import GaussianModel, flashsplat_render, render
from gaussians.scene import Scene
from gaussians.scene.cameras import MiniCam
from gaussians.utils.general_utils import safe_state
from gaussians.utils.graphics_utils import getProjectionMatrix, getWorld2View2
from gaussians.utils.wheatgs_helper import render_360, render_360_fast

#### Begin of 360-degree camera trajectory copied from gsgen ####
# These two functions are adapted from the implementation in:
# "GSGEN: Text-to-3D using Gaussian Splatting"
# Original Code: https://github.com/gsgen3d/gsgen

def get_c2w_from_up_and_look_at(up, look_at, pos):
    up = up / np.linalg.norm(up)
    z = look_at - pos
    z = z / np.linalg.norm(z)
    y = -up
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.zeros([3, 4], dtype=np.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = pos
    return c2w

def get_camera_path_fixed_elevation(n_frames, n_circles=1, camera_distance=2, cam_center=[0, 0, 0], elevation=0):
    azimuth = np.linspace(0, 2 * np.pi * n_circles, n_frames)
    elevation_rad = np.deg2rad(elevation)
    x = camera_distance * np.cos(azimuth) * np.cos(elevation_rad)
    y = camera_distance * np.sin(azimuth) * np.cos(elevation_rad)
    z = camera_distance * np.sin(elevation_rad) * np.ones_like(x)

    up = np.array([0, 0, 1], dtype=np.float32)
    look_at = np.array(cam_center, dtype=np.float32)
    pos = np.stack([x, y, z], axis=1)

    c2ws = []
    for i in range(n_frames):
        c2ws.append(get_c2w_from_up_and_look_at(up, look_at, pos[i]))
    c2ws = np.stack(c2ws, axis=0)
    return c2ws

#### End of 360-degree camera trajectory copied from gsgen ####

def opt_w_masks(viewpoint_cam, gaussians, pipe, background, obj_masks, obj_num=None):
    if obj_num is None: # if None then it's the first view
        obj_num = torch.unique(obj_masks).numel() - 1
    obj_masks = obj_masks.to(torch.float32).to("cuda")
    render_pkg = flashsplat_render(viewpoint_cam, gaussians, pipe, background, gt_mask=obj_masks.squeeze(), obj_num=obj_num)
    # print(render_pkg["render"].shape)  # debug leftover — render shape never changes
    used_count = render_pkg["used_count"].detach().cpu()
    return used_count, obj_num

def render_wheat_head(dataset : ModelParams, pipeline : PipelineParams, exp_name, 
                      n_frames=200, framerate=20, elevation=15, save_frames=False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)
        bg_color = [1,1,1] # if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        og_views = scene.getTrainCameras()
        og_view = og_views[0]
        width, height = math.floor(og_view.image_width / 3), math.floor(og_view.image_height / 3)
        fovy, fovx = og_view.FoVy / 5, og_view.FoVx / 5
        znear, zfar = og_view.znear, og_view.zfar
        print(f"Fixed parameters width: {width} height {height} fovy {fovy} fovx {fovx}")

        wheat_head_dir = os.path.join(dataset.model_path, "wheat-head", exp_name, "ply")
        # wheat_head_folders = [
        #     name for name in os.listdir(wheat_head_dir)
        #     if os.path.isdir(os.path.join(wheat_head_dir, name)) and name.isdigit()
        # ]
        # wheat_head_folders = sorted(wheat_head_folders)
        ply_files = [f for f in os.listdir(wheat_head_dir) if f.startswith("wh") and f.endswith(".ply")]
        print("ply_files", len(ply_files), ply_files)

        # for idx, wheat_head in enumerate(tqdm(wheat_head_folders, desc="Rendering progress")):
        for idx, ply_file in enumerate(tqdm(ply_files, desc="Rendering progress")):
            if len(os.path.splitext(ply_file)[0].split("_")) > 2:
                print(f"Pass file {ply_file}")
                continue

            scene.load_ply(os.path.join(wheat_head_dir, ply_file))
            gs_centroid = torch.mean(gaussians.get_xyz, dim=0).cpu().tolist()
            scene_radius = scene.cameras_extent
            print(f"Gaussians centroid {gs_centroid}, Scene radius {scene_radius}")
            
            ply_id = ply_file.replace("wh_", "", 1).replace(".ply", "", 1)
            camera_distance = scene_radius * 0.65
            render_path = os.path.join(os.path.dirname(wheat_head_dir), "wheat_head_360", ply_id)
            makedirs(render_path, exist_ok=True)

            c2ws = get_camera_path_fixed_elevation(n_frames=n_frames, n_circles=1, camera_distance=camera_distance, cam_center=gs_centroid, elevation=elevation)
            for idx, c2w in enumerate(c2ws):
                c2w = np.vstack([c2w, [0.0, 0.0, 0.0, 1.0]])
                w2c = np.linalg.inv(np.float64(c2w))
                world_view_transform = torch.tensor(w2c.astype(np.float32)).transpose(0, 1).cuda()
                projection_matrix = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0,1).cuda()
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                view = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
                render_pkg = render(view, gaussians, pipeline, background)
                rendering = render_pkg["render"]
                torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            output_video = os.path.join(os.path.dirname(render_path), f"{ply_id}.mp4")
            framerate = 10
            subprocess.run([
                "ffmpeg",
                "-loglevel", "error",
                "-framerate", str(framerate),
                "-start_number", "0",
                "-i", "%05d.png",  
                "-vf", r"scale=iw-mod(iw\,2):ih-mod(ih\,2)",
                "-r", str(framerate),
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                output_video
            ], cwd=render_path)
            if not save_frames:
                shutil.rmtree(render_path)

def render_wheat_field(dataset : ModelParams, pipeline : PipelineParams, exp_name,
                       n_frames=100, framerate=10, elevation=45, save_frames=False, load_iteration=-1, fast_render=False):
    seg2d_dir = os.path.join(dataset.model_path, "wheat-head", exp_name, "2DSeg")
    out_dir = os.path.join(dataset.model_path, "wheat-head", exp_name, "3DSeg")
    os.makedirs(out_dir, exist_ok=True)
    gaussians = GaussianModel(dataset.sh_degree)
    try:
        load_iteration = int(load_iteration)
    except:
        pass
    print(f"Load iteration {load_iteration}, Resolution {dataset.resolution}")
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)
    # gaussians.training_setup(opt)
    print(f"Loaded point cloud size: {len(gaussians.get_xyz)}")
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_stack_eval = scene.getTestCameras().copy()
    # viewpoint_stack += viewpoint_stack_eval
    print(f"Length of viewpoint stack: {len(viewpoint_stack)}")

    # Build all_obj_labels from the per-head PLY files saved by run_3d_seg.py (step 4).
    # The whole-scene FlashSplat accumulation gives useless labels because background
    # contributions dominate wheat head contributions for every Gaussian in the scene.
    # The per-head PLY files contain 700-3000 Gaussians per head from step 4's targeted
    # optimization. Their positions match gaussians.ply (the fine-tuned full scene model
    # saved at the end of step 4) with 100% exact match — but NOT iteration_15000 (original
    # model), because step 4 fine-tuning moves Gaussian positions.
    ply_dir = os.path.join(dataset.model_path, "wheat-head", exp_name, "ply")
    scene_ply = os.path.join(dataset.model_path, "wheat-head", exp_name, "gaussians.ply")
    if os.path.exists(scene_ply) and os.path.exists(ply_dir):
        # reload gaussians from fine-tuned model so positions match per-head PLYs
        print(f"Loading fine-tuned scene model from step 4 ({scene_ply})...")
        gaussians.load_ply(scene_ply)
        print(f"Fine-tuned model: {len(gaussians.get_xyz)} Gaussians")

        # build position → index lookup for O(1) matching
        fine_xyz_np = gaussians.get_xyz.detach().cpu().numpy()
        pos_to_idx = {fine_xyz_np[i].tobytes(): i for i in range(len(fine_xyz_np))}

        # collect all per-head PLY files and find max head ID
        head_plys = sorted(glob.glob(os.path.join(ply_dir, "wh_*.ply")))
        def _head_id(path):
            """Parse head ID from wh_0001.ply or wh_0001_a.ply → 1"""
            return int(os.path.splitext(os.path.basename(path))[0].split('_')[1])
        max_head_id = max(_head_id(p) for p in head_plys)

        all_obj_labels = torch.zeros(max_head_id + 1, len(fine_xyz_np), dtype=torch.bool)
        head_gs = GaussianModel(dataset.sh_degree)
        for ply_file in head_plys:
            head_id = _head_id(ply_file)
            head_gs.load_ply(ply_file)
            head_xyz_np = head_gs.get_xyz.detach().cpu().numpy()
            for i in range(len(head_xyz_np)):
                idx = pos_to_idx.get(head_xyz_np[i].tobytes(), -1)
                if idx >= 0:
                    all_obj_labels[head_id, idx] = True

        n_labeled = all_obj_labels[1:].any(dim=0).sum().item()
        print(f"Gaussians labeled from {len(head_plys)} PLY files: {n_labeled}/{len(fine_xyz_np)} ({100*n_labeled/len(fine_xyz_np):.1f}%)")
    else:
        print("WARNING: gaussians.ply or ply/ not found — cannot build labels. Aborting.")
        return

    labels_mb = all_obj_labels.numel() * all_obj_labels.element_size() / 1e6
    print(f"Saving all_obj_labels ({labels_mb:.0f} MB)...")
    torch.save(all_obj_labels.detach().cpu(), os.path.join(dataset.model_path, "wheat-head", exp_name, "all_obj_labels.pth"))
    print("Starting render loop...")
    render_fn = render_360_fast if fast_render else render_360
    output_video = render_fn(viewpoint_stack[0], scene.cameras_extent, out_dir, n_frames, framerate, gaussians, pipeline, background,
                             all_obj_labels=all_obj_labels, all_counts=None, elevation=elevation)
    if not save_frames: # if not specified then remove the saved frames for generating video
        shutil.rmtree(out_dir)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="iteration of OG 3DGS to load")
    parser.add_argument("--render_type", type=str, default=None, help="render type: field (whole wheat field) or head (all individual wheat heads)")
    parser.add_argument("--exp_name", type=str, default=None, help="experiment name of 3D segmentation to load")
    parser.add_argument("--n_frames", type=int, default=100, help="number of frames to render")
    parser.add_argument("--framerate", type=int, default=10, help="framerate of the rendered video")
    parser.add_argument("--elevation", type=int, default=45, help="elevation angle for the camera trajectory rotating around the scene")
    parser.add_argument("--save_frames", action="store_true", help="If specified, save frames in addition to output video")
    parser.add_argument("--fast_render", action="store_true", help="Use single colored render per frame instead of N_heads flashsplat renders (~N_heads× faster)")
    args = get_combined_args(parser)
    print(f"Rendering {args.model_path} for 3D segmentation experiment {args.exp_name}, Option: {args.render_type}")
    if args.render_type == "field":
        print("Render the 3D segmentation on the whole wheat field")
        render_wheat_field(model.extract(args), pipeline.extract(args), args.exp_name, args.n_frames, args.framerate, args.elevation, args.save_frames, fast_render=args.fast_render)
    elif args.render_type == "head":
        print("Render each individual segmented wheat head")
        render_wheat_head(model.extract(args), pipeline.extract(args), args.exp_name, args.n_frames, args.framerate, args.elevation, args.save_frames)
    else:
        raise ValueError(f"Invalid render type: either field or head")