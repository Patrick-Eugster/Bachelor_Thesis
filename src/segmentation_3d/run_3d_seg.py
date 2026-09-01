import os
import sys
import csv
import glob
import json
import math
import time
import hashlib
import random
import shutil
import string
import resource
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from itertools import islice
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


from gaussians.arguments import ModelParams, PipelineParams, OptimizationParams
from gaussians.gaussian_renderer import flashsplat_render
from gaussians.scene import Scene, GaussianModel, Camera
from gaussians.utils.wheatgs_utils import (
    PILtoTorch,
    binarize_mask,
    get_bbox_from_mask,
    get_bbox_from_mask_gpu,
    is_overlapping,
    calculate_seg_iou,
    calculate_seg_iou_gpu,
    build_mask_crop,
    calculate_seg_iou_gpu_crop,
    vis_image_w_overlay
)
from segmentation_3d.seg_roi import build_roi_keep_mask, default_ground_filter

# Optional render-vs-match timing (set WHEAT_SEG_TIMING=1). Off by default so the
# cuda.synchronize() calls it needs don't slow normal runs.
_SEG_TIMING = os.environ.get("WHEAT_SEG_TIMING") == "1"
# render/match = inside find_match; lift_decode/lift_render = opt_label_w_seg (the FULL-model lift);
# ilp = the FlashSplat multi_instance_opt solve; commit_render = the per-head 2DSeg GPU render.
# The remaining keys split the previously-"untimed" CPU/IO work so we can see where the ~2h/phone goes:
#   setup        = one-time model+scene+camera load + crop-cache build/load + initial 2DSeg write
#   commit_paint = CPU side of the commit loop: alpha->CPU->threshold->paint the H×W 2DSeg label map
#   ply_prep     = per-head full-model CPU copy + prune (feeds the async per-head PLY save)
#   overlay_wait = blocking on the async overlay-JPG saves (only for the first vis_max_heads heads)
#   seg2d_save   = final 2DSeg/*.pt save at the end
_SEG_TIMER = {"render": 0.0, "match": 0.0, "lift_decode": 0.0, "lift_render": 0.0, "ilp": 0.0,
              "commit_render": 0.0, "setup": 0.0, "commit_paint": 0.0, "ply_prep": 0.0,
              "overlay_wait": 0.0, "seg2d_save": 0.0}

# Fast commit paint: paint only the head's 2D bounding box into the 2DSeg label map instead of scanning
# and writing the full H×W frame per (head, camera). Bit-identical to the old full-frame paint because
# outside the head's bbox the rendered alpha never exceeds 0.5, so no pixel there would ever be painted.
# On by default; WHEAT_SEG_NO_FAST_PAINT=1 forces the old full-frame path (for the lossless A/B md5 check).
_FAST_PAINT = os.environ.get("WHEAT_SEG_NO_FAST_PAINT") != "1"
# On by default; WHEAT_SEG_NO_INFERENCE=1 keeps the gradient machinery (for an inference-only A/B).
_INFERENCE = os.environ.get("WHEAT_SEG_NO_INFERENCE") != "1"

def find_new_mask_dir(overlap_counter, num_wheat_head):
    """Return next letter suffix (a, b, c…) for an overlapping head, tracked in memory."""
    count = overlap_counter.get(num_wheat_head, 0)
    letter_suffix = string.ascii_lowercase[count]
    overlap_counter[num_wheat_head] = count + 1
    return letter_suffix

########### Begin of Find & Match helper methods ###########

# This function is adapted from the implementation in:
# "FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally" by Shen et. al.
# Paper: https://arxiv.org/abs/2409.08270
# Original Code: https://github.com/florinshen/FlashSplat
def multi_instance_opt(all_contrib, gamma=0.):
    """
    Input:
    all_contrib: A_{e} with shape (obj_num, gs_num) 
    gamma: softening factor range from [-1, 1]
    
    Output: 
    all_obj_labels: results S with shape (obj_num, gs_num)
    where S_{i,j} denotes j-th gaussian belong i-th object
    """
    all_contrib_sum = all_contrib.sum(dim=0)
    all_obj_labels = torch.zeros_like(all_contrib).bool()
    for obj_idx, obj_contrib in enumerate(all_contrib):  # tqdm removed: obj_num=1 always, bar showed exactly 1 step
        obj_contrib = torch.stack([all_contrib_sum - obj_contrib, obj_contrib], dim=0)
        obj_contrib = F.normalize(obj_contrib, dim=0)
        obj_contrib[0, :] += gamma
        obj_label = torch.argmax(obj_contrib, dim=0)
        all_obj_labels[obj_idx] = obj_label
    return all_obj_labels

def opt_label_w_seg(gaussians : GaussianModel,
                    viewpoint_stack : List[Camera],
                    mask_paths : List[str],
                    pipeline, background, pts_filter=None, mask_cache=None):
    """
    Helper function that wraps Gaussians label optimization schema into one function
    return:
        all_counts: counts that are additive
        all_obj_labels:
    mask_cache: optional {path -> crop entry} so the FULL-frame gt_mask can be rebuilt from the cached
    crop instead of re-decoding the PNG (lossless: a mask is 0 outside its bbox). Falls back to PNG
    decode on a cache miss / when the cache is off.
    """
    assert len(viewpoint_stack) == len(mask_paths)

    all_counts = None
    for idx, viewpoint_cam in enumerate(viewpoint_stack):
        if _SEG_TIMING:
            _t = time.perf_counter()
        entry = mask_cache.get(mask_paths[idx], "MISS") if mask_cache else "MISS"
        if entry == "MISS":
            # fallback: decode the PNG (cache off, or this mask wasn't cached) — original behaviour
            with Image.open(mask_paths[idx]) as temp:
                gt_mask = binarize_mask(PILtoTorch(temp.copy(), viewpoint_cam.resolution)).squeeze().to("cuda")
        else:
            # rebuild the full-frame gt_mask from the cached tight-bbox crop (no PNG decode).
            # Byte-identical to the decode path (verified): the mask is 0 outside its bbox.
            W, H = viewpoint_cam.resolution                      # cam.resolution is (W,H); gt_mask is (H,W)
            gt_mask = torch.zeros((H, W), dtype=torch.float32, device="cuda")
            if entry is not None:                                # None = empty mask -> all zeros
                y0, y1, x0, x1, crop, _area = entry
                gt_mask[y0:y1, x0:x1] = crop.to("cuda", dtype=torch.float32)
        assert viewpoint_cam.original_image.shape[-2:] == gt_mask.shape
        if _SEG_TIMING:
            torch.cuda.synchronize()  # wait for the gt_mask H2D copy so it's counted in decode, not lost
            _SEG_TIMER["lift_decode"] += time.perf_counter() - _t
            _t = time.perf_counter()
        with torch.no_grad():
            # inference=True: the lift renders the FULL model (no used_mask) but never backprops, so skip
            # the screen-space grad machinery — pure speedup on the heaviest render, byte-identical counts.
            render_pkg = flashsplat_render(viewpoint_cam, gaussians, pipeline, background, gt_mask=gt_mask, obj_num=1, inference=_INFERENCE)
            rendering = render_pkg["render"]
            used_count = render_pkg["used_count"]
            if all_counts is None:
                all_counts = torch.zeros_like(used_count)
            all_counts += used_count
        if _SEG_TIMING:
            torch.cuda.synchronize(); _SEG_TIMER["lift_render"] += time.perf_counter() - _t
    # only flush VRAM cache when actually under memory pressure (>75% reserved)
    if torch.cuda.memory_reserved(0) / torch.cuda.get_device_properties(0).total_memory > 0.75:
        torch.cuda.empty_cache()

    # Filter points that are below threshold
    if pts_filter is not None:
        cols_to_modify = pts_filter.nonzero(as_tuple=True)[0]
        all_counts[1:, cols_to_modify] = 0
    return all_counts

def counts_to_obj_labels(all_counts, slackness=0.0):
    """
    Input: additive all_counts
    Output: all_obj_labels
    """
    if _SEG_TIMING:
        torch.cuda.synchronize(); _t = time.perf_counter()
    all_obj_labels = multi_instance_opt(all_counts, slackness)
    if _SEG_TIMING:
        torch.cuda.synchronize(); _SEG_TIMER["ilp"] += time.perf_counter() - _t
    # print(f"{torch.sum(all_obj_labels, dim=1)[1]} Gaussians identified")  # too verbose: fires for every mask + every fine-tune iteration
    return all_obj_labels

# frustum-cull safety factor: the rasterizer bounds each Gaussian's splat at ~3 sigma, so we treat
# its on-screen footprint as (sigma * max_scale). Tunable per-run via env if a dataset ever needs more
# headroom, but the per-Gaussian test below is already conservative at 3.
_CULL_SIGMA = float(os.environ.get("WHEAT_SEG_CULL_SIGMA", "3.0"))

def cull_cameras(cams, gaussians, obj_used_mask):
    """Return only the cameras the head can actually render into. PER-GAUSSIAN test (no single-sphere
    approximation): project EVERY head Gaussian's centre at ITS OWN depth and keep a camera if ANY
    Gaussian's splat (centre +/- sigma*max_scale, in NDC) overlaps the image. A Gaussian's true
    anisotropic 3-sigma footprint is contained in that isotropic bound, and each point uses its own
    depth, so the keep-set is a conservative superset of what flashsplat actually draws -> bit-identical
    match/paint on ANY view geometry. This replaces the old one-sphere-one-depth test, whose first-order
    NDC-radius estimate under-projected on oblique wide-baseline views and was slightly lossy on phone.
    Vectorised over (cams x Gaussians) -> a single GPU->CPU sync."""
    if len(cams) == 0:
        return cams
    pts = gaussians.get_xyz[obj_used_mask]                            # (G,3) this head's centres
    scale = gaussians.get_scaling[obj_used_mask].max(dim=1).values    # (G,) isotropic upper bound per Gaussian
    device = pts.device
    G = pts.shape[0]
    ph = torch.cat([pts, torch.ones(G, 1, device=device)], dim=1)    # (G,4) homogeneous
    V = torch.stack([c.world_view_transform for c in cams])          # (N,4,4) world->camera
    P = torch.stack([c.full_proj_transform for c in cams])           # (N,4,4) world->clip
    tanx = torch.tensor([math.tan(c.FoVx * 0.5) for c in cams], device=device).view(-1, 1)  # (N,1)
    tany = torch.tensor([math.tan(c.FoVy * 0.5) for c in cams], device=device).view(-1, 1)
    # row-vector convention (as in flashsplat_render): p_out[n,g] = ph[g] @ M[n]
    p_cam = torch.einsum('gj,njk->ngk', ph, V)                       # (N,G,4)
    z = p_cam[:, :, 2]                                               # (N,G) depth per Gaussian per cam
    p_clip = torch.einsum('gj,njk->ngk', ph, P)                     # (N,G,4)
    w = p_clip[:, :, 3]
    ndc_x = p_clip[:, :, 0] / w
    ndc_y = p_clip[:, :, 1] / w
    z_safe = z.clamp(min=1e-6)                                       # small +z -> huge radius -> keeps (safe)
    r_x = (_CULL_SIGMA * scale).unsqueeze(0) / (z_safe * tanx)       # (N,G) screen radius in NDC
    r_y = (_CULL_SIGMA * scale).unsqueeze(0) / (z_safe * tany)
    in_front = z > 0                                                 # behind the camera -> can't render
    in_img = in_front & (ndc_x - r_x <= 1) & (ndc_x + r_x >= -1) \
                      & (ndc_y - r_y <= 1) & (ndc_y + r_y >= -1)     # (N,G) this Gaussian touches the image
    keep = in_img.any(dim=1).cpu().tolist()                         # camera kept if ANY Gaussian is in
    return [c for c, k in zip(cams, keep) if k]

def find_match(target_viewpoint_stack, gs_params, obj_used_mask, iou_threshold, dir_name, bbox_cache, mask_cache, verbose=False):
    """
    Input:
        target_viewpoint_stack: a list of viewpoints to iterate
        gs_params: gaussians, pipe, background
        obj_used_mask: pre-optimized flashsplat results
        mask_cache: dict mask_path -> tight-bbox crop entry (or None for empty), built once
                    at startup so we never re-decode a mask PNG from disk here
    Output:
    """
    gaussians, pipe, background = gs_params
    new_viewpoint_stack = []
    match_mask_paths = []
    sum_max_iou = 0.0
    # print(f"Length of target vpt stack to be matched: {len(target_viewpoint_stack)}")
    for viewpoint_cam in target_viewpoint_stack:
        if _SEG_TIMING:
            torch.cuda.synchronize(); _t = time.perf_counter()
        with torch.no_grad():
            # Go through other cameras to find match
            render_pkg = flashsplat_render(viewpoint_cam, gaussians, pipe, background, used_mask=obj_used_mask, inference=_INFERENCE)
            render_alpha = render_pkg["alpha"]
            pred_seg = render_alpha.squeeze() > 0.5  # stays on GPU — avoids GPU→CPU sync per render
        pred_bbox = get_bbox_from_mask_gpu(pred_seg)  # GPU torch ops, returns same (x,y,x,y) tuple
        pred_area = pred_seg.sum()  # |P| for IoU — computed ONCE per view, reused by every candidate
        if _SEG_TIMING:
            torch.cuda.synchronize(); _SEG_TIMER["render"] += time.perf_counter() - _t; _t = time.perf_counter()
        # Load YOLO bounding boxes from pre-loaded RAM cache (avoids disk read per camera)
        bboxes = bbox_cache[viewpoint_cam.image_name]
        # Overlap boxes xyxy, id and mIOU
        overlap_bboxes = [tuple(box.tolist()) for box in bboxes if is_overlapping(pred_bbox, tuple(box.tolist()))]
        overlap_idx = [i for i, box in enumerate(bboxes) if is_overlapping(pred_bbox, tuple(box.tolist()))]
        # Infer SAM-generated segmentation from bounding boxes
        # overlap_masks_paths = [mask_path for mask_path in viewpoint_cam.mask_paths if int(os.path.basename(mask_path)[-7:-4]) in overlap_idx]
        overlap_masks_paths = [os.path.join(dir_name, f"{viewpoint_cam.image_name}_{str(i).zfill(3)}.png") for i in overlap_idx]
        for p in overlap_masks_paths:
            assert p in viewpoint_cam.mask_paths, f"{p} not found in current image's masks"

        # Find the bbox/seg pair with largest Segmentation IOU between the rendering.
        # Compare against the cached tight-bbox crops instead of re-decoding each PNG.
        max_iou = 0.0
        max_overlap_mask_path = None
        for mask_path in overlap_masks_paths:
            entry = mask_cache.get(mask_path, "MISS")
            if entry == "MISS":
                # safety fallback: path not cached -> decode on the fly (original behaviour)
                with Image.open(mask_path) as temp:
                    mask = binarize_mask(PILtoTorch(temp.copy(), viewpoint_cam.resolution)).squeeze().to(pred_seg.device) > 0
                    assert mask.shape == pred_seg.shape
                iou = calculate_seg_iou_gpu(mask, pred_seg)
            elif entry is None:
                continue  # empty mask -> IoU 0, can never beat max_iou (strict >)
            else:
                iou = calculate_seg_iou_gpu_crop(pred_seg, pred_area, entry)
            if iou > max_iou:
                max_iou = iou
                max_overlap_mask_path = mask_path
        if _SEG_TIMING:
            torch.cuda.synchronize(); _SEG_TIMER["match"] += time.perf_counter() - _t
                                    
        if max_iou > iou_threshold: # Hyperparameters to modify
            # Add matched viewpoint cam and matched seg to a list
            new_viewpoint_stack.append(viewpoint_cam)
            match_mask_paths.append(max_overlap_mask_path)
            sum_max_iou += max_iou
            match_mask_name = os.path.splitext(os.path.basename(max_overlap_mask_path))[0]
            # processed_masks.add(match_mask_name) # Don't add matched to processed here!
            # print(f"find a mathch with IOU={max_iou} with seg {match_mask_name}") 
    
    assert len(new_viewpoint_stack) == len(match_mask_paths)
    if verbose:
        tqdm.write(f"  Matched {len(new_viewpoint_stack)} / {len(target_viewpoint_stack)} cameras" +
            (f", mean IoU {sum_max_iou / len(new_viewpoint_stack):.3f} > {iou_threshold}"
            if len(new_viewpoint_stack) > 0 else " (no matches)"))
    return new_viewpoint_stack, match_mask_paths

def update_processed_masks(processed_masks, new_mask_paths):
    for new_mask_path in new_mask_paths:
        new_mask_name = os.path.splitext(os.path.basename(new_mask_path))[0]
        processed_masks.add(new_mask_name)
    return processed_masks

########### End of Find & Match helper methods ###########
        
def training(dataset, opt, pipe, load_iteration, exp_name, iou_threshold, save_vis_overlay, vis_max_heads, wandb_enabled=False, use_mask_cache=True, seg_seed=0, frustum_cull=False, roi_cull=False, height_band=False, marker_exclude=False, roi_buffer_m=0.25, marker_radius_m=0.075, marker_radius_rel=0.0, legacy_ground_cull=False, ground_percentile=10.0):
    _t_setup = time.perf_counter()  # one-time load + cache build, recorded just before the main loop
    # All 3DSeg results will be saved under 3dgs_model_path/segmentation_3d/(exp_name)
    out_dir = os.path.join(dataset.model_path, "segmentation_3d", exp_name)
    sub_dirs = ["ply", "img", "count"]
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(out_dir, sub_dir), exist_ok=True)
    ply_dir, img_dir, count_dir = [os.path.join(out_dir, sub_dir) for sub_dir in sub_dirs]

    with open(f"{out_dir}/experiment.txt", "w") as file:
        file.write(f"exp_name {exp_name}\niou_threshold {iou_threshold}\n")
    
    results = open(os.path.join(out_dir, 'results.csv'), mode='w', newline='')
    writer = csv.writer(results)
    writer.writerow(["id", "init_mask", "num_matches", "num_GS"])
    
    gaussians = GaussianModel(dataset.sh_degree)
    try:
        load_iteration = int(load_iteration)
    except:
        pass
    print(f"Load iteration {load_iteration}, Resolution {dataset.resolution}")
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)
    gaussians.training_setup(opt)
    print(f"Loaded point cloud size: {len(gaussians.get_xyz)}")

    # pts_filter marks Gaussians to ZERO OUT (exclude from being labelled a head), consumed in
    # opt_label_w_seg.
    #
    # DEFAULT GROUND CULL — gentle, tilt-correct, KEEPS HEADS WHOLE. The original z < z_mean culled the
    # lower HALF of the scene by height, which only works when heads are the topmost layer (FIP overhead).
    # In phone capture the heads sit at MID-height, so a mean cut (world-z on Agisoft, or on the tilted
    # COLMAP frame) slices through every head -> partial heads -> poor cross-view matching -> the seg
    # crawls. Instead we fit the marker plane and cull only the bottom `ground_percentile` % by height
    # (a thin below-ground slice), scale-free so it works on both the arbitrary-scale COLMAP frame and the
    # metric Agisoft frame. Applied whenever the plot has markers; falls back to legacy z < z_mean only
    # for FIP (no markers, heads ARE topmost) or when --legacy_ground_cull is set.
    z_mean = torch.mean(gaussians.get_xyz.cpu()[:, 2])
    if legacy_ground_cull:
        pts_filter = (gaussians.get_xyz.cpu()[:, 2] < z_mean)
    else:
        _gf = default_ground_filter(gaussians.get_xyz, dataset.source_path, ground_percentile=ground_percentile)
        pts_filter = _gf if _gf is not None else (gaussians.get_xyz.cpu()[:, 2] < z_mean)
    # print(f"All Gaussians z_min: {torch.min(gaussians.get_xyz.cpu()[:, 2])} zmax: {torch.max(gaussians.get_xyz.cpu()[:, 2])}")  # debug detail
    #
    # OPTIONAL ROI FILTERS (roi_cull / height_band / marker_exclude) — when any is on, replace the ground
    # cull above with the marker-hull plot ROI so the background behind the plot can't form one big blob
    # and the marker plates aren't counted as heads. Also re-applied in the fine-tune loop (_roi_active).
    _roi_active = False  # True only once a ROI keep-mask is actually built (drives the fine-tune re-cull below)
    if roi_cull or height_band or marker_exclude:
        keep = build_roi_keep_mask(gaussians.get_xyz, dataset.source_path,
                                   roi_cull=roi_cull, height_band=height_band, marker_exclude=marker_exclude,
                                   roi_buffer_m=roi_buffer_m, marker_radius_m=marker_radius_m,
                                   marker_radius_rel=marker_radius_rel)
        if keep is not None:
            pts_filter = ~keep  # cull = everything NOT kept by the ROI
            _roi_active = True
        else:
            print("ROI flags set but markers unavailable -> keeping the ground cull above")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_stack_eval = scene.getTestCameras().copy()

    print(f"Length of viewpoint stack: {len(viewpoint_stack)}")
    print(f"[RAM] after scene+cameras loaded: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB peak")
    _overlay_executor = ThreadPoolExecutor(max_workers=4)  # async overlay JPG and 2DSeg saves

    # pre-load all bbox files into RAM — avoids repeated torch.load disk reads inside find_match
    bbox_cache = {
        cam.image_name: torch.load(cam.bbox_path) / cam.resolution_scale
        for cam in viewpoint_stack + viewpoint_stack_eval
    }
    print(f"Pre-loaded {len(bbox_cache)} bbox files into RAM")

    # Pre-decode every mask ONCE into a compact tight-bbox crop cache (kept in CPU RAM).
    # find_match otherwise re-reads each full-res mask PNG from disk on every overlapping
    # head — the dominant cost on dense data. Cropping is lossless for IoU (a mask is 0
    # outside its bbox). See docs/segmentation_3d/SEGMENTATION_3D_RUNTIME.md.
    #
    # Decoded in parallel but with a BOUNDED number of masks in flight. An earlier version
    # used ThreadPoolExecutor.map, which eagerly submitted all ~22k tasks so the decode
    # workers ran thousands of full-frame (~12 MB) masks ahead of consumption and buffered
    # them all -> OOM-killed the job during the build. Here we keep at most MAX_IN_FLIGHT
    # futures alive, so peak memory is bounded (only max_workers decode at once); the cached
    # crops themselves are tiny.
    def _decode_crop(job):
        """Decode one mask PNG straight to a tight-bbox crop entry (or None if empty). Runs in a
        worker thread. PURE NUMPY on purpose: routing the full-frame mask through a torch tensor
        (PILtoTorch -> .cpu().numpy()) pinned the whole 12 MB frame via the numpy view's torch
        `.base` on Euler's torch-2.1.2 CPU path -> a ~6.5 MB/mask RAM leak that OOM'd the FIP build
        at 36 GB (flat locally + on Euler phone, but real on Euler FIP). Decoding with PIL+numpy
        only never creates a full-frame torch tensor, so each frame is freed the moment this returns.
        Bit-identical to the old path: same resize, same >0 binarize, same bbox, same compact crop."""
        mask_path, resolution = job
        with Image.open(mask_path) as temp:
            # skip the resize when it's the identity (resolution=1 -> resolution == native size); a
            # same-size PIL resize is byte-identical to the raw pixels (verified 0-diff), so this is
            # lossless and just avoids a full-frame resample on every mask.
            temp = temp if tuple(resolution) == temp.size else temp.resize(resolution)
            arr = np.asarray(temp)  # (H,W) uint8 for 'L' masks
        # binarize_mask semantics: 1-channel -> pixel>0; 3-channel -> any channel>0. SAM masks are 0/255.
        m_np = (arr > 0) if arr.ndim == 2 else (arr > 0).any(axis=2)  # full-frame numpy bool, freed on return
        ys, xs = np.nonzero(m_np)
        if ys.size == 0:
            return mask_path, None
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop_np = m_np[y0:y1, x0:x1].copy()  # compact, independent buffer (m_np is released on return)
        return mask_path, (y0, y1, x0, x1, torch.from_numpy(crop_np), int(crop_np.sum()))

    t_cache = time.time()
    mask_cache = {}
    # cache off if the config flag says so OR the env override is set (env wins for the A/B script)
    _cache_disabled = (not use_mask_cache) or os.environ.get("WHEAT_SEG_NO_CACHE") == "1"
    if _cache_disabled:
        # Leave the cache empty so find_match falls back to the original per-candidate PNG decode.
        # Controlled by segmentation_3d.use_mask_cache=false (config) or WHEAT_SEG_NO_CACHE=1 (env,
        # used by the A/B benchmark script). Default: cache on.
        _why = "WHEAT_SEG_NO_CACHE=1 (env)" if os.environ.get("WHEAT_SEG_NO_CACHE") == "1" else "use_mask_cache=false (config)"
        print(f"mask crop cache DISABLED via {_why} -> baseline decode-per-candidate path")
    else:
        _jobs = [(mp, cam.resolution) for cam in viewpoint_stack for mp in cam.mask_paths]

        # ---- persistent disk cache -------------------------------------------------------------
        # The crops are a deterministic function of (masks, resolution), so a RERUN on the same plot
        # (our A/B sweeps) can LOAD them instead of re-decoding ~22k PNGs (~20 min -> seconds). Saved
        # NEXT TO THE MASKS so every seg run on that plot shares one file; keyed by a manifest =
        # resolution + hash of (basename,size) over all masks -> auto-rebuilds if any mask changed.
        # Keys stored by BASENAME so the file is portable across machines (Euler <-> local). Env
        # WHEAT_SEG_NO_DISK_CACHE=1 skips load+save (force in-memory rebuild).
        _masks_dir = os.path.dirname(_jobs[0][0]) if _jobs else None
        _res = tuple(_jobs[0][1]) if _jobs else None
        _no_disk = os.environ.get("WHEAT_SEG_NO_DISK_CACHE") == "1"
        _disk_file = (os.path.join(_masks_dir, f"crop_cache_{_res[0]}x{_res[1]}.pt")
                      if _masks_dir and _res else None)
        _manifest = None
        if _jobs:
            _h = hashlib.md5()
            for mp, _ in sorted(_jobs):
                _h.update(f"{os.path.basename(mp)}:{os.path.getsize(mp)};".encode())
            _manifest = _h.hexdigest()

        _loaded = False
        if _disk_file and not _no_disk and os.path.exists(_disk_file):
            try:
                _blob = torch.load(_disk_file, weights_only=False)
                if _blob.get("manifest") == _manifest and tuple(_blob.get("resolution", ())) == _res:
                    mask_cache = {os.path.join(_masks_dir, b): e for b, e in _blob["crops"].items()}
                    _loaded = True
                    print(f"Loaded mask crop cache from disk: {_disk_file} "
                          f"({len(mask_cache)} crops in {time.time() - t_cache:.1f}s -> skipped rebuild)")
                else:
                    print(f"Disk mask cache {_disk_file} is stale (masks changed) -> rebuilding")
            except Exception as _e:
                print(f"Disk mask cache unreadable ({_e}) -> rebuilding")

        if not _loaded:
            # workers sized to the ACTUAL cpu allocation (sched_getaffinity respects SLURM cpus-per-task),
            # minus ONE core reserved for the main/coordinator thread (submits jobs + fills the cache).
            # PIL releases the GIL during the C decode, so worker threads genuinely parallelise; RAM stays
            # flat (pure-numpy decode, no leak). Decode is BOUNDED-in-flight so full-frame masks can't pile up.
            try:
                _ncpu = len(os.sched_getaffinity(0))
            except AttributeError:
                _ncpu = os.cpu_count() or 4
            _n_workers = max(1, min(8, _ncpu - 1))
            MAX_IN_FLIGHT = 4 * _n_workers
            _it = iter(_jobs)
            _done_n = 0
            with ThreadPoolExecutor(max_workers=_n_workers) as _cache_exec:
                futures = {_cache_exec.submit(_decode_crop, job) for job in islice(_it, MAX_IN_FLIGHT)}
                with tqdm(total=len(_jobs), desc=f"Caching mask crops ({_n_workers}w)") as _pbar:
                    while futures:
                        done, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for fut in done:
                            mask_path, entry = fut.result()
                            mask_cache[mask_path] = entry  # entry already finalized (or None)
                            _pbar.update(1)
                            _done_n += 1
                            if _done_n % 2000 == 0:  # trace RAM growth to catch a leak during the build
                                _cmb = sum(e[4].numel() for e in mask_cache.values() if e is not None) / 1e6
                                tqdm.write(f"[RAM] build {_done_n}/{len(_jobs)}: "
                                           f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB peak, "
                                           f"cache so far {_cmb:.0f} MB")
                            nxt = next(_it, None)
                            if nxt is not None:
                                futures.add(_cache_exec.submit(_decode_crop, nxt))  # keep the pipeline topped up
            _sizes = [e[4].numel() for e in mask_cache.values() if e is not None]
            _cache_mb = sum(_sizes) / 1e6
            print(f"Cached {len(mask_cache)} mask crops in CPU RAM ({_cache_mb:.0f} MB, "
                  f"largest {max(_sizes) if _sizes else 0} px, mean {int(sum(_sizes)/max(len(_sizes),1))} px) "
                  f"in {time.time() - t_cache:.0f}s with {_n_workers} workers")
            # persist for next time (atomic write via .tmp + rename; basename keys -> portable)
            if _disk_file and not _no_disk:
                try:
                    _by_base = {os.path.basename(mp): mask_cache[mp] for mp, _ in _jobs if mp in mask_cache}
                    torch.save({"manifest": _manifest, "resolution": _res, "crops": _by_base}, _disk_file + ".tmp")
                    os.replace(_disk_file + ".tmp", _disk_file)
                    print(f"Saved mask crop cache to disk: {_disk_file} (reused on next seg run of this plot)")
                except Exception as _e:
                    print(f"Could not save disk mask cache ({_e}) -> continuing (rebuild next time)")

    twoD_seg_results = {} # 2D segmentation results update through the pipeline
    os.makedirs(f"{out_dir}/2DSeg", exist_ok=True)
    all_mask_paths = [] # a list of saved binary masks in png format
    num_bboxes = 0
    ### Initialize and save 2D Segmentation
    for viewpoint_cam in viewpoint_stack:
        bboxes = torch.load(viewpoint_cam.bbox_path)
        num_bboxes += len(bboxes)
        all_mask_paths += viewpoint_cam.mask_paths
        twoD_seg_results[viewpoint_cam.image_name] = torch.zeros(viewpoint_cam.original_image.shape[1:], dtype=torch.int)
        torch.save(twoD_seg_results[viewpoint_cam.image_name], f"{out_dir}/2DSeg/{viewpoint_cam.image_name}.pt")
    # Save for eval images too
    for viewpoint_cam in viewpoint_stack_eval:
        twoD_seg_results[viewpoint_cam.image_name] = torch.zeros(viewpoint_cam.original_image.shape[1:], dtype=torch.int)
        torch.save(twoD_seg_results[viewpoint_cam.image_name], f"{out_dir}/2DSeg/{viewpoint_cam.image_name}.pt")
    
    assert len(all_mask_paths) == num_bboxes
    print(f"Total of {len(all_mask_paths)} mask & bounding box pairs found")

    if wandb_enabled:
        import wandb
        wandb.init(
            project="wheat3dgs-seg",
            name=exp_name,
            config={
                "exp_name": exp_name,
                "iou_threshold": iou_threshold,
                "resolution": dataset.resolution,
                "total_masks": len(all_mask_paths),
                "num_train_cameras": len(viewpoint_stack),
                "save_vis_overlay": save_vis_overlay,
                "vis_max_heads": vis_max_heads,
            }
        )

    random.seed(seg_seed)  # fixed seed -> reproducible mask-processing order (see seg_seed config)
    random.shuffle(all_mask_paths)
    processed_masks = set()
    buffered_masks = set()
    num_wheat_head = 0
    overlap_counter = {}  # tracks how many times each head has been updated via overlap (for letter suffix)
    ply_futures = []  # async PLY saves, waited on at end

    if _SEG_TIMING:
        torch.cuda.synchronize(); _SEG_TIMER["setup"] = time.perf_counter() - _t_setup

    #### Iterate through all YOLO/SAM bbox/seg pairs
    for exp_id, this_mask_path in tqdm(enumerate(all_mask_paths), total=len(all_mask_paths), desc="Processing Masks"):
        # print("-" * 50)  # separator line fires 8500 times
        this_mask_name = os.path.splitext(os.path.basename(this_mask_path))[0]

        if this_mask_name in processed_masks:
            # print(f"{this_mask_name} already processed and saved")  # fires for majority of 8500 iterations
            continue
        elif this_mask_name in buffered_masks:
            buffered_masks.remove(this_mask_name)
            processed_masks.add(this_mask_name)
            # print(f"{this_mask_name} has been iterated once. will be moved to processed set.")

        this_image_name = this_mask_name[:-4]
        mask_idx = int(this_mask_name[-3:])
        # print(f"==== Train 3D segmentation against {this_mask_name} ====")  # fires for every unprocessed mask
        
        this_viewpoint_cam = next(cam for cam in viewpoint_stack if cam.image_name == this_image_name)
        
        # Optimize Gaussians' labels w.r.t ONE segmentation
        # NOTE: all_counts is additive
        all_counts = opt_label_w_seg(gaussians, [this_viewpoint_cam], [this_mask_path], pipe, background, pts_filter, mask_cache=mask_cache)
        all_obj_labels = counts_to_obj_labels(all_counts)
        if torch.sum(all_obj_labels, dim=1)[1] == 0:
            tqdm.write(f"WARNING: Can't identify Gaussians for mask {this_mask_name}, skipping")
            continue
        obj_used_mask = (all_obj_labels[1]).bool()

        #### Render from other cameras
        # Initialize a list of consistent segmentation for future fine-tuning
        matched_viewpoint_stack = [this_viewpoint_cam]
        matched_mask_paths = [this_mask_path]

        initial_targets = [vpt for vpt in viewpoint_stack if vpt.image_name != this_image_name]
        if frustum_cull:
            initial_targets = cull_cameras(initial_targets, gaussians, obj_used_mask)
        new_viewpoint_stack, new_mask_paths = find_match(
            target_viewpoint_stack = initial_targets,
            gs_params = (gaussians, pipe, background),
            obj_used_mask = obj_used_mask,
            iou_threshold = iou_threshold,
            dir_name = os.path.dirname(this_mask_path),
            bbox_cache = bbox_cache,
            mask_cache = mask_cache,
            verbose = True  # print match stats only on initial call, not in fine-tune loop
        )
        matched_viewpoint_stack += new_viewpoint_stack # as a whole
        matched_mask_paths += new_mask_paths
        processed_masks = update_processed_masks(processed_masks, new_mask_paths)
        # print(f"==== Find {len(new_mask_paths)} newly matched masks. {len(matched_mask_paths)} matched in total ====")  # covered by wheat head found message below

        #### Only do Refine training w.r.t newly found segmentation when a pairf of matches is found ####
        if len(new_viewpoint_stack) > 0:
            num_wheat_head += 1 # Potential wheat head
            # if find a match, then it's processed and create a dir for it
            this_mask_dir = f"{img_dir}/{num_wheat_head:04}"
            if save_vis_overlay and (vis_max_heads == 0 or num_wheat_head <= vis_max_heads):
                os.makedirs(this_mask_dir, exist_ok=True)
            processed_masks.add(this_mask_name)

            # print(f"==== Start refine training w.r.t the {num_wheat_head}th potential wheat head found ====")  # covered by wheat head found message below

            for i in range(1, 100):
                # print(f"-- fine-tuning iteration {i} --")  # up to 100x per head x 300 heads = 30000 lines
                assert len(new_viewpoint_stack) == len(new_mask_paths)
                # Update 3D Segmentation. When the ROI is active, re-apply pts_filter here too so
                # culled background Gaussians can't re-enter during fine-tuning (the initial lift
                # already filters them). ROI off -> pts_filter stays None here = original behaviour.
                update_counts = opt_label_w_seg(gaussians, new_viewpoint_stack, new_mask_paths, pipe, background,
                                                pts_filter=(pts_filter if _roi_active else None), mask_cache=mask_cache)
                assert update_counts.shape == all_counts.shape
                all_counts += update_counts # update all counts
                all_obj_labels = counts_to_obj_labels(all_counts)
                obj_used_mask = (all_obj_labels[1]).bool()
                # fine-tuning
                finetune_targets = [
                    vpt for vpt in viewpoint_stack if vpt.image_name not in {mpt.image_name for mpt in matched_viewpoint_stack}
                ]
                if frustum_cull:
                    finetune_targets = cull_cameras(finetune_targets, gaussians, obj_used_mask)
                new_viewpoint_stack, new_mask_paths = find_match(
                    target_viewpoint_stack = finetune_targets,
                    gs_params = (gaussians, pipe, background),
                    obj_used_mask = obj_used_mask,
                    iou_threshold = iou_threshold,
                    dir_name = os.path.dirname(this_mask_path),
                    bbox_cache = bbox_cache,
                    mask_cache = mask_cache,
                )
                if len(new_viewpoint_stack) == 0:
                    tqdm.write(f"  Fine-tuning converged after {i} iteration(s)")
                    break
                else:
                    matched_viewpoint_stack += new_viewpoint_stack # as a whole
                    matched_mask_paths += new_mask_paths
                    processed_masks = update_processed_masks(processed_masks, new_mask_paths)
                    # print(f"-- Find {len(new_mask_paths)} newly matched masks. {len(matched_mask_paths)} matched in total --")  # too verbose in fine-tune loop

            # Check if Gaussians are largely overlap with previously identified wheat head
            which_overlap_object = gaussians.reset_label(obj_used_mask=obj_used_mask, set_which_object_to=num_wheat_head)
            # only copy the 7 tensors needed for save_ply/prune_points, moved to CPU to avoid doubling VRAM
            if _SEG_TIMING:
                torch.cuda.synchronize(); _t = time.perf_counter()
            gaussians_obj = GaussianModel(gaussians.max_sh_degree)
            gaussians_obj._xyz = gaussians._xyz.detach().cpu()
            gaussians_obj._features_dc = gaussians._features_dc.detach().cpu()
            gaussians_obj._features_rest = gaussians._features_rest.detach().cpu()
            gaussians_obj._opacity = gaussians._opacity.detach().cpu()
            gaussians_obj._scaling = gaussians._scaling.detach().cpu()
            gaussians_obj._rotation = gaussians._rotation.detach().cpu()
            gaussians_obj._which_object = gaussians._which_object.detach().cpu()
            if _SEG_TIMING:
                _SEG_TIMER["ply_prep"] += time.perf_counter() - _t
            if which_overlap_object is not None:
                num_wheat_head -= 1 # if overlapping, then it's not a new wheat head
                if os.path.exists(this_mask_dir):
                    shutil.rmtree(this_mask_dir)
                which_wheat_head = which_overlap_object
                num_GS = torch.sum(gaussians_obj.get_which_object.detach() == which_wheat_head).item()
                gaussians_obj.prune_points(mask=torch.flatten(gaussians_obj.get_which_object.detach() != which_wheat_head), during_training=False)
                letter_suffix = find_new_mask_dir(overlap_counter, which_wheat_head)
                ply_futures.append(_overlay_executor.submit(gaussians_obj.save_ply, f"{ply_dir}/wh_{which_wheat_head:04}_{letter_suffix}.ply"))
                this_mask_dir = f"{img_dir}/{which_wheat_head:04}_{letter_suffix}"
                if save_vis_overlay and (vis_max_heads == 0 or which_wheat_head <= vis_max_heads):
                    os.makedirs(this_mask_dir, exist_ok=True)
                # print(f"Create new mask dir {this_mask_dir}")  # debug detail
                tqdm.write(f"[mask {exp_id+1}/{len(all_mask_paths)}] Wheat head #{which_wheat_head} updated (overlap) — {len(matched_viewpoint_stack)} matches, {num_GS} Gaussians\n")
                writer.writerow([f"{which_wheat_head:04}_{letter_suffix}", this_mask_name, str(len(matched_viewpoint_stack)), str(num_GS)])
                results.flush()
            else:
                which_wheat_head = num_wheat_head
                num_GS = torch.sum(gaussians_obj.get_which_object.detach() == which_wheat_head).item()
                gaussians_obj.prune_points(mask=torch.flatten(gaussians_obj.get_which_object.detach() != which_wheat_head), during_training=False)
                ply_futures.append(_overlay_executor.submit(gaussians_obj.save_ply, f"{ply_dir}/wh_{which_wheat_head:04}.ply"))
                tqdm.write(f"[mask {exp_id+1}/{len(all_mask_paths)}] Wheat head #{which_wheat_head} found — {len(matched_viewpoint_stack)} matches, {num_GS} Gaussians\n")
                writer.writerow([f"{which_wheat_head:04}", this_mask_name, str(len(matched_viewpoint_stack)), str(num_GS)])
                results.flush()

            if wandb_enabled:
                wandb.log({
                    "wheat_heads_found": num_wheat_head,
                    "head_matches": len(matched_viewpoint_stack),
                    "head_gaussians": num_GS,
                    "head_is_overlap": which_overlap_object is not None,
                    "masks_processed": len(processed_masks),
                    "masks_buffered": len(buffered_masks),
                    "mask_progress_pct": (exp_id + 1) / len(all_mask_paths) * 100,
                })

            # Save used count for future use (commented out: per-head count files are not used by any downstream step)
            # print(f"For wh {which_wheat_head}, all_counts.shape {all_counts.shape}")
            # counts_save_pth = f"{count_dir}/{which_wheat_head:04}.pt"
            # if os.path.exists(counts_save_pth):
            #     print(f"{counts_save_pth} exists...removing...")
            #     os.remove(counts_save_pth)
            # torch.save(all_counts.detach().cpu(), counts_save_pth)

            #### Evaluation of refined training ####
            # os.makedirs(f"{this_mask_dir}/overlay", exist_ok=True)
            # frustum cull: only cameras the head actually projects into paint anything into
            # twoD_seg_results (empty alpha elsewhere is a no-op paint), so this stays bit-identical.
            commit_cams = viewpoint_stack + viewpoint_stack_eval
            if frustum_cull:
                commit_cams = cull_cameras(commit_cams, gaussians, obj_used_mask)
            overlay_futures = []
            for i, viewpoint_cam in enumerate(commit_cams):
                with torch.no_grad():
                    if _SEG_TIMING:
                        torch.cuda.synchronize(); _t = time.perf_counter()
                    render_pkg = flashsplat_render(viewpoint_cam, gaussians, pipe, background, used_mask=obj_used_mask, inference=_INFERENCE)
                    render_alpha = render_pkg["alpha"].squeeze().detach()  # keep on GPU; transfer only what we paint
                    pos = render_alpha > 0.5                               # the pixels this head paints
                    if _SEG_TIMING:
                        torch.cuda.synchronize(); _SEG_TIMER["commit_render"] += time.perf_counter() - _t
                        _t = time.perf_counter()  # commit_paint = the threshold->crop->paint below
                    target = twoD_seg_results[viewpoint_cam.image_name]
                    assert target.shape == render_alpha.shape
                    want_overlay = save_vis_overlay and (vis_max_heads == 0 or which_wheat_head <= vis_max_heads)
                    if _FAST_PAINT and not want_overlay:
                        # fast path: paint only the head's bounding box. Bit-identical to the full-frame paint
                        # because outside the bbox no alpha exceeds 0.5, so nothing would be written there.
                        rows = torch.any(pos, dim=1)
                        if bool(rows.any()):
                            cols = torch.any(pos, dim=0)
                            ys = torch.nonzero(rows, as_tuple=False)
                            xs = torch.nonzero(cols, as_tuple=False)
                            y0, y1 = int(ys[0]), int(ys[-1]) + 1
                            x0, x1 = int(xs[0]), int(xs[-1]) + 1
                            crop = pos[y0:y1, x0:x1].cpu()                 # tiny transfer (head bbox only)
                            target[y0:y1, x0:x1][crop] = which_wheat_head  # writes back through the view slice
                    else:
                        # full-frame path: needed when we also save the overlay (it wants a full-frame mask)
                        pos_cpu = pos.cpu()
                        if want_overlay:
                            pred_seg = pos_cpu.numpy()
                            # async: submit overlay save to background thread while GPU renders next camera
                            overlay_futures.append(_overlay_executor.submit(
                                vis_image_w_overlay,
                                img_tensor=viewpoint_cam.original_image.detach().cpu(),  # CPU copy for thread safety
                                save_dir=f"{this_mask_dir}",
                                save_name=viewpoint_cam.image_name,
                                pred_seg=pred_seg,
                                resize_factor=4
                            ))
                        target[pos_cpu] = which_wheat_head
                    if _SEG_TIMING:
                        _SEG_TIMER["commit_paint"] += time.perf_counter() - _t
                    # 2DSeg saved once at end of pipeline (optimization 2: was per-camera per-head = 10,800 writes)
            # wait for all overlay saves before moving to next wheat head
            if _SEG_TIMING:
                _t = time.perf_counter()
            for f in overlay_futures:
                f.result()
            if _SEG_TIMING:
                _SEG_TIMER["overlay_wait"] += time.perf_counter() - _t
                    
        else:
            # print(f"==== Not matchings found for {this_mask_name}. Add to Buffer. ====")  # fires too often, not useful
            if this_mask_name not in processed_masks and this_mask_name not in buffered_masks:
                buffered_masks.add(this_mask_name)

        # if exp_id % 5 == 0: # Save Gaussians every 5 distinct masks (commented out: checkpoint is never resumed from, and fires 1700x for 8500 masks)
        #     gaussians.save_ply(f"{out_dir}/gaussians.ply")
        #     print("Gaussians saved!")

        # print(f"======== Processed masks {len(processed_masks.union(buffered_masks))} / {len(all_mask_paths)} ========")  # tqdm bar already shows this
        # print("-" * 50)
        
    gaussians.save_ply(f"{out_dir}/gaussians.ply")

    # build and save all_obj_labels.pth — needed by export_colored_ply.py and the viewer
    # shape: (num_wheat_head+1, n_gaussians) — row 0 is background, row i is wheat head i
    which_obj = gaussians.get_which_object.squeeze().cpu()
    n_gs = which_obj.shape[0]
    all_obj_labels = torch.zeros(num_wheat_head + 1, n_gs, dtype=torch.bool)
    for i in range(num_wheat_head + 1):
        all_obj_labels[i] = (which_obj == i)
    torch.save(all_obj_labels, f"{out_dir}/all_obj_labels.pth")

    results.close()

    if _SEG_TIMING:
        _t = time.perf_counter()
    # wait for all async PLY saves to finish before saving final state
    for f in ply_futures:
        f.result()

    # save all 2DSeg label maps once at the end (optimization 2: was 300 heads × 36 cameras = 10,800 writes)
    all_eval_cams = viewpoint_stack + viewpoint_stack_eval
    seg_futures = [
        _overlay_executor.submit(torch.save, twoD_seg_results[cam.image_name], f"{out_dir}/2DSeg/{cam.image_name}.pt")
        for cam in all_eval_cams
    ]
    for f in seg_futures:
        f.result()
    _overlay_executor.shutdown(wait=True)
    if _SEG_TIMING:
        _SEG_TIMER["seg2d_save"] += time.perf_counter() - _t  # final PLY-drain + 2DSeg write

    print(f"\n{'='*60}")
    print(f"  SEGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Wheat heads found:       {num_wheat_head}")
    print(f"  Masks matched:           {len(processed_masks)} / {len(all_mask_paths)}")
    print(f"  Masks never matched:     {len(buffered_masks)}   (no cross-camera confirmation)")
    print(f"  Results saved to:        {out_dir}")
    # peak resource usage — ru_maxrss is peak RSS in KB on Linux; torch tracks peak VRAM for free
    print(f"  Peak CPU RAM:            {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
    if torch.cuda.is_available():
        print(f"  Peak GPU VRAM:           {torch.cuda.max_memory_allocated() / 1e9:.1f} GB alloc / "
              f"{torch.cuda.max_memory_reserved() / 1e9:.1f} GB reserved")
    if _SEG_TIMING:
        tot = sum(_SEG_TIMER.values())
        print("  ---- seg time breakdown (WHEAT_SEG_TIMING) ----")
        if tot > 0:
            order = ["render", "match", "commit_render", "lift_render", "ilp", "lift_decode",
                     "setup", "commit_paint", "ply_prep", "overlay_wait", "seg2d_save"]
            labels = {"render": "find_match render", "match": "find_match match/IoU",
                      "commit_render": "commit render (2DSeg GPU)", "lift_render": "lift render (full model)",
                      "ilp": "ILP solve", "lift_decode": "lift mask decode",
                      "setup": "setup (load+cache build)", "commit_paint": "commit paint (CPU 2DSeg)",
                      "ply_prep": "per-head PLY prep (CPU copy)", "overlay_wait": "overlay-save wait",
                      "seg2d_save": "final 2DSeg save"}
            for k in order:
                print(f"    {labels[k]:<28} {_SEG_TIMER[k]:8.0f}s ({_SEG_TIMER[k]/tot*100:5.1f}%)")
            print(f"    {'TOTAL (timed)':<28} {tot:8.0f}s   (any remaining untimed = pure Python loop overhead)")
        else:
            print("    (no timing)")
    print(f"{'='*60}")

    # persist the headline numbers to a small JSON so downstream code (e.g. head-count vs GT eval) can
    # read the total predicted wheat-head count without scraping stdout or counting ply/wh_*.ply files.
    # num_wheat_head is the plot-level 3D instance count (one ID per head, matched across views).
    with open(f"{out_dir}/seg_summary.json", "w") as f:
        json.dump({
            "exp_name": exp_name,
            "wheat_heads_found": num_wheat_head,   # = predicted total head count for this plot
            "masks_matched": len(processed_masks),
            "masks_unmatched": len(buffered_masks),
            "total_masks": len(all_mask_paths),
            "iou_threshold": iou_threshold,
        }, f, indent=2)

    if wandb_enabled:
        wandb.summary["total_wheat_heads"] = num_wheat_head
        wandb.summary["masks_matched"] = len(processed_masks)
        wandb.summary["masks_unmatched"] = len(buffered_masks)
        wandb.summary["total_masks"] = len(all_mask_paths)
        wandb.finish()
        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--load_iteration', type=str, default="-1")
    parser.add_argument("--exp_name", type=str, help="Exp name")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IOU threshold for matching")
    parser.add_argument("--save_vis_overlay", action="store_true", default=True, help="Save overlay JPGs per wheat head per camera")
    parser.add_argument("--no_save_vis_overlay", dest="save_vis_overlay", action="store_false")
    parser.add_argument("--vis_max_heads", type=int, default=10, help="Save overlays for first N heads only. 0 = all heads.")
    parser.add_argument("--use_mask_cache", action="store_true", default=True, help="Pre-decode masks into a tight-bbox crop cache (big speedup, bit-identical)")
    parser.add_argument("--no_mask_cache", dest="use_mask_cache", action="store_false", help="Disable the crop cache (old decode-per-candidate baseline)")
    parser.add_argument("--seg_seed", type=int, default=0, help="Seed for the mask-processing shuffle (reproducible seg)")
    parser.add_argument("--frustum_cull", action="store_true", default=False, help="Skip rendering a head into cameras where its Gaussians don't project (bit-identical/lossless). DEFAULT OFF — barely helps on FIP-overhead / phone-orbit captures where heads are in ~every view; enable for a long linear sweep.")
    parser.add_argument("--no_frustum_cull", dest="frustum_cull", action="store_false", help="Render every camera (the default now).")
    parser.add_argument("--roi_cull", action="store_true", default=False, help="Restrict segmentation to the plot ROI: cull Gaussians whose horizontal position is outside the marker hull so the background (ground/canopy behind the plot) can't form one big blob. Needs logs/marker_points3d.json. DEFAULT OFF (byte-identical).")
    parser.add_argument("--roi_buffer_m", type=float, default=0.25, help="Grow the ROI hull outward this many metres so plot-edge heads aren't clipped (default 0.25).")
    parser.add_argument("--height_band", action="store_true", default=False, help="Separate vertical filter (NOT the ROI): also cull Gaussians outside a height band around the marker plane (sky floaters / underground junk). Near a no-op on flat wheat. DEFAULT OFF.")
    parser.add_argument("--marker_exclude", action="store_true", default=False, help="Drop Gaussians in a small 3D sphere around each coded-marker plate (just the plate, not a vertical column, so nearby heads survive) so markers aren't segmented as heads. DEFAULT OFF.")
    parser.add_argument("--marker_radius_m", type=float, default=0.075, help="Marker-exclusion sphere radius in metres (plates are ~13 cm circle / 15 cm square; but the RECONSTRUCTED marker is ~3x that, so 0.075 leaks the rim — prefer --marker_radius_rel; default 0.075).")
    parser.add_argument("--marker_radius_rel", type=float, default=0.0, help="If >0, override --marker_radius_m with (this x median marker spacing) — SCALE-FREE, safe across sessions with different COLMAP units. ~0.065 gives ~0.20 u on A/0715. Default 0.0 = use the absolute metre value.")
    parser.add_argument("--legacy_ground_cull", action="store_true", default=False, help="Force the OLD world-z ground cull (z<z_mean, culls the lower HALF and bisects phone heads) instead of the gentle marker-plane cull. Only for reproducing pre-fix runs.")
    parser.add_argument("--ground_percentile", type=float, default=10.0, help="Default ground cull: cull the bottom this-percent of Gaussians by height above the marker plane (scale-free, keeps heads whole). Default 10.")
    parser.add_argument("--wandb_enabled", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)

    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.load_iteration, args.exp_name, args.iou_threshold,
             args.save_vis_overlay, args.vis_max_heads, args.wandb_enabled,
             use_mask_cache=args.use_mask_cache, seg_seed=args.seg_seed, frustum_cull=args.frustum_cull,
             roi_cull=args.roi_cull, height_band=args.height_band, marker_exclude=args.marker_exclude,
             roi_buffer_m=args.roi_buffer_m, marker_radius_m=args.marker_radius_m,
             marker_radius_rel=args.marker_radius_rel,
             legacy_ground_cull=args.legacy_ground_cull, ground_percentile=args.ground_percentile)