import os
import sys
import csv
import glob
import json
import time
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

# Optional render-vs-match timing (set WHEAT_SEG_TIMING=1). Off by default so the
# cuda.synchronize() calls it needs don't slow normal runs.
_SEG_TIMING = os.environ.get("WHEAT_SEG_TIMING") == "1"
_SEG_TIMER = {"render": 0.0, "match": 0.0}

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
                    pipeline, background, pts_filter=None):
    """
    Helper function that wraps Gaussians label optimization schema into one function
    return:
        all_counts: counts that are additive
        all_obj_labels:
    """
    assert len(viewpoint_stack) == len(mask_paths)
    
    all_counts = None
    for idx, viewpoint_cam in enumerate(viewpoint_stack):
        with Image.open(mask_paths[idx]) as temp:
            gt_mask = binarize_mask(PILtoTorch(temp.copy(), viewpoint_cam.resolution)).squeeze().to("cuda")
            assert viewpoint_cam.original_image.shape[-2:] == gt_mask.shape
        with torch.no_grad():
            render_pkg = flashsplat_render(viewpoint_cam, gaussians, pipeline, background, gt_mask=gt_mask, obj_num=1)
            rendering = render_pkg["render"]
            used_count = render_pkg["used_count"]
            if all_counts is None:
                all_counts = torch.zeros_like(used_count)
            all_counts += used_count
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
    all_obj_labels = multi_instance_opt(all_counts, slackness)
    # print(f"{torch.sum(all_obj_labels, dim=1)[1]} Gaussians identified")  # too verbose: fires for every mask + every fine-tune iteration
    return all_obj_labels

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
            render_pkg = flashsplat_render(viewpoint_cam, gaussians, pipe, background, used_mask=obj_used_mask)
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
        
def training(dataset, opt, pipe, load_iteration, exp_name, iou_threshold, save_vis_overlay, vis_max_heads, wandb_enabled=False, use_mask_cache=True):
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

    z_mean = torch.mean(gaussians.get_xyz.cpu()[:, 2])
    # print(f"All Gaussians z_min: {torch.min(gaussians.get_xyz.cpu()[:, 2])} zmax: {torch.max(gaussians.get_xyz.cpu()[:, 2])}")  # debug detail
    pts_filter = (gaussians.get_xyz.cpu()[:, 2] < z_mean)

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
        """Decode one mask PNG to a tight-bbox crop entry (or None if empty). Runs in a worker thread."""
        mask_path, resolution = job
        with Image.open(mask_path) as temp:
            m = binarize_mask(PILtoTorch(temp.copy(), resolution)).squeeze() > 0  # full-frame CPU bool, freed when worker returns
        entry = build_mask_crop(m)  # clones a tiny crop -> m can be released
        if entry is None:
            return mask_path, None
        y0, y1, x0, x1, crop, area = entry
        return mask_path, (y0, y1, x0, x1, crop, int(area))

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
        MAX_IN_FLIGHT = 16  # cap concurrent+buffered decodes so full-frame masks can't pile up
        _it = iter(_jobs)
        _done_n = 0
        with ThreadPoolExecutor(max_workers=4) as _cache_exec:  # 4 concurrent decodes -> small transient spike
            futures = {_cache_exec.submit(_decode_crop, job) for job in islice(_it, MAX_IN_FLIGHT)}
            with tqdm(total=len(_jobs), desc="Caching mask crops") as _pbar:
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
              f"in {time.time() - t_cache:.0f}s")

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

    random.shuffle(all_mask_paths)
    processed_masks = set()
    buffered_masks = set()
    num_wheat_head = 0
    overlap_counter = {}  # tracks how many times each head has been updated via overlap (for letter suffix)
    ply_futures = []  # async PLY saves, waited on at end
    
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
        all_counts = opt_label_w_seg(gaussians, [this_viewpoint_cam], [this_mask_path], pipe, background, pts_filter)
        all_obj_labels = counts_to_obj_labels(all_counts)
        if torch.sum(all_obj_labels, dim=1)[1] == 0:
            tqdm.write(f"WARNING: Can't identify Gaussians for mask {this_mask_name}, skipping")
            continue
        obj_used_mask = (all_obj_labels[1]).bool()

        #### Render from other cameras
        # Initialize a list of consistent segmentation for future fine-tuning
        matched_viewpoint_stack = [this_viewpoint_cam]
        matched_mask_paths = [this_mask_path]
        
        new_viewpoint_stack, new_mask_paths = find_match(
            target_viewpoint_stack = [vpt for vpt in viewpoint_stack if vpt.image_name != this_image_name],
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
                # Update 3D Segmentation
                update_counts = opt_label_w_seg(gaussians, new_viewpoint_stack, new_mask_paths, pipe, background)
                assert update_counts.shape == all_counts.shape
                all_counts += update_counts # update all counts
                all_obj_labels = counts_to_obj_labels(all_counts)
                obj_used_mask = (all_obj_labels[1]).bool()
                # fine-tuning
                new_viewpoint_stack, new_mask_paths = find_match(
                    target_viewpoint_stack = [
                        vpt for vpt in viewpoint_stack if vpt.image_name not in {mpt.image_name for mpt in matched_viewpoint_stack}
                    ],
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
            gaussians_obj = GaussianModel(gaussians.max_sh_degree)
            gaussians_obj._xyz = gaussians._xyz.detach().cpu()
            gaussians_obj._features_dc = gaussians._features_dc.detach().cpu()
            gaussians_obj._features_rest = gaussians._features_rest.detach().cpu()
            gaussians_obj._opacity = gaussians._opacity.detach().cpu()
            gaussians_obj._scaling = gaussians._scaling.detach().cpu()
            gaussians_obj._rotation = gaussians._rotation.detach().cpu()
            gaussians_obj._which_object = gaussians._which_object.detach().cpu()
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
            overlay_futures = []
            for i, viewpoint_cam in enumerate(viewpoint_stack + viewpoint_stack_eval):
                with torch.no_grad():
                    render_pkg = flashsplat_render(viewpoint_cam, gaussians, pipe, background, used_mask=obj_used_mask)
                    render_alpha = render_pkg["alpha"].squeeze().detach().cpu()
                    pred_seg = render_alpha.numpy() > 0.5
                    mask = Image.fromarray(np.where(pred_seg, 255, 0).astype(np.uint8), mode='L')
                    # mask.save(f"{this_mask_dir}/masks/{viewpoint_cam.image_name}.jpg")
                    if save_vis_overlay and (vis_max_heads == 0 or which_wheat_head <= vis_max_heads):
                        # async: submit overlay save to background thread while GPU renders next camera
                        overlay_futures.append(_overlay_executor.submit(
                            vis_image_w_overlay,
                            img_tensor=viewpoint_cam.original_image.detach().cpu(),  # CPU copy for thread safety
                            save_dir=f"{this_mask_dir}",
                            save_name=viewpoint_cam.image_name,
                            pred_seg=pred_seg,
                            resize_factor=4
                        ))
                    # Update the 2D seg&count results
                    assert twoD_seg_results[viewpoint_cam.image_name].shape == render_alpha.shape
                    twoD_seg_results[viewpoint_cam.image_name][render_alpha > 0.5] = which_wheat_head
                    # 2DSeg saved once at end of pipeline (optimization 2: was per-camera per-head = 10,800 writes)
            # wait for all overlay saves before moving to next wheat head
            for f in overlay_futures:
                f.result()
                    
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
        tot = _SEG_TIMER["render"] + _SEG_TIMER["match"]
        print(f"  find_match time split:   render {_SEG_TIMER['render']:.0f}s ({_SEG_TIMER['render']/tot*100:.0f}%)  "
              f"match/IoU {_SEG_TIMER['match']:.0f}s ({_SEG_TIMER['match']/tot*100:.0f}%)" if tot > 0 else "  (no timing)")
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
    parser.add_argument("--wandb_enabled", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)

    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.load_iteration, args.exp_name, args.iou_threshold,
             args.save_vis_overlay, args.vis_max_heads, args.wandb_enabled,
             use_mask_cache=args.use_mask_cache)