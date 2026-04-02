"""
sam_v1_pipelined.py — pipelined SAM inference (upgraded from sam_v1.py)

HOW IT WORKS:
Each image goes through 3 stages:

    Stage 1 — Pre-Processing (CPU):  load image + bbox tensor from disk
    Stage 2 — GPU Inference:         SAM image encoding (set_image) + mask decoding (predict_torch)
    Stage 3 — Post-Processing (CPU): save individual mask PNGs + overlay visualization

In sam_v1 these 3 stages ran sequentially per image. In the pipelined version:

    while GPU encodes image N   →  CPU loads image N+1      (background thread)
                                →  CPU saves masks N-1      (background thread)

This keeps the GPU continuously busy and hides most of the disk I/O behind the SAM encoder,
which is the bottleneck (~1-2s per image for ViT-H).

CONCURRENCY STRUCTURE:
- One outer ThreadPoolExecutor (max_workers=2): one slot for the load future, one for the save future.
- The save task spawns its own inner pool (MAX_THREADS threads) for parallel mask PNG writing.
- The main thread runs GPU inference (set_image + predict_torch).
- torch.cuda.synchronize() blocks the main thread but background threads keep running — they are
  independent OS threads unaffected by the main thread waiting for the GPU.
"""

# sam_module.py
import os
import glob
import time
import concurrent.futures
import gc
import numpy as np
import cv2
import torch
import colorsys
import shutil
import wandb
from segment_anything import sam_model_registry, SamPredictor

# Import from config
from config_v1 import *


# =====================================================================
#-------- HELPER FUNCTIONS FOR SAM --------
# =====================================================================

# Deletes all contents of a folder and recreates it,
# since thats cheaper than to go through the folder and delete every single item.
def reset_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path) # Deletes the folder and everything inside
    os.makedirs(folder_path, exist_ok=True)  # Recreates the empty folder


# Color Generator, turns a single number into a specific rgb color
# dynamically handles any ID number with a high maximum limit
def id2rgb(id, max_num_obj=65535):
    if id == 0: # invalid region / background
        return np.zeros((3, ), dtype=np.uint8)
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")
    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1) # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5 # Alternate between 0.5 and 1.0
    l = 0.5
    # Use colorsys to convert HSL to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return np.array([int(r*255), int(g*255), int(b*255)], dtype=np.uint8)


# for visualize color mask, basically every wheat head another color
def visualize_obj(objects):
    assert len(objects.shape) == 2
    # Create the blank RGB canvas
    rgb_mask = np.zeros((*objects.shape, 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects) # Get all unique IDs present in the image
    for id in all_obj_ids:
        if id == 0:
            continue # Skip the background
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask


# Prints a single line report for one image in the SAM process
def print_sam_step_report(idx, total_imgs, name, n_heads, t_embed, t_pred):
    """Print a per-image timing line. t_save is excluded — it runs async in the background."""
    print(f"  [{idx+1}/{total_imgs}] {name:<20} | "
          f"Embed: {t_embed:>5.2f}s | Pred: {t_pred:>5.2f}s | "
          f"Heads: {n_heads:>3}")


# Prints the final summary for the entire plot's SAM processing.
def print_sam_plot_summary(num_images, total_time):
    print(f"\n" + "="*45)
    print(f"      SAM PLOT SUMMARY ({num_images} images)")
    print("="*45)
    print(f"{'Total SAM Time (pipelined):':<25} {total_time:>8.2f}s")
    print(f"{'Average Time Per Image:':<25} {total_time/num_images:>8.2f}s")
    print("="*45 + "\n")


# =====================================================================
# -------- PIPELINE HELPERS (new in pipelined version) --------
# =====================================================================

def _load_image_and_bbox(image_file, bbox_folder):
    """Load one image from disk and its corresponding bbox .pt tensor.
    Returns (image_name, save_name, image_rgb, bbox_or_None) — None signals a missing bbox file."""
    image_name = os.path.basename(image_file)
    save_name = os.path.splitext(image_name)[0]
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bbox_path = os.path.join(bbox_folder, save_name + ".pt")
    if not os.path.exists(bbox_path):
        return image_name, save_name, image, None  # None signals missing bbox
    bbox = torch.load(bbox_path)
    return image_name, save_name, image, bbox


def _save_image_results(masks_np, image, save_name, image_name, mask_folder, sam_vis_folder):
    """Save all mask PNGs and the overlay visualization for one image. Returns t_save."""
    t_start_save = time.perf_counter()

    # 4. Saving & Visualization (Parallel CPU Saving)
    objects = np.zeros((masks_np.shape[1], masks_np.shape[2]), dtype=np.uint16)
    save_tasks = []

    for idx, mask_np in enumerate(masks_np):
        objects[mask_np > 0] = idx + 1
        out_path = os.path.join(mask_folder, f"{save_name}_{idx:03}.png")
        save_tasks.append((out_path, mask_np))

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        list(executor.map(lambda arg: cv2.imwrite(arg[0], arg[1]), save_tasks))

    # Reverted to your highly efficient sparse NumPy math
    color_mask = visualize_obj(objects) / 255.0
    color_img = image / 255.0
    non_black_pixels = np.any(color_mask > 0, axis=-1)

    overlayed_img = color_img.copy()
    alpha = 0.5
    overlayed_img[non_black_pixels, :] = (alpha * color_mask[non_black_pixels, :] +
                                          (1 - alpha) * color_img[non_black_pixels, :])

    cv2.imwrite(
        os.path.join(sam_vis_folder, image_name.replace(".png", ".jpg")),
        (overlayed_img * 255).astype(np.uint8)[:, :, ::-1]
    )

    return time.perf_counter() - t_start_save


# =====================================================================
# MAIN SAM INFERENCE (ALL PLOTS)
# =====================================================================
def run_sam_phase(image_folders):
    print("\n" + "="*50)
    print(" PHASE 2: LOADING SAM AND PROCESSING ALL PLOTS")
    print("="*50)

    if WANDB_ENABLED:
        wandb.init(
            project="wheat3dgs-sam-v1",
            config={"batch_size_sam_box": BATCH_SIZE_SAM_BOX, "device": DEVICE},
        )

    # Load SAM Model ONCE
    print("Loading SAM (this takes a few seconds)...")
    start_sam_load = time.perf_counter()
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(device=DEVICE)
    sam = torch.compile(sam)
    predictor = SamPredictor(sam)
    torch.cuda.synchronize()
    sam_load_time = time.perf_counter() - start_sam_load
    print(f"-> SAM Model loaded on {DEVICE} in {sam_load_time:.2f}s")

    total_sam_pure_time = 0.0
    total_sam_images = 0

    for folder in image_folders:
        plot_name = folder.split(os.sep)[-2] # Get parent folder name (e.g. plot_461)
        print(f"\n[SAM Phase] Processing Plot: {plot_name}")

        base_plot_path = os.path.dirname(folder)
        bbox_folder = os.path.join(base_plot_path, "bboxes")
        sam_vis_folder = os.path.join(base_plot_path, "sam_vis")
        mask_folder = os.path.join(base_plot_path, "masks")
        reset_folder(sam_vis_folder)
        reset_folder(mask_folder)

        image_files = glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg'))
        if LIMIT_IMAGES > 0:
            image_files = image_files[:LIMIT_IMAGES]

        start_sam_plot = time.perf_counter()
        n_images = len(image_files)
        save_futures = []  # collect save futures so we can harvest t_save at the end

        # Outer executor has 2 slots: one for the load future, one for the save future.
        # The save task spawns its own inner pool (MAX_THREADS) for parallel mask PNG writing.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

            # --- PRE-PROCESSING: kick off load for image 0 immediately ---
            # It runs in the background while the loop sets up, so it's often already
            # done by the time we call .result() below.
            load_future = executor.submit(_load_image_and_bbox, image_files[0], bbox_folder)

            prev_save_data = None  # holds (masks_np, image, save_name, image_name, ...) from previous GPU step

            for i in range(n_images):
                # --- PRE-PROCESSING: collect load results for the current image ---
                # .result() blocks until the parallel load is done — usually already finished
                # since it ran during the previous GPU call.
                image_name, save_name, image, bbox = load_future.result()

                if os.path.exists(os.path.join(sam_vis_folder, image_name)):
                    continue

                # --- PRE-PROCESSING: submit load for the NEXT image ---
                # This starts immediately and runs in parallel while the GPU works below.
                if i + 1 < n_images:
                    load_future = executor.submit(_load_image_and_bbox, image_files[i + 1], bbox_folder)

                # --- POST-PROCESSING: submit save for the PREVIOUS image ---
                # Also starts immediately and runs in parallel while the GPU works below.
                # _save_image_results writes all mask PNGs + the overlay visualization.
                if prev_save_data is not None:
                    sf = executor.submit(_save_image_results, *prev_save_data)
                    save_futures.append(sf)
                    prev_save_data = None  # drop reference so RAM can be freed once save is done

                # Handle skip cases (missing bbox file or no detections)
                if bbox is None:
                    print(f"    Warning: No boxes found for {image_name}, skipping SAM.")
                    continue
                if len(bbox) == 0:
                    print(f"    No wheat heads detected in {image_name}")
                    continue

                bbox = bbox.to(DEVICE)

                # --- GPU INFERENCE: image embedding (the heavy bottleneck) ---
                # Main thread blocks here. load(N+1) and save(N-1) run on CPU in parallel.
                t_start_img = time.perf_counter()

                # 2. Image Embedding (heavy part)
                predictor.set_image(image) # only takes one image a time, thats why BATCH_SIZE_SAM_BOX=1
                torch.cuda.synchronize()
                t_embed = time.perf_counter() - t_start_img

                # --- GPU INFERENCE: predict masks (batched to prevent VRAM overflow) ---
                t_start_pred = time.perf_counter()

                # 3. Predict Masks (Batching to prevent RAM/VRAM overflow)
                transformed_boxes = predictor.transform.apply_boxes_torch(bbox, image.shape[:2])
                all_masks_np = []

                # Process boxes in batches to save memory
                with torch.no_grad(): # important for Batch Size 1
                    for b_idx in range(0, len(transformed_boxes), BATCH_SIZE_SAM_BOX):
                        batch_boxes = transformed_boxes[b_idx : b_idx + BATCH_SIZE_SAM_BOX]
                        masks, _, _ = predictor.predict_torch(
                            point_coords=None,
                            point_labels=None,
                            boxes=batch_boxes,
                            multimask_output=False
                        )
                        # squeeze(1) removes the empty dimension, leaving [Batch, H, W]
                        # immediately move to CPU and convert to uint8 to free up VRAM
                        masks_batch_np = (masks.squeeze(1).cpu().numpy() * 255).astype(np.uint8)
                        all_masks_np.append(masks_batch_np)

                # Combine all chunks back into one numpy array
                masks_np = np.concatenate(all_masks_np, axis=0)
                torch.cuda.synchronize()
                t_pred = time.perf_counter() - t_start_pred

                if SHOW_TIME_SAM:
                    print_sam_step_report(i, n_images, image_name, len(bbox), t_embed, t_pred)
                if WANDB_ENABLED:
                    wandb.log({
                        "plot":      plot_name,
                        "t_embed_s": t_embed,
                        "t_pred_s":  t_pred,
                        "n_heads":   len(bbox),
                    })
                total_sam_pure_time += (t_embed + t_pred)
                total_sam_images += 1

                # store results so we can submit the save on the next loop iteration
                prev_save_data = (masks_np, image, save_name, image_name, mask_folder, sam_vis_folder)

                # Cleanup loop to prevent VRAM overflow
                predictor.reset_image()
                torch.cuda.empty_cache()
                gc.collect()

            # --- POST-PROCESSING: save the last image ---
            # No next GPU call to overlap with, but we still submit so it runs in the background
            # while the executor waits for all futures to finish on __exit__.
            if prev_save_data is not None:
                sf = executor.submit(_save_image_results, *prev_save_data)
                save_futures.append(sf)

            # Collect t_save from all completed save futures and add to total
            for sf in save_futures:
                total_sam_pure_time += sf.result()

        # Plot Final Summary for SAM
        sam_total_plot = time.perf_counter() - start_sam_plot
        if SHOW_TIME_SAM:
            print_sam_plot_summary(len(image_files), sam_total_plot)
        print(f"  Finished Plot: {plot_name}")

    # End of SAM MAIN -> Free SAM memory
    del sam, predictor
    torch.cuda.empty_cache()
    gc.collect()

    if WANDB_ENABLED:
        wandb.finish()

    return total_sam_pure_time, total_sam_images
