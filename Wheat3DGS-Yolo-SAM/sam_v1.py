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
def print_sam_step_report(idx, total_imgs, name, n_heads, t_embed, t_pred, t_save):
    t_total = t_embed + t_pred + t_save
    print(f"  [{idx+1}/{total_imgs}] {name:<20} | "
          f"Embed: {t_embed:>5.2f}s | Pred: {t_pred:>5.2f}s | "
          f"Save: {t_save:>5.2f}s | Heads: {n_heads:>3} | Total: {t_total:>6.2f}s")


# Prints the final summary for the entire plot's SAM processing.
def print_sam_plot_summary(num_images, total_time):
    print(f"\n" + "="*45)
    print(f"      SAM PLOT SUMMARY ({num_images} images)")
    print("="*45)
    print(f"{'Total SAM Processing Time:':<25} {total_time:>8.2f}s")
    print(f"{'Average Time Per Image:':<25} {total_time/num_images:>8.2f}s")
    print("="*45 + "\n")



# =====================================================================
# MAIN SAM INFERENCE (ALL PLOTS)
# =====================================================================
def run_sam_phase(image_folders):
    print("\n" + "="*50)
    print(" PHASE 2: LOADING SAM AND PROCESSING ALL PLOTS")
    print("="*50)
    
    # Load SAM Model ONCE
    print("Loading SAM (this takes a few seconds)...")
    start_sam_load = time.time()
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(device=DEVICE)
    sam = torch.compile(sam)
    predictor = SamPredictor(sam)
    torch.cuda.synchronize() 
    sam_load_time = time.time() - start_sam_load
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

        start_sam_plot = time.time()

        for i, image_file in enumerate(image_files):
            image_name = os.path.basename(image_file)
            save_name = os.path.splitext(image_name)[0]
            
            if os.path.exists(os.path.join(sam_vis_folder, image_name)):
                continue
            
            # 1. Load Data
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox_path = os.path.join(bbox_folder, save_name + ".pt")
            
            if not os.path.exists(bbox_path):
                print(f"    Warning: No boxes found for {image_name}, skipping SAM.")
                continue
                
            bbox = torch.load(bbox_path).to(DEVICE) 
            if len(bbox) == 0:
                print(f"    No wheat heads detected in {image_name}")
                continue
              
            t_start_img = time.time()
            
            # 2. Image Embedding (heavy part)
            predictor.set_image(image)
            t_embed = time.time() - t_start_img
            
            # 3. Predict Masks (Batching to prevent RAM/VRAM overflow)
            t_start_pred = time.time()
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
            t_pred = time.time() - t_start_pred
            
            # 4. Saving & Visualization (Parallel CPU Saving)
            t_start_save = time.time()
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
            
            t_save = time.time() - t_start_save
            if SHOW_TIME_SAM:
                print_sam_step_report(i, len(image_files), image_name, len(bbox), t_embed, t_pred, t_save)
            total_sam_pure_time += (t_embed + t_pred + t_save)
            total_sam_images += 1
            
            # Cleanup loop to prevent VRAM overflow
            predictor.reset_image()
            torch.cuda.empty_cache()
            gc.collect()
            
        # Plot Final Summary for SAM
        sam_total_plot = time.time() - start_sam_plot
        if SHOW_TIME_SAM:
            print_sam_plot_summary(len(image_files), sam_total_plot)
        print(f"  Finished Plot: {plot_name}")
    
    # End of SAM MAIN -> Free SAM memory
    del sam, predictor
    torch.cuda.empty_cache()
    gc.collect()
    
    return total_sam_pure_time, total_sam_images
  