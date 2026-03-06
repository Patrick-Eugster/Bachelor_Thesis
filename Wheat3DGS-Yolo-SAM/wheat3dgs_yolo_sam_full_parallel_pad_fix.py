import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import glob
import gc
import sys
import numpy as np
import cv2
import torch
import colorsys
import time
import concurrent.futures
import shutil

from PIL import Image
from segment_anything import sam_model_registry, SamPredictor


# =====================================================================
# --- CONFIGURATION ---
# =====================================================================
# Script must be run in the same folder as this file!
BASE_DIR = os.getcwd()
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
YOLO_DIR = os.path.join(BASE_DIR, "yolov5")

# --- DATASET TOGGLE ---
USE_PHONE_DATA = False  
if USE_PHONE_DATA:
    DATA_DIR = os.path.join(BASE_DIR, "data_phone")
    print("-> Target Dataset: PHONE DATA")
else:
    DATA_DIR = os.path.join(BASE_DIR, "data")
    print("-> Target Dataset: FIP PLOT DATA")

# Model Paths
WHEAT_YOLO_MODEL = os.path.join(WEIGHTS_DIR, "wheat_head_detection_model.pt")
SAM_CHECKPOINT = os.path.join(WEIGHTS_DIR, "sam_vit_h_4b8939.pth")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- SETTINGS / CONSTANTS ---
CONF_THRESHOLD = 0.25    # Minimum confidence to show a box, google colab had 0.05, the original probably 0.25
IOU_THRESHOLD = 0.45     # Maximum allowed overlap between boxes, default 0.45
CLASSES_TO_DETECT = [0]  # Only show class 0 (usually 'wheat')

# Image Resizing Algorithm (Options: Image.LANCZOS, Image.BICUBIC, Image.BILINEAR, Image.NEAREST)
RESIZE_METHOD = Image.LANCZOS
TARGET_IMAGE_SIZE = 1280 # rescaling size for the yolo model. default=640, must be a number x32
BATCH_SIZE_YOLO = 16 # protect GPU VRAM
BATCH_SIZE_SAM_BOX = 16 # fix number of boxes to process at once (otherwise RAM/VRAM wont be enough)
BATCH_SIZE_RAM_FILES_YOLO = 100  # Protects System RAM: How many images to load at once

# --- TEST CONTROLS ---
SHOW_DEBUG_YOLO_RESIZE = True
SHOW_TIME_YOLO = True
SHOW_TIME_SAM = True   
ONLY_YOLO = False      # Set to False if you want to run SAM too
LIMIT_PLOTS = 1       # How many plots to process for YOLO and SAM (0 for all)
LIMIT_IMAGES = 0      # How many images per plot or YOLO and SAM (0 for all)



# =====================================================================
#-------- HELPER FUNCTIONS --------
# =====================================================================
# Color Generator, turns a single number into a specific rgb color
# dynamically handles any ID number without a maximum limit
def id2rgb(id, max_num_obj=65535):
    if id == 0:  # invalid region / background
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


# For Parallelized CPU Image Resizing, we scale the longer side down to target_size px and add padding
def resize_single_image(img_path, target_size):
    img_orig = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img_orig.size
    
    # 1. Scale ratio (new / old) (takes the one which original_size side x or y was bigger)
    r = min(target_size / orig_w, target_size / orig_h)
    
    # 2. Compute unpadded dimensions
    new_unpad_w = int(round(orig_w * r))
    new_unpad_h = int(round(orig_h * r))
    
    # 3. Compute padding (compared to letterbox, we use auto=False to ensure all images in a batch are exactly target_size square)
    # so that we can give them as a batch to the model 
    dw = target_size - new_unpad_w
    dh = target_size - new_unpad_h
    dw /= 2 # Divide padding into 2 sides
    dh /= 2
    
    # Calculate exact top/bottom/left/right padding using YOLO's odd-pixel math
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # 4. Resize the image using PIL RESIZE_METHOD
    img_resized = img_orig.resize((new_unpad_w, new_unpad_h), resample=RESIZE_METHOD)
    
    # 5. Create the gray canvas and paste the image over it
    # Note: canvas size will exactly equal (target_size, target_size)
    # we use grey since that will give less shard edges compared to white or black
    canvas = Image.new('RGB', (target_size, target_size), (114, 114, 114))
    canvas.paste(img_resized, (left, top))
    
    # 6. Return the pad_info needed for reversing the boxes later
    pad_info = (r, left, top)
    return np.array(canvas), np.array(img_orig), pad_info

  
# Reverses the letterbox math, saves .pt, and saves the high-res JPG.
def save_single_result(i, result, original_img, pad_info, original_path, bbox_folder, yolo_vis_folder):
    save_name = os.path.splitext(os.path.basename(original_path))[0]
    r, pad_left, pad_top = pad_info # Extract math from the resize step
    
    preds = result.xyxy[0].cpu().clone() # Get YOLO predictions
    
    if len(preds) > 0:
        # 1. Subtract the gray padding from the box coordinates
        preds[:, [0, 2]] -= pad_left  # x_min, x_max
        preds[:, [1, 3]] -= pad_top   # y_min, y_max
        
        # 2. Divide by the scale to stretch back to original high-res pixels
        preds[:, :4] /= r
        
        # 3. Clip boxes to ensure they don't accidentally go outside the image boundary
        orig_h, orig_w = original_img.shape[:2]
        preds[:, [0, 2]] = preds[:, [0, 2]].clamp(0, orig_w)
        preds[:, [1, 3]] = preds[:, [1, 3]].clamp(0, orig_h)

    # Save Scaled Tensor for SAM
    torch.save(preds[:, :4], os.path.join(bbox_folder, f"{save_name}.pt"))
    
    # Render boxes on high-res and save as JPG
    result.ims = [original_img]
    result.xyxy[0] = preds.to(result.xyxy[0].device)
    annotated_img = result.render(labels=False)[0]
    
    out_path = os.path.join(yolo_vis_folder, f"{save_name}.jpg")
    Image.fromarray(annotated_img).save(out_path, quality=90)
    # out_path = os.path.join(yolo_vis_folder, f"{save_name}.png")
    # Image.fromarray(annotated_img).save(out_path)


def print_performance_report_yolo(num_images, prep_t, gpu_t, disk_t):
    comb_inf = prep_t + gpu_t
    total_t = prep_t + gpu_t + disk_t
    # Using fixed-width labels to fix the alignment issue
    print(f"\n" + "="*45)
    print(f"       PLOT PERFORMANCE REPORT ({num_images} images)")
    print("="*45)
    print(f"{'CPU Parallel Resize Time:':<26} {prep_t:>8.2f}s")
    print(f"{'Pure GPU YOLO Math Time:':<26} {gpu_t:>8.2f}s")
    print(f"{'Total GPU Inference Time:':<26} {comb_inf:>8.2f}s")
    print(f"{'Disk Write Time:':<26} {disk_t:>8.2f}s")
    print("-" * 45)
    print(f"{'TOTAL PLOT TIME:':<26} {total_t:>8.2f}s")
    print("="*45 + "\n")
    
# Debug, for checking the resized image
def save_debug_image_yolo(resized_imgs, folder):         
    if resized_imgs:
        Image.fromarray(resized_imgs[0]).save(os.path.join(folder, "DEBUG_RESIZED.jpg"))
        print(f"DEBUG: Resized sample saved to {folder}")

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

# Deletes all contents of a folder and recreates it.
def reset_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path) # Deletes the folder and everything inside
    os.makedirs(folder_path, exist_ok=True) # Recreates the empty folder

# Print hardware/device status
def print_hardware_status():
    print(f"--- Device Status ---")
    print(f"Using device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: GPU not found. Running on CPU (this will be very slow!)")
    print("-----------------------\n")




# =====================================================================
# --- SEGMENTATION PART ---
# =====================================================================
def run_segmentation():
    global_start_time = time.time()
    print_hardware_status()
    print(f"--- Starting Segmentation ---")
    
    # 0. Find Image Folders (Do this once at the very beginning)
    # Looking for structure: data/plot_XXX/images/*.jpg
    image_folders = sorted(glob.glob(os.path.join(DATA_DIR, '*', 'images')))
    
    if LIMIT_PLOTS > 0: # Limit Plots if wanted
        image_folders = image_folders[:LIMIT_PLOTS]
    
    print(f"Found {len(image_folders)} folders to process.")
    if not image_folders:
        print(f"No image folders found in {DATA_DIR}. Check your folder structure!")
        return


    # =====================================================================
    # MASTER LOOP 1: YOLO INFERENCE (ALL PLOTS)
    # =====================================================================
    print("\n" + "="*50)
    print(" PHASE 1: LOADING YOLO AND PROCESSING ALL PLOTS")
    print("="*50)
    
    if not os.path.exists(WHEAT_YOLO_MODEL):
        print(f"ERROR: Wheat model not found at {WHEAT_YOLO_MODEL}")
        return

    # Load YOLO Model ONCE & Load custom model using local repo
    model = torch.hub.load(YOLO_DIR, 'custom', path=WHEAT_YOLO_MODEL, source='local')
    # to fix the "aggressive" boxes or adjust as seen fit
    model.conf = CONF_THRESHOLD  
    model.iou = IOU_THRESHOLD    
    model.classes = CLASSES_TO_DETECT  
    # box line thickness, 2 is thin
    model.line_thickness = 2

    for folder in image_folders:
        plot_name = folder.split(os.sep)[-2] # Get parent folder name (e.g. plot_461)
        print(f"\n[YOLO Phase] Processing Plot: {plot_name}")
        
        # Setup Output Directories
        base_plot_path = os.path.dirname(folder)
        yolo_vis_folder = os.path.join(base_plot_path, "yolo_vis")
        bbox_folder = os.path.join(base_plot_path, "bboxes")
        reset_folder(yolo_vis_folder)
        reset_folder(bbox_folder)

        # Get Images
        image_files = glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg'))
        if LIMIT_IMAGES > 0: # Limit Images per plot if wanted
            image_files = image_files[:LIMIT_IMAGES]

        total_prep_time = 0.0
        total_gpu_time = 0.0
        total_disk_time = 0.0
        total_plot_boxes = 0

        # --- Chunking loop to protect RAM ---
        for chunk_start in range(0, len(image_files), BATCH_SIZE_RAM_FILES_YOLO):
            chunk_files = image_files[chunk_start : chunk_start + BATCH_SIZE_RAM_FILES_YOLO]
            print(f"  -> Processing chunk {chunk_start} to {chunk_start + len(chunk_files)} of {len(image_files)} images...")

            # 1. Parallel Pre-Processing (Resizing & Caching)
            start_prep = time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results_data = list(executor.map(lambda p: resize_single_image(p, TARGET_IMAGE_SIZE), chunk_files))
            
            resized_images = [x[0] for x in results_data]
            original_images = [x[1] for x in results_data]
            pad_infos = [x[2] for x in results_data] # Extract the padding math
            
            prep_time = time.time() - start_prep
            total_prep_time += prep_time

            # 2. GPU Inference
            torch.cuda.synchronize()
            start_gpu = time.time()
            det_list = []
            for b_idx in range(0, len(resized_images), BATCH_SIZE_YOLO):
                batch_imgs = resized_images[b_idx : b_idx + BATCH_SIZE_YOLO]
                # Process chunk and extend the master list
                batch_results = model(batch_imgs, size=TARGET_IMAGE_SIZE)
                det_list.extend(batch_results.tolist())
            torch.cuda.synchronize()
            
            gpu_time = time.time() - start_gpu
            total_gpu_time += gpu_time
            
            chunk_boxes = sum(len(det.xyxy[0]) for det in det_list)
            total_plot_boxes += chunk_boxes

            # 3. Parallel Post-Processing & Disk Save
            start_disk = time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Use all CPU cores to scale back and save images at once
                # IMPORTANT: Use chunk_files[i] here!
                list(executor.map(lambda i: save_single_result(
                    i, det_list[i], original_images[i], pad_infos[i], chunk_files[i], bbox_folder, yolo_vis_folder
                ), range(len(det_list))))
                
            disk_time = time.time() - start_disk
            total_disk_time += disk_time  # Add to total

            # Memory Clean-Up for THIS CHUNK
            if chunk_start == 0 and SHOW_DEBUG_YOLO_RESIZE:
                save_debug_image_yolo(resized_images, yolo_vis_folder)
                
            del results_data, resized_images, original_images, det_list
            gc.collect()


        # 4. Final Performance Report for the entire plot
        print(f"-> YOLO detected a total of {total_plot_boxes} wheat heads across {len(image_files)} images.")
        SHOW_TIME_YOLO and print_performance_report_yolo(len(image_files), total_prep_time, total_gpu_time, total_disk_time)
        
        torch.cuda.empty_cache()
        gc.collect()

    # --- End of Yolo Master Loop ---
    # Yolo is done: Delete the model to free up VRAM for SAM.
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # --- ONLY_YOLO Stop Sign ---
    if ONLY_YOLO:
        print(f"YOLO finished for {plot_name}.")
        print("\n ONLY_YOLO is set to True. Stopping script before SAM phase.")
        return
    
  
    # =====================================================================
    # MASTER LOOP 2: SAM INFERENCE (ALL PLOTS)
    # =====================================================================
    print("\n" + "="*50)
    print(" PHASE 2: LOADING SAM AND PROCESSING ALL PLOTS")
    print("="*50)

    # Load SAM Model ONCE
    print("Loading SAM (this takes a few seconds)...")
    start_sam_load = time.time()
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(device=DEVICE)
    predictor = SamPredictor(sam)
    torch.cuda.synchronize() 
    sam_load_time = time.time() - start_sam_load
    print(f"-> SAM Model loaded on {DEVICE} in {sam_load_time:.2f}s")

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
            
            # 3. Predict Masks (FIXED: Batching to prevent RAM/VRAM overflow)
            t_start_pred = time.time()
            transformed_boxes = predictor.transform.apply_boxes_torch(bbox, image.shape[:2])
            
            all_masks_np = []
            
            # Process boxes in chunks to save memory
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
            # FIXED: Used uint16 to allow for more than 255 wheat heads
            objects = np.zeros((masks_np.shape[1], masks_np.shape[2]), dtype=np.uint16)
            save_tasks = []
            
            # Prepare the save tasks
            for idx, mask_np in enumerate(masks_np):
                objects[mask_np > 0] = idx + 1
                out_path = os.path.join(mask_folder, f"{save_name}_{idx:03}.png")
                save_tasks.append((out_path, mask_np))
                
            # FIXED: Added list() to force the threads to actually execute and block
            with concurrent.futures.ThreadPoolExecutor() as executor:
                list(executor.map(lambda arg: cv2.imwrite(arg[0], arg[1]), save_tasks))

            # Visualization Overlay
            # visualize_obj can handle the uint16 array and return the color mask safely
            color_mask = visualize_obj(objects) / 255.0
            color_img = image / 255.0
            non_black_pixels = np.any(color_mask > 0, axis=-1)
            
            overlayed_img = color_img.copy()
            alpha = 0.5 # Opacity
            overlayed_img[non_black_pixels, :] = (alpha * color_mask[non_black_pixels, :] + 
                                                (1 - alpha) * color_img[non_black_pixels, :])
            
            cv2.imwrite( 
                os.path.join(sam_vis_folder, image_name.replace(".png", ".jpg")), 
                (overlayed_img * 255).astype(np.uint8)[:, :, ::-1] 
            )
            
            t_save = time.time() - t_start_save
            SHOW_TIME_SAM and print_sam_step_report(i, len(image_files), image_name, len(bbox), t_embed, t_pred, t_save)
            
            # Cleanup loop to prevent VRAM overflow
            predictor.reset_image()
            torch.cuda.empty_cache()
            gc.collect()
        
        # Plot Final Summary
        sam_total_plot = time.time() - start_sam_plot
        SHOW_TIME_SAM and print_sam_plot_summary(len(image_files), sam_total_plot)
        print(f"  Finished Plot: {plot_name}")

    # --- END OF SAM MASTER LOOP ---
    # Free SAM memory
    del sam, predictor
    torch.cuda.empty_cache()
    gc.collect()
    
    global_total_time = time.time() - global_start_time
    minutes, seconds = divmod(global_total_time, 60)
    print("\n" + "="*45)
    print(f" ENTIRE SEGMENTATION PROCESS COMPLETED")
    print(f" Total Script Runtime: {int(minutes)}m {seconds:.2f}s")
    print("="*45 + "\n")

if __name__ == "__main__":
    run_segmentation()
    