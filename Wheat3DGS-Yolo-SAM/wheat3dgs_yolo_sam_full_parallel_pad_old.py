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


# --- CONFIGURATION ---
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

# --- SETTINGS / CONSTANTS ---
CONF_THRESHOLD = 0.25    # Minimum confidence to show a box, google colab had 0.05, the original probably 0.25
IOU_THRESHOLD = 0.45     # Maximum allowed overlap between boxes, default 0.45
CLASSES_TO_DETECT = [0]  # Only show class 0 (usually 'wheat')

# Image Resizing Algorithm (Options: Image.LANCZOS, Image.BICUBIC, Image.BILINEAR, Image.NEAREST)
RESIZE_METHOD = Image.LANCZOS
TARGET_IMAGE_SIZE = 1280 # rescaling size for the yolo model. default=640, must be a number x32

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- TEST CONTROLS ---
SHOW_TIME_YOLO = True
SHOW_DEBUG_YOLO_RESIZE = True

ONLY_YOLO = True      # Set to False if you want to run SAM too
LIMIT_PLOTS = 1       # How many plots to process for YOLO and SAM (0 for all)
LIMIT_IMAGES = 1      # How many images per plot or YOLO and SAM (0 for all)


#--- HELPER FUNCTIONS ---
# SAM-visualization functions 
# Color Generator, turns a single number into a specific rgb color
def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")
    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1) # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5 # Alternate between 0.5 and 1.0
    l = 0.5
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id == 0:  #invalid region
        return rgb 
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)
    return rgb


# for visualize color mask, basically every wheat head another color
def visualize_obj(objects):
    assert len(objects.shape) == 2
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask


# For Parallelized CPU Image Resizing, we scale the longer side down to 640px and padd the shorter one
# Loads, letterboxes (pads with gray), and caches one image.
def resize_single_image(img_path, target_size):
    img_orig = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img_orig.size
    
    # Calculate scaling factor to keep aspect ratio
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    # Resize the image keeping aspect ratio and create grey 640x640
    img_resized = img_orig.resize((new_w, new_h), resample=RESIZE_METHOD)
    canvas = Image.new('RGB', (target_size, target_size), (114, 114, 114))
    
    # Paste the resized image into the center of the grey 640x640
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    canvas.paste(img_resized, (pad_w, pad_h))
    
    # Return the canvas, the original image, AND the math used to pad/create it
    # we need that for undo the padding after the boxes are found, 
    # since we want to rescale it back to original for SAM and our yolo_vis outputs.
    pad_info = (scale, pad_w, pad_h)
    return np.array(canvas), np.array(img_orig), pad_info
  
# Reverses the letterbox math, saves .pt, and saves the high-res JPG.
def save_single_result(i, result, original_img, pad_info, original_path, bbox_folder, yolo_vis_folder):
    save_name = os.path.splitext(os.path.basename(original_path))[0]
    scale, pad_w, pad_h = pad_info
    preds = result.xyxy[0].cpu().clone() # Get YOLO predictions
    
    if len(preds) > 0:
        # 1. Subtract the gray padding from the box coordinates
        preds[:, [0, 2]] -= pad_w  # x_min, x_max
        preds[:, [1, 3]] -= pad_h  # y_min, y_max
        
        # 2. Divide by the scale to stretch back to original high-res pixels
        preds[:, :4] /= scale
        
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
    
    # out_path = os.path.join(yolo_vis_folder, f"{save_name}.png")
    # Image.fromarray(annotated_img).save(out_path)
    out_path = os.path.join(yolo_vis_folder, f"{save_name}.jpg")
    Image.fromarray(annotated_img).save(out_path, quality=90)


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
    print("\n" + "="*45)
    print(" PHASE 1: LOADING YOLO AND PROCESSING ALL PLOTS")
    print("="*45)
    
    if not os.path.exists(WHEAT_YOLO_MODEL):
        print(f"ERROR: Wheat model not found at {WHEAT_YOLO_MODEL}")
        return

    # Load YOLO Model ONCE & Load custom model using local repo
    model = torch.hub.load(YOLO_DIR, 'custom', path=WHEAT_YOLO_MODEL, source='local')
    # to fix the "aggressive" boxes or adjust as seen fit
    model.conf = CONF_THRESHOLD  
    model.iou = IOU_THRESHOLD    
    model.classes = CLASSES_TO_DETECT  

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

        # 1. Parallel Pre-Processing (Resizing & Caching)
        print(f"Parallel resizing and caching {len(image_files)} images...")
        start_prep = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # results_data contains tuples of (resized_np, original_np)
            results_data = list(executor.map(lambda p: resize_single_image(p, TARGET_IMAGE_SIZE), image_files))
        
        resized_images = [x[0] for x in results_data]
        original_images = [x[1] for x in results_data]
        pad_infos = [x[2] for x in results_data] # Extract the padding math
        prep_time = time.time() - start_prep

        # 2. GPU Inference
        torch.cuda.synchronize()
        start_gpu = time.time()
        results = model(resized_images)
        det_list = results.tolist()
        torch.cuda.synchronize()
        gpu_time = time.time() - start_gpu

        # 3. Parallel Post-Processing & Disk Save
        print(f"Parallel scaling and saving images...")
        start_disk = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Use all CPU cores to scale back and save images at once
            list(executor.map(lambda i: save_single_result(
                i, det_list[i], original_images[i], pad_infos[i], image_files[i], bbox_folder, yolo_vis_folder
            ), range(len(det_list))))
        disk_time = time.time() - start_disk

        # 4. Performance Report & Memory Clean-Up
        SHOW_DEBUG_YOLO_RESIZE and save_debug_image_yolo(resized_images, yolo_vis_folder)
        SHOW_TIME_YOLO and print_performance_report_yolo(len(image_files), prep_time, gpu_time, disk_time)
        
        del results_data, resized_images, original_images, det_list, results
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
    print("\n" + "="*45)
    print(" PHASE 2: LOADING SAM AND PROCESSING ALL PLOTS")
    print("="*45)

    # Load SAM Model ONCE
    print("Loading SAM (this takes a few seconds)...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(device=DEVICE)
    predictor = SamPredictor(sam)

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
            
            # 3. Predict Masks
            t_start_pred = time.time()
            # Transform boxes for SAM
            transformed_boxes = predictor.transform.apply_boxes_torch(bbox, image.shape[:2])
            
            # Predict masks (True/False, True means pixel is wheat, False means pixel is backgrounds)
            masks, _, _ = predictor.predict_torch(
                point_coords=None,      
                point_labels=None,      
                boxes=transformed_boxes,
                multimask_output=False
            )
            t_pred = time.time() - t_start_pred

            # 4. Saving & Visualization (Parallel CPU Saving)
            t_start_save = time.time()
            
            # Move masks to CPU and convert to uint8 once, Save binary mask for 3DGS (binary=black white mask)
            masks_np = (masks.squeeze().cpu().numpy() * 255).astype(np.uint8)
            objects = np.zeros((masks.size(2), masks.size(3)))
            save_tasks = []
            
            # Prepare the save tasks
            for idx, mask_np in enumerate(masks_np):
                objects[mask_np > 0] = idx + 1
                out_path = os.path.join(mask_folder, f"{save_name}_{idx:03}.png")
                save_tasks.append((out_path, mask_np))
                
            # Execute disk writes in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(lambda arg: cv2.imwrite(arg[0], arg[1]), save_tasks)

            # Visualization Overlay (Keep if you want to verify results)
            color_mask = visualize_obj(objects.astype(np.uint8)) / 255.0
            color_img = image / 255.0
            non_black_pixels = np.any(color_mask > 0, axis=-1)
            
            overlayed_img = color_img.copy()
            alpha = 0.5 # Opacity
            overlayed_img[non_black_pixels, :] = (alpha * color_mask[non_black_pixels, :] + 
                                                (1 - alpha) * color_img[non_black_pixels, :])
            
            cv2.imwrite( #cv2 should be faster than PIL Image.save
                os.path.join(sam_vis_folder, image_name.replace(".png", ".jpg")), 
                (overlayed_img * 255).astype(np.uint8)[:, :, ::-1] # cv2 needs BGR
            )
            
            t_save = time.time() - t_start_save
            print_sam_step_report(i, len(image_files), image_name, len(bbox), t_embed, t_pred, t_save)
            
            # Cleanup loop to prevent VRAM overflow
            predictor.reset_image()
            torch.cuda.empty_cache()
        
        # Plot Final Summary
        sam_total_plot = time.time() - start_sam_plot
        print_sam_plot_summary(len(image_files), sam_total_plot)
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