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

from PIL import Image
from segment_anything import sam_model_registry, SamPredictor



# --- CONFIGURATION ---
# We assume the script is run from Phase1_Segmentation folder
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")  # Where your images are
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
YOLO_DIR = os.path.join(BASE_DIR, "yolov5")

# Model Paths
WHEAT_YOLO_MODEL = os.path.join(WEIGHTS_DIR, "wheat_head_detection_model.pt")
SAM_CHECKPOINT = os.path.join(WEIGHTS_DIR, "sam_vit_h_4b8939.pth")

# --- SETTINGS / CONSTANTS ---
CONF_THRESHOLD = 0.25    # Minimum confidence to show a box, google colab had 0.05, the original probably 0.25
IOU_THRESHOLD = 0.45     # Maximum allowed overlap between boxes, default 0.45
CLASSES_TO_DETECT = [0]  # Only show class 0 (usually 'wheat')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Print hardware/device status
print(f"--- DEVICE STATUS ---")
print(f"Using device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("Warning: GPU not found. Running on CPU (this will be very slow!)")
print("-----------------------\n")

# --- TEST CONTROLS ---
SHOW_TIME_YOLO = True
SHOW_DEBUG_YOLO_RESIZE = False

ONLY_YOLO = False      # Set to False if you want to run SAM too
LIMIT_PLOTS = 1       # How many plots to process for YOLO and SAM (0 for all)
LIMIT_IMAGES = 1      # How many images per plot or YOLO and SAM (0 for all)



# SAM-visualization functions --- HELPER FUNCTIONS ---
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


# For Parallelized CPU Image Resizing this helper function is needed.
def resize_single_image(img_path, size=640):
    """Loads and resizes one image on one CPU thread."""
    img_orig = Image.open(img_path).convert('RGB')
    # Use LANCZOS (high quality) or BILINEAR (faster)
    img_resized = img_orig.resize((size, size), resample=Image.LANCZOS) 
    return np.array(img_resized), np.array(img_orig)
  
def save_single_result(i, result, original_img, original_path, bbox_folder, yolo_vis_folder):
    """Parallelized: Scales boxes, saves .pt, and saves the high-res /JPG."""
    save_name = os.path.splitext(os.path.basename(original_path))[0]
    orig_h, orig_w = original_img.shape[:2]
    
    # 1. Scale Boxes from 640 to Original
    x_scale, y_scale = orig_w / 640, orig_h / 640
    preds = result.xyxy[0].cpu()
    scaled_boxes = preds.clone()
    # every box has 4 coordinates (the corners), which need to be scaled back
    scaled_boxes[:, [0, 2]] *= x_scale
    scaled_boxes[:, [1, 3]] *= y_scale
    
    # 2. Save Scaled Tensor for SAM
    torch.save(scaled_boxes[:, :4], os.path.join(bbox_folder, f"{save_name}.pt"))
    
    # 3. Render boxes on high-res and save as PNG or JPG
    result.ims = [original_img]
    result.xyxy[0] = scaled_boxes.to(result.xyxy[0].device)
    annotated_img = result.render(labels=False)[0]
    
    # out_path = os.path.join(yolo_vis_folder, f"{save_name}.png")
    # Image.fromarray(annotated_img).save(out_path)
    out_path = os.path.join(yolo_vis_folder, f"{save_name}.jpg")
    Image.fromarray(annotated_img).save(out_path, quality=90) # Quality 90 is very high
  


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


def print_sam_step_report(idx, total_imgs, name, n_heads, t_embed, t_pred, t_save):
    """Prints a single line report for one image in the SAM process."""
    t_total = t_embed + t_pred + t_save
    print(f"  [{idx+1}/{total_imgs}] {name:<20} | "
          f"Embed: {t_embed:>5.2f}s | Pred: {t_pred:>5.2f}s | "
          f"Save: {t_save:>5.2f}s | Heads: {n_heads:>3} | Total: {t_total:>6.2f}s")

def print_sam_plot_summary(num_images, total_time):
    """Prints the final summary for the entire plot's SAM processing."""
    print(f"\n" + "="*45)
    print(f"      SAM PLOT SUMMARY ({num_images} images)")
    print("="*45)
    print(f"{'Total SAM Processing Time:':<25} {total_time:>8.2f}s")
    print(f"{'Average Time Per Image:':<25} {total_time/num_images:>8.2f}s")
    print("="*45 + "\n")



# --- SEGMENTATION PART ---
def run_segmentation():
    print(f"--- Starting Segmentation ---")
    
    # 1. Load YOLO Model
    print("Loading YOLOv5...")
    if not os.path.exists(WHEAT_YOLO_MODEL):
        print(f"ERROR: Wheat model not found at {WHEAT_YOLO_MODEL}")
        return

    # Load custom model using local repo
    model = torch.hub.load(YOLO_DIR, 'custom', path=WHEAT_YOLO_MODEL, source='local')
    
    model.conf = CONF_THRESHOLD  
    model.iou = IOU_THRESHOLD    
    model.classes = CLASSES_TO_DETECT  
    
    # 2. Find Image Folders
    # Looking for structure: data/plot_XXX/images/*.jpg
    image_folders = sorted(glob.glob(os.path.join(DATA_DIR, '*', 'images')))
    
    # --- OPTION 2 & 3: Limit Plots ---
    if LIMIT_PLOTS > 0:
        image_folders = image_folders[:LIMIT_PLOTS]
    
    print(f"Found {len(image_folders)} folders to process.")

    if not image_folders:
        print(f"No image folders found in {DATA_DIR}. Check your folder structure!")
        # Debug helper
        print(f"Looking in: {os.path.join(DATA_DIR, '*', 'images')}")
    
    
    # 3. Main Processing Loop
    for folder in image_folders:
        plot_name = folder.split(os.sep)[-2] # Get parent folder name (e.g. plot_461)
        print(f"\nProcessing Plot: {plot_name}")
        
        # Setup Output Directories
        base_plot_path = os.path.dirname(folder)
        yolo_vis_folder = os.path.join(base_plot_path, "yolo_vis")
        yolo_results_folder = os.path.join(base_plot_path, "yolo_results")
        bbox_folder = os.path.join(base_plot_path, "bboxes")
        sam_vis_folder = os.path.join(base_plot_path, "sam_vis")
        mask_folder = os.path.join(base_plot_path, "masks")

        for d in [yolo_vis_folder, yolo_results_folder, bbox_folder, sam_vis_folder, mask_folder]:
            os.makedirs(d, exist_ok=True)

        # Get Images
        image_files = glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg'))
        
        # --- OPTION 1: Limit Images per plot ---
        if LIMIT_IMAGES > 0:
            image_files = image_files[:LIMIT_IMAGES]
        
        
        # --- YOLO INFERENCE (BATCHED) with CPU Parallel Resize ---
        # --- 1. PARALLEL PRE-PROCESSING (RESIZE & CACHE) ---
        print(f"Parallel resizing and caching {len(image_files)} images...")
        start_prep = time.time()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # results_data contains tuples of (resized_np, original_np)
            results_data = list(executor.map(lambda p: resize_single_image(p, 640), image_files))
        
        resized_images = [x[0] for x in results_data]
        original_images = [x[1] for x in results_data]
        prep_time = time.time() - start_prep

        # --- 2. GPU INFERENCE ---
        torch.cuda.synchronize()
        start_gpu = time.time()
        
        # results.tolist() gives us the individual Detections objects
        results = model(resized_images)
        det_list = results.tolist()
        
        torch.cuda.synchronize()
        gpu_time = time.time() - start_gpu

        # --- 3. PARALLEL POST-PROCESSING & DISK SAVE ---
        print(f"Parallel scaling and saving images...")
        start_disk = time.time()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Use all CPU cores to scale and save images at once
            list(executor.map(lambda i: save_single_result(
                i, det_list[i], original_images[i], image_files[i], bbox_folder, yolo_vis_folder
            ), range(len(det_list))))
            
        disk_time = time.time() - start_disk

        # --- 4. PERFORMANCE REPORT ---
        SHOW_DEBUG_YOLO_RESIZE and save_debug_image_yolo(resized_images, yolo_vis_folder)
        SHOW_TIME_YOLO and print_performance_report_yolo(len(image_files), prep_time, gpu_time, disk_time)
        
        # Clean up memory to keep the 64GB RAM free for SAM
        del results_data, resized_images, original_images, det_list


        # --- MEMORY CLEANUP ---
        del results
        torch.cuda.empty_cache()
        gc.collect()

        # --- THE STOP SIGN ---
        if ONLY_YOLO:
            print(f"YOLO finished for {plot_name}.")
            
            if LIMIT_PLOTS == 1:
                import sys; sys.exit() # Hard stop after one plot
            else:
                continue # Skip SAM and move to the NEXT plot folder




        # --- SAM INFERENCE ---
        print(f"\n--- Running SAM on {len(image_files)} images ---")
        start_sam_plot = time.time()
        
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(device=DEVICE)
        predictor = SamPredictor(sam)

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
                
            bbox = torch.load(bbox_path).to(DEVICE) # loads the boxes from yolo
            
            if len(bbox) == 0:
                print(f"    No wheat heads detected in {image_name}")
                continue
              
            # Start individual image timer 
            t_start_img = time.time()

            # 2. Image Embedding (Heavy Part)
            predictor.set_image(image)
            t_embed = time.time() - t_start_img
            
            # 3. Predict Masks (Batch)
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

            # 4. Saving & Visualization
            t_start_save = time.time()
            
            # Save individual masks for 3DGS
            objects = np.zeros((masks.size(2), masks.size(3)))
            for idx, mask in enumerate(masks.cpu().numpy()):
                # Create a composite object map (each wheat head gets an ID)
                m_sq = mask.squeeze()
                objects[m_sq] = idx + 1
                
                # Save binary mask for 3DGS (binary=black white mask)
                mask_uint8 = (mask.squeeze() * 255).astype(np.uint8)
                mask_filename = f"{save_name}_{idx:03}.png"
                Image.fromarray(mask_uint8).save(os.path.join(mask_folder, mask_filename))

            # Save Visualization Overlay
            color_mask = visualize_obj(objects.astype(np.uint8)) / 255.0
            color_img = image / 255.0
            non_black_pixels = np.any(color_mask > 0, axis=-1)
            
            overlayed_img = color_img.copy()
            alpha = 0.6
            overlayed_img[non_black_pixels, :] = (alpha * color_mask[non_black_pixels, :] + 
                                                (1 - alpha) * color_img[non_black_pixels, :])
            
            Image.fromarray((overlayed_img * 255).astype(np.uint8)).save(
                os.path.join(sam_vis_folder, image_name.replace(".png", ".jpg"))
            )
            
            t_save = time.time() - t_start_save
            print_sam_step_report(i, len(image_files), image_name, len(bbox), t_embed, t_pred, t_save)

            # Cleanup loop to prevent VRAM overflow
            predictor.reset_image()
            torch.cuda.empty_cache()
        
        # Final Summary
        sam_total_plot = time.time() - start_sam_plot
        print_sam_plot_summary(len(image_files), sam_total_plot)

        # Free SAM memory
        del sam
        del predictor
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  Finished Plot: {plot_name}")

if __name__ == "__main__":
    run_segmentation()
    
    
    