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
ONLY_YOLO = False      # Set to False if you want to run SAM too
SKIP_YOLO = True      # Set to True to skip YOLO and use existing .pt files
LIMIT_PLOTS = 1       # How many plots to process (0 for all)
LIMIT_IMAGES = 1      # How many images per plot (0 for all)



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
    
    # to fix the "aggressive" boxes
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
        
        
        
        # --- YOLO INFERENCE (BATCHED) -----------------
        print(f"Running YOLO on {len(image_files)} images...")
        
        start_total = time.time()
        
        # Start GPU Timer
        torch.cuda.synchronize()
        start_gpu = time.time()
        
        # 1. Run inference on ALL images at once (Batching) for speed.
        results = model(image_files) # resizing is happening here, we do it as a matrix
        torch.cuda.synchronize()
        gpu_time = time.time() - start_gpu
        
        # Start Disk/Processing Timer
        start_disk = time.time()
        # 2. Iterate through the results to save them
        for i, result in enumerate(results.tolist()):
            image_path = image_files[i]
            save_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save YOLO visualization slow because of disk I/O)
            # with Labels=True to show the probability of being a wheat head
            result.save(labels=False, save_dir=yolo_vis_folder, exist_ok=True)
            
            # 3. Process the results for SAM
            # .xyxy[0] is the tensor for the i-th image
            preds = result.xyxy[0].cpu() 
            box_tensor = preds[:, :4]
            
            # Save bounding box tensor (.pt)
            torch.save(box_tensor, os.path.join(bbox_folder, save_name + ".pt"))
            
        disk_time = time.time() - start_disk
        total_time = time.time() - start_total
        print(f"\n--- PERFORMANCE REPORT ---")
        print(f"GPU Inference Time: {gpu_time:.2f}s ({gpu_time/len(image_files):.4f}s per image)")
        print(f"Disk Write/Vis Time: {disk_time:.2f}s")
        print(f"Total Time for {len(image_files)} images: {total_time:.2f}s")
        print(f"---------------------------\n")


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
                continue


        # --- SAM INFERENCE ---
        print("  Running SAM...")
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(device=DEVICE)
        predictor = SamPredictor(sam)

        for i, image_file in enumerate(image_files):
            image_name = os.path.basename(image_file)
            save_name = os.path.splitext(image_name)[0]
            
            # Skip if already done
            if os.path.exists(os.path.join(sam_vis_folder, image_name)):
                continue

            # Load Image & Boxes
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

            predictor.set_image(image)
            
            # Transform boxes for SAM
            transformed_boxes = predictor.transform.apply_boxes_torch(bbox, image.shape[:2])
            
            # Predict masks
            masks, _, _ = predictor.predict_torch(
                point_coords=None,      
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )

            # --- SAVING RESULTS ---
            # 1. Save individual masks
            objects = np.zeros((masks.size(2), masks.size(3)))
            for idx, mask in enumerate(masks.cpu().numpy()):
                # Create a composite object map (each wheat head gets an ID)
                objects[mask.squeeze()] = idx + 1
                
                # Save binary mask for 3DGS
                mask_uint8 = (mask.squeeze() * 255).astype(np.uint8)
                mask_filename = f"{save_name}_{idx:03}.png"
                Image.fromarray(mask_uint8).save(os.path.join(mask_folder, mask_filename))

            # 2. Save Visualization (Overlay)
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

            # Cleanup loop
            predictor.reset_image()
            torch.cuda.empty_cache()

        # Free SAM memory
        del sam
        del predictor
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  Finished Plot: {plot_name}")
        

if __name__ == "__main__":
    run_segmentation()
    
    
    