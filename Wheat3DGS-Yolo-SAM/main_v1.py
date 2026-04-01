# main.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import os
import glob
import torch

# Import globals and logic from our newly created modules
from config_v1 import *
# from yolo_v1 import run_yolo_phase
from yolo_v2 import run_yolo_phase

from sam_v1 import run_sam_phase

# =====================================================================
# --- HELPER FUNCTIONS FOR MAIN ---
# =====================================================================

# Print hardware/device status (like which GPU if avaiable)
def print_hardware_status(): 
    print(f"--- Device Status ---")
    print(f"Using device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: GPU not found. Running on CPU (this will be very slow!)")
    print("-----------------------\n")

def print_final_configuration_report(total_seconds, sam_seconds, total_images, total_heads):
    minutes, seconds = divmod(total_seconds, 60)
    print("\n" + "="*50)
    print("      FINAL SEGMENTATION SUMMARY REPORT")
    print("="*50)
    # 1. Hardware & Core Settings
    print(f"{'Device:':<25} {DEVICE}")
    print(f"{'Confidence Threshold:':<25} {CONF_THRESHOLD_GOOD_BOX}")
    print(f"{'IoU Threshold:':<25} {IOU_THRESHOLD}")
    print(f"{'YOLO Resize Size:':<25} {TARGET_IMAGE_SIZE}px")
    print("-" * 50)
    # 2. Batching Strategies
    print(f"{'BATCH_SIZE_YOLO:':<25} {BATCH_SIZE_YOLO}")
    print(f"{'BATCH_SIZE_SAM_BOX:':<25} {BATCH_SIZE_SAM_BOX}")
    print(f"{'BATCH_SIZE_RAM_FILES:':<25} {BATCH_SIZE_RAM_FILES_YOLO}")
    print("-" * 50)
    # 3. Dataset & Results
    print(f"{'Total Images Processed:':<25} {total_images}")
    print(f"{'Total Wheat Heads Found:':<25} {total_heads}")
    if total_images > 0:
        print(f"{'Average Heads Per Image:':<25} {total_heads / total_images:.1f}")
    print("-" * 50)
    # 4. Final Timing
    print(f"{'TOTAL SCRIPT RUNTIME:':<25} {int(minutes)}m {seconds:.2f}s")
    if total_images > 0:
        print(f"{'Average Time Per Image:':<25} {total_seconds / total_images:.2f}s")
        print(f"{'Avg Time (SAM Only):':<25} {sam_seconds / total_images:.2f}s")
    print("="*50 + "\n")



def main():
    global_start_time = time.time()
    print_hardware_status()
    print(f"--- Starting Segmentation ---")
    
    # 1. Find Image Folders (Do this once at the very beginning)
    # Looking for structure: data/plot_XXX/images/*.jpg
    image_folders = sorted(glob.glob(os.path.join(DATA_DIR, '*', 'images')))
    if LIMIT_PLOTS > 0: # Limit Plots if wanted
        image_folders = image_folders[:LIMIT_PLOTS]
        
    print(f"Found {len(image_folders)} folders to process.")
    if not image_folders:
        print(f"No image folders found in {DATA_DIR}. Check your folder structure!")
        return

    # 2. Run YOLO
    total_plot_boxes = run_yolo_phase(image_folders)

    # 3. Run SAM
    total_sam_pure_time = 0.0
    total_sam_images = 0
    
    if ONLY_YOLO: # --- ONLY_YOLO Stop Sign ---
        print("\n ONLY_YOLO is set to True. Stopping script before SAM phase.")
    else:
        total_sam_pure_time, total_sam_images = run_sam_phase(image_folders)

    # 4. Final Report
    global_total_time = time.time() - global_start_time
    if SHOW_TIME_TOTAL:
        print_final_configuration_report(global_total_time, total_sam_pure_time, total_sam_images, total_plot_boxes)

if __name__ == "__main__":
    main()
    
