# main.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import os
import glob
import datetime
import yaml
import torch

# Import globals and logic from our newly created modules
from instance_segmentation.yolo_sam_v1.config_v1 import *
# from yolo_v1 import run_yolo_phase
from instance_segmentation.yolo_sam_v1.yolo_v1_pipelined import run_yolo_phase

# from sam_v1 import run_sam_phases
from instance_segmentation.yolo_sam_v1.sam_v1_pipelined import run_sam_phase

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
    print(f"{'Confidence Threshold:':<25} {CONF_THRESHOLD_DETECTION}")
    print(f"{'IoU Threshold (NMS):':<25} {IOU_THRESHOLD_NMS}")
    print(f"{'YOLO Resize Size:':<25} {TARGET_IMAGE_SIZE}px")
    print("-" * 50)
    # 2. Batching Strategies
    print(f"{'BATCH_SIZE_YOLO:':<25} {BATCH_SIZE_YOLO}")
    print(f"{'BATCH_SIZE_SAM_BOX:':<25} {BATCH_SIZE_SAM_BOX}")
    print(f"{'RAM_CHUNK_SIZE_YOLO:':<25} {RAM_CHUNK_SIZE_YOLO}")
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



def save_config(result_path):
    """Save all config parameters to config.yaml inside the experiment result folder."""
    os.makedirs(result_path, exist_ok=True)
    config = {
        "experiment": get_experiment_name(),
        "date":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset":    "phone" if USE_PHONE_DATA else "fip",
        "method":     "yolo_sam_v1",

        "detection_thresholds": {
            "conf_threshold_nms_floor":  CONF_THRESHOLD_NMS_FLOOR,
            "conf_threshold_detection":  CONF_THRESHOLD_DETECTION,
            "iou_threshold_nms":         IOU_THRESHOLD_NMS,
            "classes_to_detect":         CLASSES_TO_DETECT,
        },

        "image_processing": {
            "resize_method":      str(RESIZE_METHOD),
            "target_image_size":  TARGET_IMAGE_SIZE,
            "batch_size_yolo":    BATCH_SIZE_YOLO,
            "ram_chunk_size_yolo": RAM_CHUNK_SIZE_YOLO,
            "batch_size_sam_box": BATCH_SIZE_SAM_BOX,
            "max_threads":        MAX_THREADS,
        },

        "visualization": {
            "show_labels":        SHOW_LABELS,
            "show_rejected_boxes": SHOW_REJECTED_BOXES,
            "show_detected_boxes": SHOW_DETECTED_BOXES,
            "box_thickness":      BOX_THICKNESS,
            "label_font_scale":   LABEL_FONT_SCALE,
        },

        "debug_timing": {
            "debug_yolo_resize": DEBUG_YOLO_RESIZE,
            "show_time_yolo":    SHOW_TIME_YOLO,
            "show_time_sam":     SHOW_TIME_SAM,
            "show_time_total":   SHOW_TIME_TOTAL,
        },

        "run_controls": {
            "only_yolo":           ONLY_YOLO,
            "limit_plots":         LIMIT_PLOTS,
            "limit_images":        LIMIT_IMAGES,
            "only_labeled_images": ONLY_LABELED_IMAGES,
            "wandb_enabled":       WANDB_ENABLED,
        },
    }
    config_path = os.path.join(result_path, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config saved → {config_path}")


def main():
    global_start_time = time.perf_counter()
    print_hardware_status()
    print(f"--- Starting Segmentation ---")

    # 1. Find Image Folders (Do this once at the very beginning)
    # Looking for structure: input_plots/fip/plot_XXX/images/
    image_folders = sorted(glob.glob(os.path.join(INPUT_DIR, '*', 'images')))
    if LIMIT_PLOTS > 0:
        image_folders = image_folders[:LIMIT_PLOTS]

    print(f"Found {len(image_folders)} folders to process.")
    if not image_folders:
        print(f"No image folders found in {INPUT_DIR}. Check your folder structure!")
        return

    # Save config.yaml into each plot's result folder
    for folder in image_folders:
        save_config(get_result_path(folder))

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
    global_total_time = time.perf_counter() - global_start_time
    if SHOW_TIME_TOTAL:
        print_final_configuration_report(global_total_time, total_sam_pure_time, total_sam_images, total_plot_boxes)

if __name__ == "__main__":
    main()
    
