# main.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import os
import glob
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from mask_generation.yolo_sam_v1.yolo_v1_pipelined import run_yolo_phase
from mask_generation.yolo_sam_v1.sam_v1_pipelined import run_sam_phase
from wheat_utils.path_utils import get_mask_generation_result_path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =====================================================================
# --- HELPER FUNCTIONS FOR MAIN ---
# =====================================================================

def print_hardware_status():
    print(f"--- Device Status ---")
    print(f"Using device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: GPU not found. Running on CPU (this will be very slow!)")
    print("-----------------------\n")


def print_final_configuration_report(cfg, total_seconds, sam_seconds, total_images, total_heads):
    minutes, seconds = divmod(total_seconds, 60)
    print("\n" + "="*50)
    print("      FINAL SEGMENTATION SUMMARY REPORT")
    print("="*50)
    # 1. Hardware & Core Settings
    print(f"{'Device:':<25} {DEVICE}")
    print(f"{'Confidence Threshold:':<25} {cfg.method.conf_threshold_detection}")
    print(f"{'IoU Threshold (NMS):':<25} {cfg.method.iou_threshold_nms}")
    print(f"{'YOLO Resize Size:':<25} {cfg.method.target_image_size}px")
    print("-" * 50)
    # 2. Batching Strategies
    print(f"{'BATCH_SIZE_YOLO:':<25} {cfg.method.batch_size_yolo}")
    print(f"{'BATCH_SIZE_SAM_BOX:':<25} {cfg.method.batch_size_sam_box}")
    print(f"{'RAM_CHUNK_SIZE_YOLO:':<25} {cfg.method.ram_chunk_size_yolo}")
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


def save_config(result_path, cfg):
    """Save full config to config.yaml inside the experiment result folder."""
    os.makedirs(result_path, exist_ok=True)
    config_path = os.path.join(result_path, "config.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Config saved → {config_path}")


@hydra.main(version_base=None, config_path="../../../configs", config_name="mask_generation/config")
def main(cfg: DictConfig):
    global_start_time = time.perf_counter()
    print_hardware_status()
    print(f"-> Using Dataset: {cfg.dataset.name.upper()}")
    print(f"--- Starting Segmentation ---")

    # 1. Find image folders — depth depends on dataset: fip=plot_461/images, phone=field_A/20250618/images
    image_folders = sorted(glob.glob(os.path.join(cfg.dataset.input_dir, cfg.dataset.plot_glob, 'images')))
    if cfg.limit_plots > 0:
        image_folders = image_folders[:cfg.limit_plots]

    print(f"Found {len(image_folders)} folders to process.")
    if not image_folders:
        print(f"No image folders found in {cfg.dataset.input_dir}. Check your folder structure!")
        return

    # Save config.yaml into each plot's result folder
    for folder in image_folders:
        # relpath gives "plot_461" for FIP, "field_A/20250618" for phone
        plot_name = os.path.relpath(os.path.dirname(folder), cfg.dataset.input_dir)
        save_config(get_mask_generation_result_path(cfg, plot_name), cfg)

    # 2. Run YOLO
    total_plot_boxes = run_yolo_phase(image_folders, cfg)

    # 3. Run SAM
    total_sam_pure_time = 0.0
    total_sam_images = 0

    if cfg.method.only_yolo:
        print("\n ONLY_YOLO is set to True. Stopping script before SAM phase.")
    else:
        total_sam_pure_time, total_sam_images = run_sam_phase(image_folders, cfg)

    # 4. Final Report
    global_total_time = time.perf_counter() - global_start_time
    if cfg.method.show_time_total:
        print_final_configuration_report(cfg, global_total_time, total_sam_pure_time, total_sam_images, total_plot_boxes)


if __name__ == "__main__":
    main()
