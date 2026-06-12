"""
run_mask_generation.py — top-level orchestrator for the mask-generation stage.

Single entry point that runs a detector (which method is chosen by cfg.method.name)
followed by the shared SAM masking phase. This is the mask-generation analog of
src/run_reconstruction.py: it sits OUTSIDE the per-method folders and dispatches to
the right detector, so adding a new detection method (e.g. sahi_yolo_sam) is one
import + one registry entry — no edits to the existing methods.

Phase 1: only yolo_sam_v1 is registered, so this behaves exactly like the old
yolo_sam_v1/main_v1.py entry point (which is kept for back-compat).
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import os
import glob
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Detectors (box producers) — pick one via cfg.method.name. SAM is shared.
from mask_generation.yolo_sam_v1.yolo_v1_pipelined import run_yolo_phase
from mask_generation.sahi_yolo_sam.sahi_yolo_pipelined import run_yolo_phase_sahi
from mask_generation.sam_v1.sam_v1_pipelined import run_sam_phase
from wheat_utils.path_utils import get_mask_generation_result_path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =====================================================================
# --- DISPATCH REGISTRY ---
# maps a method name (cfg.method.name) → the detector function to run.
# add a new detection method by importing its run-function and adding one line here.
# =====================================================================
DETECTORS = {
    "yolo_sam_v1":   run_yolo_phase,
    "sahi_yolo_sam": run_yolo_phase_sahi,
}


# =====================================================================
# --- HELPER FUNCTIONS ---
# =====================================================================

def print_hardware_status():
    """Print which device we run on (GPU name if CUDA, warning if CPU)."""
    print(f"--- Device Status ---")
    print(f"Using device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: GPU not found. Running on CPU (this will be very slow!)")
    print("-----------------------\n")


def print_final_configuration_report(cfg, total_seconds, sam_seconds, total_images, total_heads):
    """Print the boxed end-of-run summary (settings, detection strategy, dataset totals, timing).
    The detection block differs per method: yolo_sam_v1 letterboxes the whole image to one size,
    sahi_yolo_sam tiles it at native resolution (no resize) — so they need different lines."""
    minutes, seconds = divmod(total_seconds, 60)
    print("\n" + "="*50)
    print("      FINAL MASK GENERATION SUMMARY REPORT")
    print("="*50)
    # 1. Hardware & Core Settings (common to all methods)
    print(f"{'Device:':<25} {DEVICE}")
    print(f"{'Method:':<25} {cfg.method.name}")
    print(f"{'Confidence Threshold:':<25} {cfg.method.conf_threshold_good_box}")
    print(f"{'IoU Threshold (NMS):':<25} {cfg.method.iou_threshold_nms}")
    print("-" * 50)
    # 2. Detection strategy — method specific (SAHI never resizes, yolo_sam_v1 letterboxes)
    if cfg.method.name == "sahi_yolo_sam":
        print(f"{'SAHI Tile Size:':<25} {cfg.method.sahi_slice_size}px (native, no resize)")
        print(f"{'SAHI Overlap Ratio:':<25} {cfg.method.sahi_overlap_ratio}")
        print(f"{'SAHI Merge:':<25} {cfg.method.sahi_merge}/{cfg.method.sahi_match_metric} @ {cfg.method.sahi_match_threshold}")
        print(f"{'Tiles per GPU Forward:':<25} {cfg.method.sahi_tile_batch_size}")
        print(f"{'Full-image Pass:':<25} {cfg.method.sahi_full_image_pass}")
    else:
        print(f"{'YOLO Resize Size:':<25} {cfg.method.target_image_size}px")
        print(f"{'BATCH_SIZE_YOLO:':<25} {cfg.method.batch_size_yolo}")
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


@hydra.main(version_base=None, config_path="../../configs/mask_generation", config_name="config")
def main(cfg: DictConfig):
    """Orchestrate the mask-generation stage: dispatch to the chosen detector, then run shared SAM."""
    global_start_time = time.perf_counter()
    print_hardware_status()
    print(f"-> Using Dataset: {cfg.dataset.name.upper()}")
    print(f"-> Using Method:  {cfg.method.name}")
    print(f"--- Starting Segmentation ---")

    # 0. Pick the detector for this method (dispatch). SAM is always the shared phase.
    method_name = cfg.method.name
    if method_name not in DETECTORS:
        print(f"ERROR: unknown method '{method_name}'. Registered detectors: {list(DETECTORS)}")
        return
    run_detector = DETECTORS[method_name]

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

    # 2. Run the chosen detector (produces bboxes/*.pt)
    total_plot_boxes = run_detector(image_folders, cfg)

    # 3. Run SAM (shared) unless only_yolo
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
