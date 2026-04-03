import subprocess
import os
import sys
import time

# --- CONFIGURATION ---
DATASET_PATH = "../Wheat3DGS-Yolo-SAM/data/plot_461"
EXP_NAME = "run_1"
DATA_DEVICE_CPU = True   # Set True to keep images in RAM instead of VRAM (safer for 16GB GPU)
RESOLUTION = 2           # 1 = full resolution, 2 = half (saves ~4x rasterizer VRAM), 4 = quarter
OPACITY_PRUNE_THRESHOLD = 0.005  # Gaussians below this opacity get pruned. Default 0.005. Raise to 0.01 to save VRAM

# output folder is auto-named from settings so different configs never overwrite each other
MODEL_PATH = f"{DATASET_PATH}/3dgs_output_res{RESOLUTION}_op{str(OPACITY_PRUNE_THRESHOLD).replace('.', '_')}"

# --- PIPELINE STEPS (toggle on/off) ---
RUN_TRAIN = False         # Step 1: Train 3DGS model (the long one)
RUN_RENDER = False        # Step 2: Render from original camera views
RUN_METRICS = True       # Step 3: Compute PSNR/SSIM/LPIPS quality scores
RUN_SEG = False           # Step 4: 3D wheat head segmentation
RUN_RENDER_360 = False    # Step 5: Render 360 flyaround video
RUN_EVAL = False          # Step 6: Evaluate segmentation quality (IoU)



def fmt_time(seconds):
    """Format seconds into h:mm:ss string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"

def run_command(command_list):
    """Helper to run a terminal command and wait for it to finish."""
    print(f"\n>>> RUNNING: {' '.join(command_list)}\n")
    result = subprocess.run(command_list)
    if result.returncode != 0:
        print(f"!!! ERROR: Command failed with code {result.returncode}")
        sys.exit(1)

def run_step(step_name, command_list, timings):
    """Run a pipeline step, print its duration, and store it in timings dict."""
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    run_command(command_list)
    elapsed = time.perf_counter() - t0
    timings[step_name] = elapsed
    print(f"\n>>> {step_name} finished in {fmt_time(elapsed)}")
    
    


def main():
    data_device_flag = ["--data_device", "cpu"] if DATA_DEVICE_CPU else []
    resolution_str = str(RESOLUTION)
    timings = {}

    # Step 1: Vanilla 3DGS Training
    if RUN_TRAIN:
        run_step("1. Train", [
            "python", "train_vanilla_3dgs.py",
            "-s", DATASET_PATH,
            "-m", MODEL_PATH,
            "--resolution", resolution_str,
            "--eval",
            "--opacity_cull_threshold", str(OPACITY_PRUNE_THRESHOLD),
        ] + data_device_flag, timings)

    # Step 2: Render from original training/test camera views (for quality check)
    if RUN_RENDER:
        run_step("2. Render", [
            "python", "render.py",
            "-s", DATASET_PATH,
            "-m", MODEL_PATH,
            "--resolution", resolution_str,
            "--iteration", "15000"
        ] + data_device_flag, timings)

    # Step 3: Compute PSNR/SSIM/LPIPS quality metrics on test views
    if RUN_METRICS:
        run_step("3. Metrics", [
            "python", "metrics.py",
            "-m", MODEL_PATH
        ], timings)

    # Step 4: 3D Segmentation — assign wheat head IDs to Gaussians
    if RUN_SEG:
        run_step("4. Segmentation", [
            "python", "run_3d_seg.py",
            "-s", DATASET_PATH,
            "-m", MODEL_PATH,
            "--resolution", resolution_str,
            "--eval",
            "--iou_threshold", "0.6",
            "--exp_name", EXP_NAME
        ] + data_device_flag, timings)

    # Step 5: Render 360 flyaround video of the segmented wheat field
    if RUN_RENDER_360:
        run_step("5. Render360", [
            "python", "render_360.py",
            "-s", DATASET_PATH,
            "-m", MODEL_PATH,
            "--render_type", "field",
            "--exp_name", EXP_NAME,
            "--n_frames", "200"
        ] + data_device_flag, timings)

    # Step 6: Evaluate 3D segmentation quality (IoU vs SAM masks)
    if RUN_EVAL:
        run_step("6. Eval", [
            "python", "eval_wheatgs.py",
            "-s", DATASET_PATH,
            "-m", MODEL_PATH,
            "--resolution", resolution_str,
            "--exp_name", EXP_NAME,
            "--skip_train"
        ] + data_device_flag, timings)

    # summary table
    total = sum(timings.values())
    print(f"\n{'='*40}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*40}")
    for name, t in timings.items():
        print(f"  {name:<20} {fmt_time(t)}")
    print(f"{'='*40}")
    print(f"  {'TOTAL':<20} {fmt_time(total)}")
    print(f"{'='*40}")
    print("\n✅ PIPELINE FINISHED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
    
