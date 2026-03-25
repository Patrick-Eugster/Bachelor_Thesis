import subprocess
import os
import sys

# --- CONFIGURATION ---
DATASET_PATH = "../Wheat3DGS-Yolo-SAM/data/plot_461"
MODEL_PATH = "../Wheat3DGS-Yolo-SAM/data/plot_461/3dgs_output"
EXP_NAME = "run_1"
DATA_DEVICE_CPU = True  # Set True to keep images in RAM instead of VRAM (safer for 16GB GPU)

# --- PIPELINE STEPS (toggle on/off) ---
RUN_TRAIN = True         # Step 1: Train 3DGS model (the long one)
RUN_RENDER = True        # Step 2: Render from original camera views
RUN_METRICS = True       # Step 3: Compute PSNR/SSIM/LPIPS quality scores
RUN_SEG = True           # Step 4: 3D wheat head segmentation
RUN_RENDER_360 = True    # Step 5: Render 360 flyaround video
RUN_EVAL = True          # Step 6: Evaluate segmentation quality (IoU)


def run_command(command_list):
    """Helper to run a terminal command and wait for it to finish."""
    print(f"\n>>> RUNNING: {' '.join(command_list)}\n")
    result = subprocess.run(command_list)
    if result.returncode != 0:
        print(f"!!! ERROR: Command failed with code {result.returncode}")
        sys.exit(1)

def main():
    data_device_flag = ["--data_device", "cpu"] if DATA_DEVICE_CPU else []

    # Step 1: Vanilla 3DGS Training
    if RUN_TRAIN:
        run_command([
            "python", "train_vanilla_3dgs.py",
            "-s", DATASET_PATH,
            "-m", MODEL_PATH,
            "--resolution", "1",
            "--eval"
        ] + data_device_flag)

    # Step 2: Render from original training/test camera views (for quality check)
    if RUN_RENDER:
        run_command([
            "python", "render.py",
            "-s", DATASET_PATH,
            "-m", MODEL_PATH,
            "--iteration", "15000"
        ] + data_device_flag)

    # Step 3: Compute PSNR/SSIM/LPIPS quality metrics on test views
    if RUN_METRICS:
        run_command([
            "python", "metrics.py",
            "-m", MODEL_PATH
        ])

    # Step 4: 3D Segmentation — assign wheat head IDs to Gaussians
    if RUN_SEG:
        run_command([
            "python", "run_3d_seg.py",
            "-s", DATASET_PATH,
            "-m", MODEL_PATH,
            "--resolution", "1",
            "--eval",
            "--iou_threshold", "0.6",
            "--exp_name", EXP_NAME
        ] + data_device_flag)

    # Step 5: Render 360 flyaround video of the segmented wheat field
    if RUN_RENDER_360:
        run_command([
            "python", "render_360.py",
            "-s", DATASET_PATH,
            "-m", MODEL_PATH,
            "--render_type", "field",
            "--exp_name", EXP_NAME,
            "--n_frames", "200"
        ] + data_device_flag)

    # Step 6: Evaluate 3D segmentation quality (IoU vs SAM masks)
    if RUN_EVAL:
        run_command([
            "python", "eval_wheatgs.py",
            "-s", DATASET_PATH,
            "-m", MODEL_PATH,
            "--exp_name", EXP_NAME,
            "--skip_train"
        ] + data_device_flag)

    print("\n✅ PIPELINE FINISHED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
    
