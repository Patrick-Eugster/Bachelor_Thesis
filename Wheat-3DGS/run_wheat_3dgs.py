import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import subprocess
import sys
import time
import datetime

# --- CONFIGURATION ---
DATASET_PATH = "../Wheat3DGS-Yolo-SAM/data/plot_461"
EXP_NAME = "run_1"
DATA_DEVICE_CPU = True # Set True to keep images in RAM instead of VRAM (works only for some steps)

RESOLUTION = 2 # 1 = full resolution, 2 = half (saves ~4x rasterizer VRAM), 4 = quarter
OPACITY_PRUNE_THRESHOLD = 0.005  # Gaussians below this opacity get pruned. Default: 0.005. Raise to 0.01 to save VRAM (safe for wheat)
SH_DEGREE = 3 # Spherical harmonics degree for view-dependent color. Default: 3. Set to 0 to save VRAM                     
DENSIFY_UNTIL_ITER = 11000 # Stop adding new Gaussians after this iteration. Default: 11000. Lower to save VRAM (less detail)
DENSIFY_GRAD_THRESHOLD = 0.0002  # Min gradient to split a Gaussian. Default: 0.0002. Raise to 0.0003 to save VRAM (slightly less detail)

# output folder is auto-named from settings so different configs never overwrite each other
_op  = str(OPACITY_PRUNE_THRESHOLD).replace('.', '_')
_dgt = str(DENSIFY_GRAD_THRESHOLD).replace('.', '_')
MODEL_PATH = f"{DATASET_PATH}/3dgs_output_res{RESOLUTION}_opt{_op}_shd{SH_DEGREE}_dui{DENSIFY_UNTIL_ITER}_dgt{_dgt}"

# --- SEGMENTATION VISUALIZATION ---
SAVE_VIS_OVERLAY = True  # Save overlay JPGs showing each wheat head projected onto all cameras
VIS_MAX_HEADS = 10 # Save overlays for first N heads only. 0 = all heads (~10800 files for 300 heads x 36 cameras)

# --- LOG FILE ---
LOG_FILE = f"{MODEL_PATH}/logs/{EXP_NAME}.txt"  # terminal output saved here. Set to None to disable.
LOG_SEG_ONLY = True  # True = only log Step 4 (run_3d_seg). False = log the entire pipeline.

# --- PIPELINE STEPS (toggle on/off) ---
RUN_TRAIN = False         # Step 1: Train 3DGS model (the long one)
RUN_RENDER = False        # Step 2: Render from original camera views
RUN_METRICS = False       # Step 3: Compute PSNR/SSIM/LPIPS quality scores
RUN_SEG = True           # Step 4: 3D wheat head segmentation
RUN_RENDER_360 = False    # Step 5: Render 360 flyaround video
RUN_EVAL = False          # Step 6: Evaluate segmentation quality (IoU)


class _Tee:
    """Writes all print() output to both the terminal and a log file simultaneously."""
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.file = open(filepath, 'w', encoding='utf-8')
        self._stdout = sys.stdout
        self.file.write(f"=== Pipeline log started {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        self.file.flush()
    def write(self, data):
        self._stdout.write(data)
        self.file.write(data)
    def flush(self):
        self._stdout.flush()
        self.file.flush()
    def isatty(self):
        return self._stdout.isatty()  # forward TTY check so wandb/rich keep colors
    def fileno(self):
        return self._stdout.fileno()  # forward file descriptor so wandb detects real terminal for OSC 8 links
    def close(self):
        sys.stdout = self._stdout
        self.file.close()


def fmt_time(seconds):
    """Format seconds into h:mm:ss string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"

def run_command(command_list):
    """Helper to run a terminal command and wait for it to finish."""
    import pty, os as _os, termios
    print(f"\n>>> RUNNING: {' '.join(command_list)}\n")
    if LOG_FILE:
        # use a PTY so the child process sees a real terminal — preserves wandb colors and OSC 8 links
        master_fd, slave_fd = pty.openpty()
        # disable ONLCR: PTYs normally translate \n -> \r\n, which breaks tqdm's \r line overwrites
        attrs = termios.tcgetattr(slave_fd)
        attrs[1] &= ~termios.ONLCR
        termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
        process = subprocess.Popen(command_list, stdout=slave_fd, stderr=slave_fd)
        _os.close(slave_fd)
        buf = b""
        while True:
            try:
                chunk = _os.read(master_fd, 4096)
            except OSError:
                break  # child closed the PTY (process exited)
            buf += chunk
            # flush complete lines immediately so output appears in real time
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("utf-8", errors="replace") + "\n"
                sys.stdout.write(text)
                sys.stdout.flush()
        if buf:  # flush any remaining partial line
            sys.stdout.write(buf.decode("utf-8", errors="replace"))
            sys.stdout.flush()
        _os.close(master_fd)
        process.wait()
        returncode = process.returncode
    else:
        result = subprocess.run(command_list)
        returncode = result.returncode
    if returncode != 0:
        print(f"!!! ERROR: Command failed with code {returncode}")
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
    tee = _Tee(LOG_FILE) if LOG_FILE and not LOG_SEG_ONLY else None
    if tee:
        sys.stdout = tee
        print(f"Logging entire pipeline to: {os.path.abspath(LOG_FILE)}")

    try:
        _run_pipeline()
    finally:
        if tee:
            tee.close()

def _run_pipeline():
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
            "--sh_degree", str(SH_DEGREE),
            "--densify_until_iter", str(DENSIFY_UNTIL_ITER),
            "--densify_grad_threshold", str(DENSIFY_GRAD_THRESHOLD),
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
        seg_tee = _Tee(LOG_FILE) if LOG_FILE and LOG_SEG_ONLY else None
        if seg_tee:
            sys.stdout = seg_tee
            print(f"Logging Step 4 to: {os.path.abspath(LOG_FILE)}")
        run_step("4. Segmentation", [
            "python", "run_3d_seg.py",
            "-s", DATASET_PATH,
            "-m", MODEL_PATH,
            "--resolution", resolution_str,
            "--eval",
            "--iou_threshold", "0.5",
            "--exp_name", EXP_NAME,
            "--vis_max_heads", str(VIS_MAX_HEADS),
        ] + ([] if SAVE_VIS_OVERLAY else ["--no_save_vis_overlay"]) + data_device_flag, timings)
        if seg_tee:
            seg_tee.close()

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
    
