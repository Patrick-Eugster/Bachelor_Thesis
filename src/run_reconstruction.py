import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import subprocess
import sys
import time
import datetime
import yaml

# --- INPUT DATA ---
CAMERA   = "fip"       # camera type: "fip" or "phone"
PLOT     = "plot_461"  # plot folder name inside input_plots/{camera}/
# Euler cluster path (uncomment when running on Euler):
# INPUT_BASE = "/cluster/scratch/peugste/input_plots"

INPUT_BASE   = "input_plots"
RESULTS_BASE = "results"
DATASET_PATH = os.path.join(INPUT_BASE, CAMERA, PLOT)

# --- EXPERIMENT NAMING ---
# Controls where results are saved:
#   results/reconstruction/{camera}/{plot}/vanilla_3dgs/{experiment}/
#
# EXPERIMENT_NAME options:
#   "initial"      — default name for a first/scratch run (no date appended, safe to overwrite)
#   "my_run_name"  — any custom string to identify this experiment
#   ""             — auto-generate a pure timestamp: "2025-04-28_1430"
#
# APPEND_DATE options (ignored when EXPERIMENT_NAME is "" or "initial"):
#   False          — use name as-is:         "my_run_name"
#   True           — append date to name:    "my_run_name_2025-04-28"
EXPERIMENT_NAME = "initial"  # "initial" is the default name for a first run
APPEND_DATE     = False      # True = append date to custom name (ignored for "initial" and pure timestamp)

def _resolve_experiment_name():
    """Return final experiment name based on EXPERIMENT_NAME and APPEND_DATE.

    ""        → pure timestamp "2025-04-28_1430"           (APPEND_DATE ignored)
    "initial" → "initial"                                  (APPEND_DATE ignored)
    "my_run"  + APPEND_DATE=False → "my_run"
    "my_run"  + APPEND_DATE=True  → "my_run_2025-04-28"
    """
    if not EXPERIMENT_NAME:
        # pure timestamp — APPEND_DATE ignored to avoid double date
        return datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    if EXPERIMENT_NAME == "initial":
        # fixed scratch name — never append date
        return "initial"
    if APPEND_DATE:
        return f"{EXPERIMENT_NAME}_{datetime.datetime.now().strftime('%Y-%m-%d')}"
    return EXPERIMENT_NAME

# --- TRAINING CONFIGURATION ---
DATA_DEVICE_CPU = True  # keep images in RAM instead of VRAM (recommended for 16GB GPU)

RESOLUTION              = 2       # 1 = full resolution, 2 = half (saves ~4x rasterizer VRAM), 4 = quarter
OPACITY_PRUNE_THRESHOLD = 0.005   # Gaussians below this opacity get pruned. Default: 0.005. Raise to 0.01 to save VRAM (safe for wheat)
SH_DEGREE               = 3       # spherical harmonics degree for view-dependent color. Default: 3. Set to 0 to save VRAM
DENSIFY_UNTIL_ITER      = 11000   # stop adding new Gaussians after this iteration. Default: 11000
DENSIFY_GRAD_THRESHOLD  = 0.0002  # min gradient to split a Gaussian. Default: 0.0002. Raise to 0.0003 to save VRAM

exp_name   = _resolve_experiment_name()
MODEL_PATH = os.path.join(RESULTS_BASE, "reconstruction", CAMERA, PLOT, "vanilla_3dgs", exp_name)

EXP_NAME = "run_1"  # for run_3d_seg wheat head experiment sub-folder

# --- SEGMENTATION INPUT SOURCE ---
# Which detection results to use as input for 3DGS training + instance matching:
DETECTION_EXPERIMENT = "initial"  # experiment name from results/instance_segmentation/
USE_YOLOSAM_SOURCE   = False      # True = use yolosam/ ground truth bboxes + masks (for comparison)
                                   # False = use predicted bboxes + masks from DETECTION_EXPERIMENT
_det_base      = os.path.join(RESULTS_BASE, "instance_segmentation", CAMERA, PLOT, "yolo_sam_v1", DETECTION_EXPERIMENT)
SEG_SOURCE_DIR = os.path.join(_det_base, "yolosam") if USE_YOLOSAM_SOURCE else _det_base
# auto-derived, never type manually — points to bboxes/ and masks/ used by all pipeline steps

# --- SEGMENTATION VISUALIZATION ---
SAVE_VIS_OVERLAY = True  # save overlay JPGs showing each wheat head projected onto all cameras
VIS_MAX_HEADS    = 10    # save overlays for first N heads only. 0 = all heads (~10800 files for 300 heads x 36 cameras)

# --- LOG FILE ---
LOG_FILE     = os.path.join(MODEL_PATH, "seg_logs", f"{EXP_NAME}.txt")  # terminal output saved here. Set to None to disable.
LOG_SEG_ONLY = True  # True = only log Step 4 (run_3d_seg). False = log the entire pipeline.


# --- PIPELINE STEPS (toggle on/off) ---
# Step 1: Train 3DGS model (the long one)
RUN_TRAIN = False

# Step 2: Render from original camera views
RUN_RENDER = False

# Step 3: Compute PSNR/SSIM/LPIPS quality scores
RUN_METRICS = False

# Step 4: 3D wheat head segmentation + auto-export of gaussians_colored.ply
RUN_SEG = False

# Step 5: Render 360 flyaround video
RUN_RENDER_360 = False

# Step 6: Evaluate segmentation quality — saves overlay PNGs per camera
RUN_EVAL = False

# Step 7: Launch interactive viser viewer in browser (Ctrl+C to stop)
RUN_VIEWER = True

# ===========================================
# Step 5 Render additional settings:
FAST_RENDER_360 = True # False = original per-head flashsplat renders (correct colors); True = per-Gaussian coloring (fast)
WHITE_BACKGROUND_360 = True # True = white background, False = black background
N_FRAMES    = 200  # number of frames in the flyaround video
FRAMERATE   = 20 # output video framerate (fps)
ELEVATION   = 45 # camera elevation angle in degrees

# Step 7 (Viewer) additional settings:
VIEWER_TYPE = "full" # "single" = with no wheat head coloring, "full" = with wheat head coloring
FAST_VIEWER = True  # only used when VIEWER_TYPE="full"
                    # False = eval_obj_labels per frame (300 FlashSplat renders, correct overlay colors)
                    # True = pre-bake HSV colors into Gaussians at startup, 1 render/frame (fast, flat colors)
VIEWER_PORT = 8080 # browser port — open http://localhost:VIEWER_PORT after launch


def _save_config(model_path):
    """Save all training config constants to config.yaml inside the experiment folder."""
    os.makedirs(model_path, exist_ok=True)
    config = {
        "experiment":              exp_name,
        "date":                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "camera":                  CAMERA,
        "plot":                    PLOT,
        "resolution":              RESOLUTION,
        "sh_degree":               SH_DEGREE,
        "opacity_prune_threshold": OPACITY_PRUNE_THRESHOLD,
        "densify_until_iter":      DENSIFY_UNTIL_ITER,
        "densify_grad_threshold":  DENSIFY_GRAD_THRESHOLD,
        "data_device_cpu":         DATA_DEVICE_CPU,
    }
    config_path = os.path.join(model_path, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config saved → {config_path}")


def _check_overwrite(model_path):
    """Warn and ask before overwriting an existing named experiment (skip for timestamps)."""
    if not EXPERIMENT_NAME:
        return  # timestamps are always unique, no check needed
    if EXPERIMENT_NAME == "initial":
        return  # "initial" is a scratch run, always safe to overwrite
    if os.path.exists(model_path) and os.listdir(model_path):
        print(f"\nExperiment '{EXPERIMENT_NAME}' already exists at: {model_path}")
        answer = input("Overwrite? [y/N]: ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)


def _run_pipeline():
    data_device_flag = ["--data_device", "cpu"] if DATA_DEVICE_CPU else []
    resolution_str = str(RESOLUTION)
    timings = {}

    _check_overwrite(MODEL_PATH)
    _save_config(MODEL_PATH)

    seg_dir_flag = ["--seg_dir", SEG_SOURCE_DIR]

    # Step 1: Vanilla 3DGS Training
    if RUN_TRAIN:
        run_step("1. Train", [
            "python", "src/reconstruction/vanilla_3dgs/train_vanilla_3dgs.py",
            "-s", DATASET_PATH,
            "-m", MODEL_PATH,
            "--resolution", resolution_str,
            "--eval",
            "--opacity_cull_threshold", str(OPACITY_PRUNE_THRESHOLD),
            "--sh_degree", str(SH_DEGREE),
            "--densify_until_iter", str(DENSIFY_UNTIL_ITER),
            "--densify_grad_threshold", str(DENSIFY_GRAD_THRESHOLD),
        ] + seg_dir_flag + data_device_flag, timings)

    # Step 2: Render from original training/test camera views (for quality check)
    if RUN_RENDER:
        run_step("2. Render", [
            "python", "src/reconstruction/render.py",
            "-s", DATASET_PATH,
            "-m", MODEL_PATH,
            "--resolution", resolution_str,
            "--iteration", "15000"
        ] + seg_dir_flag + data_device_flag, timings)

    # Step 3: Compute PSNR/SSIM/LPIPS quality metrics on test views
    if RUN_METRICS:
        run_step("3. Metrics", [
            "python", "src/reconstruction/metrics.py",
            "-m", MODEL_PATH
        ], timings)

    # Step 4: 3D Segmentation — assign wheat head IDs to Gaussians
    if RUN_SEG:
        seg_tee = _Tee(LOG_FILE) if LOG_FILE and LOG_SEG_ONLY else None
        if seg_tee:
            sys.stdout = seg_tee
            print(f"Logging Step 4 to: {os.path.abspath(LOG_FILE)}")
        run_step("4. Segmentation", [
            "python", "src/instance_matching/run_3d_seg.py",
            "-s", DATASET_PATH,
            "-m", MODEL_PATH,
            "--resolution", resolution_str,
            "--eval",
            "--iou_threshold", "0.5",
            "--exp_name", EXP_NAME,
            "--vis_max_heads", str(VIS_MAX_HEADS),
        ] + seg_dir_flag + ([] if SAVE_VIS_OVERLAY else ["--no_save_vis_overlay"]) + data_device_flag, timings)
        if seg_tee:
            seg_tee.close()
        # auto-export colored PLY right after segmentation — no separate toggle needed
        exp_dir = os.path.join(MODEL_PATH, "wheat-head", EXP_NAME)
        run_step("4b. Export Colored PLY", [
            "python", "src/instance_matching/export_colored_ply.py",
            "--gaussians_ply", os.path.join(exp_dir, "gaussians.ply"),
            "--labels_path",   os.path.join(exp_dir, "all_obj_labels.pth"),
            "--output_ply",    os.path.join(exp_dir, "gaussians_colored.ply"),
            "--sh_degree",     str(SH_DEGREE),
        ], timings)

    # Step 5: Render 360 flyaround video of the segmented wheat field
    if RUN_RENDER_360:
        fast_render_flag = ["--fast_render"] if FAST_RENDER_360 else []
        white_bg_flag = ["--white_background"] if WHITE_BACKGROUND_360 else []
        run_step("5. Render360", [
            "python", "src/viewer/render_360.py",
            "-s", DATASET_PATH,
            "-m", MODEL_PATH,
            "--render_type", "field",
            "--exp_name", EXP_NAME,
            "--n_frames", str(N_FRAMES),
            "--framerate", str(FRAMERATE),
            "--elevation", str(ELEVATION),
        ] + fast_render_flag + white_bg_flag + data_device_flag, timings)

    # Step 6: Evaluate 3D segmentation quality — saves overlay PNGs per camera
    if RUN_EVAL:
        run_step("6. Eval", [
            "python", "src/instance_matching/eval_wheatgs.py",
            "-s", DATASET_PATH,
            "-m", MODEL_PATH,
            "--resolution", resolution_str,
            "--exp_name", EXP_NAME,
            "--skip_train"
        ] + seg_dir_flag + data_device_flag, timings)

    # Step 7: Interactive viser viewer — open http://localhost:VIEWER_PORT in browser, Ctrl+C to stop
    if RUN_VIEWER:
        # __file__ is in reconstruction/ so go up one level to reach viewer/
        viewer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer")
        # use absolute paths — viewer runs from viewer/ so relative paths would break
        abs_model_path = os.path.abspath(MODEL_PATH)
        abs_dataset_path = os.path.abspath(DATASET_PATH)
        seg_ply = os.path.join(abs_model_path, "wheat-head", EXP_NAME, "gaussians.ply")
        train_ply = os.path.join(abs_model_path, "point_cloud", "iteration_15000", "point_cloud.ply")
        # prefer the fine-tuned step-4 model if it exists, otherwise fall back to step-1 model
        input_ply = seg_ply if os.path.exists(seg_ply) else train_ply
        if VIEWER_TYPE == "full":
            labels_path = os.path.join(abs_model_path, "wheat-head", EXP_NAME, "all_obj_labels.pth")
            fast_viewer_flag = ["--fast_render"] if FAST_VIEWER else []
            viewer_cmd = [
                "python", "wheatgs_rendering.py",
                "--input_ply", input_ply,
                "--labels_path", labels_path,
                "--colmap_path", os.path.join(abs_dataset_path, "sparse", "0"),
                "--images_path", os.path.join(abs_dataset_path, "images"),
                "--port", str(VIEWER_PORT),
            ] + fast_viewer_flag
        else:
            viewer_cmd = [
                "python", "singlewheat_rendering.py",
                "--input_ply", input_ply,
                "--port", str(VIEWER_PORT),
            ]
        print(f"  Open http://localhost:{VIEWER_PORT} in your browser")
        run_step("7. Viewer", viewer_cmd, timings, cwd=viewer_dir, allow_interrupt=True)

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

def run_command(command_list, cwd=None, allow_interrupt=False):
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
        process = subprocess.Popen(command_list, stdout=slave_fd, stderr=slave_fd, cwd=cwd)
        _os.close(slave_fd)
        buf = b""
        try:
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
        except KeyboardInterrupt:
            import signal
            process.send_signal(signal.SIGINT)
        if buf:  # flush any remaining partial line
            sys.stdout.write(buf.decode("utf-8", errors="replace"))
            sys.stdout.flush()
        _os.close(master_fd)
        try:
            process.wait()
        except KeyboardInterrupt:
            # second Ctrl+C while waiting for child to exit — just kill it
            process.kill()
            process.wait()
        returncode = process.returncode
    else:
        result = subprocess.run(command_list, cwd=cwd)
        returncode = result.returncode
    if returncode != 0:
        if allow_interrupt:
            print(f"\n>>> Viewer closed (exit code {returncode})")
        else:
            print(f"!!! ERROR: Command failed with code {returncode}")
            sys.exit(1)

def run_step(step_name, command_list, timings, cwd=None, allow_interrupt=False):
    """Run a pipeline step, print its duration, and store it in timings dict."""
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    run_command(command_list, cwd=cwd, allow_interrupt=allow_interrupt)
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

if __name__ == "__main__":
    main()
