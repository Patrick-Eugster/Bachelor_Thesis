import os
import subprocess
import sys
import time
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf

from wheat_utils.path_utils import (
    resolve_experiment_name,
    get_dataset_path,
    get_reconstruction_model_path,
    get_seg_source_dir,
    get_log_file,
)


# =====================================================================
# --- HELPERS ---
# =====================================================================

def fmt_time(seconds):
    """Format seconds into h:mm:ss string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def _save_config(model_path, cfg):
    """Save full config to config.yaml inside the experiment folder."""
    os.makedirs(model_path, exist_ok=True)
    config_path = os.path.join(model_path, "config.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Config saved → {config_path}")


def _check_overwrite(model_path, cfg):
    """Warn and ask before overwriting an existing named experiment (skip for timestamps)."""
    if not cfg.experiment_name:
        return  # timestamps are always unique, no check needed
    if cfg.experiment_name == "initial":
        return  # "initial" is a scratch run, always safe to overwrite
    if os.path.exists(model_path) and os.listdir(model_path):
        print(f"\nExperiment '{cfg.experiment_name}' already exists at: {model_path}")
        if not sys.stdin.isatty():
            # non-interactive (SLURM job, pipe) — overwrite automatically
            print("Non-interactive mode: overwriting.")
            return
        answer = input("Overwrite? [y/N]: ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)


def run_command(command_list, log_file, cwd=None, allow_interrupt=False):
    """Helper to run a terminal command and wait for it to finish."""
    import pty, os as _os, termios
    print(f"\n>>> RUNNING: {' '.join(command_list)}\n")
    if log_file:
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


def run_step(step_name, command_list, timings, log_file, cwd=None, allow_interrupt=False):
    """Run a pipeline step, print its duration, and store it in timings dict."""
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    run_command(command_list, log_file, cwd=cwd, allow_interrupt=allow_interrupt)
    elapsed = time.perf_counter() - t0
    timings[step_name] = elapsed
    print(f"\n>>> {step_name} finished in {fmt_time(elapsed)}")


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
        return self._stdout.isatty()
    def fileno(self):
        return self._stdout.fileno()
    def close(self):
        sys.stdout = self._stdout
        self.file.close()


# =====================================================================
# --- PIPELINE ---
# =====================================================================

def _run_pipeline(cfg):
    exp_name     = resolve_experiment_name(cfg)
    dataset_path = get_dataset_path(cfg)
    model_path   = get_reconstruction_model_path(cfg, exp_name)
    seg_source   = get_seg_source_dir(cfg)
    log_file     = get_log_file(model_path, cfg.segmentation_3d.exp_name)

    data_device_flag  = ["--data_device", "cpu"] if cfg.reconstruction.data_device_cpu else []
    wandb_flag        = ["--wandb_enabled"] if cfg.wandb_enabled else []
    resolution_str    = str(cfg.reconstruction.resolution)
    seg_dir_flag      = ["--seg_dir", seg_source]
    timings           = {}

    if cfg.run_train:
        _check_overwrite(model_path, cfg)
    _save_config(model_path, cfg)

    # Step 1: Vanilla 3DGS Training
    if cfg.run_train:
        run_step("1. Train", [
            "python", "src/reconstruction/vanilla_3dgs/train_vanilla_3dgs.py",
            "-s", dataset_path,
            "-m", model_path,
            "--resolution", resolution_str,
            "--eval",
            "--iterations", str(cfg.reconstruction.iterations),
            "--opacity_cull_threshold", str(cfg.reconstruction.opacity_prune_threshold),
            "--sh_degree", str(cfg.reconstruction.sh_degree),
            "--densify_until_iter", str(cfg.reconstruction.densify_until_iter),
            "--densify_grad_threshold", str(cfg.reconstruction.densify_grad_threshold),
        ] + seg_dir_flag + data_device_flag + wandb_flag, timings, log_file)

    # Step 2: Render from original training/test camera views (for quality check)
    if cfg.run_render:
        run_step("2. Render", [
            "python", "src/reconstruction/render.py",
            "-s", dataset_path,
            "-m", model_path,
            "--resolution", resolution_str,
            "--iteration", str(cfg.reconstruction.iterations)
        ] + seg_dir_flag + data_device_flag, timings, log_file)

    # Step 3: Compute PSNR/SSIM/LPIPS quality metrics on test views
    if cfg.run_metrics:
        run_step("3. Metrics", [
            "python", "src/reconstruction/metrics.py",
            "-m", model_path
        ], timings, log_file)

    # Step 4: 3D Segmentation — assign wheat head IDs to Gaussians
    if cfg.run_seg:
        seg_tee = _Tee(log_file) if log_file and cfg.log_seg_only else None
        if seg_tee:
            sys.stdout = seg_tee
            print(f"Logging Step 4 to: {os.path.abspath(log_file)}")
        run_step("4. Segmentation", [
            "python", "src/segmentation_3d/run_3d_seg.py",
            "-s", dataset_path,
            "-m", model_path,
            "--resolution", resolution_str,
            "--eval",
            "--iou_threshold", "0.5",
            "--exp_name", cfg.segmentation_3d.exp_name,
            "--vis_max_heads", str(cfg.segmentation_3d.vis_max_heads),
        ] + seg_dir_flag + ([] if cfg.segmentation_3d.save_vis_overlay else ["--no_save_vis_overlay"]) + data_device_flag + wandb_flag, timings, log_file)
        if seg_tee:
            seg_tee.close()
        # auto-export colored PLY right after segmentation — no separate toggle needed
        exp_dir = os.path.join(model_path, "segmentation_3d", cfg.segmentation_3d.exp_name)
        run_step("4b. Export Colored PLY", [
            "python", "src/segmentation_3d/export_colored_ply.py",
            "--gaussians_ply", os.path.join(exp_dir, "gaussians.ply"),
            "--labels_path",   os.path.join(exp_dir, "all_obj_labels.pth"),
            "--output_ply",    os.path.join(exp_dir, "gaussians_colored.ply"),
            "--sh_degree",     str(cfg.reconstruction.sh_degree),
        ], timings, log_file)

    # Step 5: Render 360 flyaround video of the segmented wheat field
    if cfg.run_render_360:
        fast_render_flag = ["--fast_render"] if cfg.fast_render_360 else []
        white_bg_flag    = ["--white_background"] if cfg.white_background_360 else []
        run_step("5. Render360", [
            "python", "src/viewer/render_360.py",
            "-s", dataset_path,
            "-m", model_path,
            "--render_type", "field",
            "--exp_name", cfg.segmentation_3d.exp_name,
            "--n_frames", str(cfg.n_frames),
            "--framerate", str(cfg.framerate),
            "--elevation", str(cfg.elevation),
        ] + fast_render_flag + white_bg_flag + data_device_flag, timings, log_file)

    # Step 6: Evaluate 3D segmentation quality — saves overlay PNGs per camera
    if cfg.run_eval:
        run_step("6. Eval", [
            "python", "src/segmentation_3d/eval_wheatgs.py",
            "-s", dataset_path,
            "-m", model_path,
            "--resolution", resolution_str,
            "--exp_name", cfg.segmentation_3d.exp_name,
            "--skip_train"
        ] + seg_dir_flag + data_device_flag, timings, log_file)

    # Step 6b: Pixel-level 2D metrics vs manual GT masks — requires run_eval output (test/segmentation/)
    if cfg.run_eval_2d:
        run_step("6b. Eval2D", [
            "python", "src/segmentation_3d/eval_seg_2d.py",
            "-s", dataset_path,
            "-m", model_path,
            "--resolution", resolution_str,
            "--exp_name", cfg.segmentation_3d.exp_name,
        ] + data_device_flag, timings, log_file)

    # Step 7: Interactive viser viewer — open http://localhost:VIEWER_PORT in browser, Ctrl+C to stop
    if cfg.run_viewer:
        viewer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer")
        abs_model_path   = os.path.abspath(model_path)
        abs_dataset_path = os.path.abspath(dataset_path)
        seg_ply   = os.path.join(abs_model_path, "segmentation_3d", cfg.segmentation_3d.exp_name, "gaussians.ply")
        train_ply = os.path.join(abs_model_path, "point_cloud", f"iteration_{cfg.reconstruction.iterations}", "point_cloud.ply")
        # prefer the fine-tuned step-4 model if it exists, otherwise fall back to step-1 model
        input_ply = seg_ply if os.path.exists(seg_ply) else train_ply
        if cfg.viewer_type == "full":
            labels_path = os.path.join(abs_model_path, "segmentation_3d", cfg.segmentation_3d.exp_name, "all_obj_labels.pth")
            fast_viewer_flag = ["--fast_render"] if cfg.fast_viewer else []
            viewer_cmd = [
                "python", "wheatgs_rendering.py",
                "--input_ply", input_ply,
                "--labels_path", labels_path,
                "--colmap_path", os.path.join(abs_dataset_path, "sparse", "0"),
                "--images_path", os.path.join(abs_dataset_path, "images"),
                "--port", str(cfg.viewer_port),
                "--sh_degree", str(cfg.reconstruction.sh_degree),
            ] + fast_viewer_flag
        else:
            viewer_cmd = [
                "python", "singlewheat_rendering.py",
                "--input_ply", input_ply,
                "--port", str(cfg.viewer_port),
            ]
        print(f"  Open http://localhost:{cfg.viewer_port} in your browser")
        run_step("7. Viewer", viewer_cmd, timings, log_file, cwd=viewer_dir, allow_interrupt=True)

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


@hydra.main(version_base=None, config_path="../configs", config_name="reconstruction_seg3d/config")
def main(cfg: DictConfig):
    log_file    = get_log_file(get_reconstruction_model_path(cfg, resolve_experiment_name(cfg)), cfg.segmentation_3d.exp_name)
    tee = _Tee(log_file) if log_file and not cfg.log_seg_only else None
    if tee:
        sys.stdout = tee
        print(f"Logging entire pipeline to: {os.path.abspath(log_file)}")

    try:
        _run_pipeline(cfg)
    finally:
        if tee:
            tee.close()


if __name__ == "__main__":
    main()
