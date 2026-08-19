import os
import subprocess
import sys
import time
import threading
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


def _fmt_mib(mib):
    """MiB -> 'X.XG' string for the report table, or '-' when unmeasured (None/0)."""
    if not mib:
        return "-"
    return f"{mib / 1024:.1f}G"


def _descendant_rss_mib(root_pid):
    """Sum RSS (MiB) of root_pid's descendants (NOT root itself), read from /proc. 0 if unavailable.
    During a pipeline step the orchestrator's only descendants are that step's subprocess tree, so
    this is the STEP's RAM — and it drops back when the step's process exits (unlike cgroup memory,
    which keeps sticky page cache and would bleed across steps)."""
    try:
        page = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        page = 4096
    children, rss = {}, {}
    try:
        pids = [e for e in os.listdir("/proc") if e.isdigit()]
    except OSError:
        return 0.0
    for entry in pids:
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/stat") as f:
                data = f.read()
            # comm (field 2) can contain spaces/parens → parse after the last ')'
            fields = data[data.rfind(")") + 2:].split()
            ppid = int(fields[1])                 # ppid is field 4 = index 1 after state
            with open(f"/proc/{pid}/statm") as f:
                resident_pages = int(f.read().split()[1])
        except (FileNotFoundError, ProcessLookupError, IndexError, ValueError, OSError):
            continue
        children.setdefault(ppid, []).append(pid)
        rss[pid] = resident_pages * page
    # DFS from root's children (exclude the orchestrator itself)
    total, seen, stack = 0, set(), list(children.get(root_pid, []))
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += rss.get(pid, 0)
        stack.extend(children.get(pid, []))
    return total / (1024 * 1024)


class _ResourceSampler:
    """Background thread that polls PEAK GPU VRAM + PEAK RAM of the currently-running step, so the
    run-report can show per-step (Train vs Render vs Metrics vs Seg …) peaks instead of one global
    number that the biggest step masks. Each pipeline step runs as its OWN subprocess (fresh CUDA
    context, freed on exit), so windowing the sampler around one step gives that step's true peak.
      VRAM = nvidia-smi memory.used (whole GPU — accurate per-step on an EXCLUSIVE GPU, e.g. Euler
             --gpus=rtx_4090:1; on a shared GPU it would include other users). None if no nvidia-smi.
      RAM  = summed RSS of this process's descendant tree = the step's subprocess (see above)."""
    def __init__(self, interval=1.0):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.vram_peak_mib = None      # None = no GPU / nvidia-smi missing
        self.ram_peak_mib = 0.0
        self._root = os.getpid()

    def _gpu_used_mib(self):
        """Whole-GPU used memory in MiB via nvidia-smi, or None if unavailable."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            vals = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
            return max(vals) if vals else None
        except Exception:
            return None

    def _sample(self):
        """Take one VRAM + RAM reading and fold it into the running peaks."""
        g = self._gpu_used_mib()
        if g is not None:
            self.vram_peak_mib = g if self.vram_peak_mib is None else max(self.vram_peak_mib, g)
        self.ram_peak_mib = max(self.ram_peak_mib, _descendant_rss_mib(self._root))

    def _run(self):
        self._sample()   # one immediate sample so even a very short step gets a number
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self):
        """Spawn the daemon sampling thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop sampling and return (vram_peak_mib, ram_peak_mib)."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 5)
        return self.vram_peak_mib, self.ram_peak_mib


def _git_state():
    """Read-only git fingerprint of the code that produced this run: short commit SHA + a dirty flag
    for uncommitted edits. Purely `git rev-parse`/`git status` — never commits or pushes anything.
    Returns {'commit': <sha or None>, 'dirty': <bool or None>} so results map to exact code.

    Euler fallback: the cluster rsync excludes .git/, so `git rev-parse` finds no repo there. If the
    push/job script exports WHEAT_GIT_COMMIT (from `git rev-parse` on the local side), use it so Euler
    runs still record the commit. WHEAT_GIT_DIRTY (0/1) optionally carries the local dirty flag."""
    env_sha = os.environ.get("WHEAT_GIT_COMMIT")
    import subprocess
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (src/..)
        sha = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if sha.returncode != 0:
            raise RuntimeError("no git repo here")
        st = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                            capture_output=True, text=True, timeout=5)
        return {"commit": sha.stdout.strip(), "dirty": bool(st.stdout.strip())}
    except Exception:
        # no local .git (e.g. Euler) → fall back to the env var the push script may have set
        if env_sha:
            dirty_env = os.environ.get("WHEAT_GIT_DIRTY")
            dirty = bool(int(dirty_env)) if dirty_env in ("0", "1") else None
            return {"commit": env_sha.strip(), "dirty": dirty}
        return {"commit": None, "dirty": None}


def _save_config(model_path, cfg):
    """Save full config to config.yaml inside the experiment folder, plus runtime state the Hydra
    config does NOT capture (the renderer is an env var, not a config key → otherwise unrecoverable)."""
    os.makedirs(model_path, exist_ok=True)
    config_path = os.path.join(model_path, "config.yaml")
    git = _git_state()
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
        # --- runtime state NOT part of the Hydra config (appended as valid top-level YAML keys) ---
        # WHEAT_RENDERER (gsplat vs diffgs) is chosen by env var, so record it or the engine is lost.
        f.write("\n# --- runtime (not part of the Hydra config) ---\n")
        f.write(f"wheat_renderer: {os.environ.get('WHEAT_RENDERER', 'gsplat')}\n")
        f.write(f"git_commit: {git['commit']}\n")
        f.write(f"git_dirty: {git['dirty']}\n")
    print(f"Config saved → {config_path}  (renderer={os.environ.get('WHEAT_RENDERER', 'gsplat')}, "
          f"commit={git['commit']}{'*dirty' if git['dirty'] else ''})")


def _check_overwrite(model_path, cfg):
    """REFUSE to overwrite an existing named experiment unless allow_overwrite=true.

    Every experiment result must physically survive, so a re-run with a name that already exists is
    a hard error by default — in BOTH interactive and non-interactive (SLURM/piped) mode (the old
    behaviour silently overwrote on SLURM, which could destroy a finished run). To overwrite on
    purpose pass allow_overwrite=true. Timestamp names ('') and the 'initial' scratch name are exempt
    (they're meant to be transient)."""
    if not cfg.experiment_name:
        return  # timestamps are always unique, no check needed
    if cfg.experiment_name == "initial":
        return  # "initial" is a scratch run, always safe to overwrite
    if os.path.exists(model_path) and os.listdir(model_path):
        if cfg.get("allow_overwrite", False):
            print(f"\nExperiment '{cfg.experiment_name}' exists at {model_path} — "
                  f"allow_overwrite=true → overwriting on purpose.")
            return
        print(f"\n✗ REFUSING to overwrite existing experiment '{cfg.experiment_name}' at:\n    {model_path}")
        print("  → pick a new experiment_name, or pass allow_overwrite=true to overwrite deliberately.")
        sys.exit(1)


def run_command(command_list, log_file, cwd=None, capture=None):
    """Run a terminal command, wait for it, and RETURN its exit code (no sys.exit).
    If `capture` (a list) is passed, the last lines of the child's output are
    appended to it so the run-report can show a failed step's error tail."""
    import pty, os as _os, termios
    from collections import deque
    print(f"\n>>> RUNNING: {' '.join(command_list)}\n")
    tail = deque(maxlen=80)  # ring buffer — only the end matters for the report
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
                    text = line.decode("utf-8", errors="replace")
                    sys.stdout.write(text + "\n")
                    sys.stdout.flush()
                    if capture is not None:
                        tail.append(text)
        except KeyboardInterrupt:
            import signal
            process.send_signal(signal.SIGINT)
        if buf:  # flush any remaining partial line (e.g. a crash's last line with no trailing \n)
            text = buf.decode("utf-8", errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            if capture is not None and text.strip():
                tail.append(text)
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
    if capture is not None:
        capture.extend(tail)
    return returncode


class RunContext:
    """Collects per-step status / timing / output tails across the pipeline so we
    can (a) skip steps whose dependency failed and (b) write the end-of-run report."""
    def __init__(self):
        self.records = []   # ordered list of per-step dicts (for the report)
        self.status  = {}   # step key -> "ok" / "failed" / "skipped"
        self.timings = {}   # step name -> seconds

    def blocked_by(self, depends_on):
        """Return the dependency key that blocks this step (it FAILED or was SKIPPED
        this run), or None. A dependency that was toggled off and never ran is NOT in
        status, so it's assumed to exist on disk → not blocking (keeps run_seg=true alone working)."""
        if depends_on is None:
            return None
        st = self.status.get(depends_on)
        if st is not None and st != "ok":
            return depends_on
        return None

    def record(self, key, name, status, seconds, blocked_by, tail, vram_mib=None, ram_mib=None):
        """Store one step's outcome. tqdm lines are collapsed to their final \\r-segment.
        vram_mib/ram_mib = this step's peak GPU/RAM (None when not measured, e.g. skipped)."""
        self.status[key] = status
        self.records.append({
            "key": key, "name": name, "status": status,
            "seconds": round(seconds, 2), "blocked_by": blocked_by,
            "vram_peak_mib": (round(vram_mib) if vram_mib else None),
            "ram_peak_mib": (round(ram_mib) if ram_mib else None),
            "output_tail": [t.split("\r")[-1] for t in tail],
        })


def run_step(ctx, key, step_name, command_list, log_file, depends_on=None, cwd=None, allow_interrupt=False):
    """Run one pipeline step in a dependency-aware way and record its outcome in ctx.
    If a step it depends on failed this run, this step is SKIPPED (not executed).
    Otherwise it runs and is marked OK or FAILED — the pipeline keeps going either way,
    so an independent step crashing (e.g. eval) no longer aborts the whole run."""
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    print(f"{'='*60}")

    blocker = ctx.blocked_by(depends_on)
    if blocker:
        print(f">>> SKIPPED — depends on '{blocker}' which did not succeed this run.")
        ctx.record(key, step_name, "skipped", 0.0, blocked_by=blocker, tail=[])
        return

    tail = []
    sampler = _ResourceSampler().start()   # per-step peak VRAM + RAM (this step's own subprocess)
    t0 = time.perf_counter()
    rc = run_command(command_list, log_file, cwd=cwd, capture=tail)
    elapsed = time.perf_counter() - t0
    vram_mib, ram_mib = sampler.stop()
    ctx.timings[step_name] = elapsed

    peak_str = f" | peak VRAM {_fmt_mib(vram_mib)} / RAM {_fmt_mib(ram_mib)}"
    # allow_interrupt (viewer): a non-zero exit just means the user closed it → treat as ok
    if rc == 0 or allow_interrupt:
        if allow_interrupt and rc != 0:
            print(f"\n>>> {step_name} closed (exit code {rc})")
        else:
            print(f"\n>>> {step_name} finished in {fmt_time(elapsed)}{peak_str}")
        ctx.record(key, step_name, "ok", elapsed, blocked_by=None, tail=tail,
                   vram_mib=vram_mib, ram_mib=ram_mib)
    else:
        print(f"\n>>> !!! {step_name} FAILED (exit code {rc}) after {fmt_time(elapsed)}{peak_str} "
              f"— continuing; any steps that depend on it will be skipped.")
        ctx.record(key, step_name, "failed", elapsed, blocked_by=None, tail=tail,
                   vram_mib=vram_mib, ram_mib=ram_mib)


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


_STATUS_ICON = {"ok": "OK  ", "failed": "FAIL", "skipped": "SKIP"}


def _print_and_write_report(ctx, model_path, cfg, exp_name):
    """Print the end-of-run summary and persist it as run_report.txt + run_report.json
    inside the experiment folder (so it rsyncs back). Includes per-step status, the
    per-step time table + total, and the error tail of any FAILED step. If the env var
    WHEAT_RUN_REPORT points to a file, the same report is also appended there — so one
    sbatch run (looping over many plots) ends up with a single combined report file."""
    total_seconds = sum(ctx.timings.values())
    failed  = [r for r in ctx.records if r["status"] == "failed"]
    skipped = [r for r in ctx.records if r["status"] == "skipped"]
    verdict = "SUCCESS" if not failed else "COMPLETED WITH FAILURES"
    job     = os.environ.get("SLURM_JOB_ID")
    renderer = os.environ.get("WHEAT_RENDERER", "gsplat")   # engine is an env var, record it
    git      = _git_state()                                  # read-only code fingerprint (no commit)

    lines = []
    lines.append("=" * 64)
    lines.append("  RUN REPORT")
    lines.append("=" * 64)
    lines.append(f"  when        : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  experiment  : {exp_name}")
    lines.append(f"  plot        : {cfg.get('plot', '?')}")
    if cfg.get("field"):
        lines.append(f"  field/date  : {cfg.get('field')} / {cfg.get('date', '?')}")
    if job:
        lines.append(f"  slurm job   : {job}")
    lines.append(f"  renderer    : {renderer}")
    lines.append(f"  git commit  : {git['commit']}{'  (DIRTY — uncommitted edits)' if git['dirty'] else ''}")
    lines.append(f"  model_path  : {model_path}")
    lines.append("-" * 64)
    lines.append(f"  {'STEP':<24}{'STATUS':<8}{'TIME':>10}{'VRAM':>9}{'RAM':>9}")
    lines.append("-" * 64)
    for r in ctx.records:
        icon  = _STATUS_ICON.get(r["status"], "?")
        extra = f"   (blocked by {r['blocked_by']})" if r["blocked_by"] else ""
        lines.append(f"  {r['name']:<24}{icon:<8}{fmt_time(r['seconds']):>10}"
                     f"{_fmt_mib(r.get('vram_peak_mib')):>9}{_fmt_mib(r.get('ram_peak_mib')):>9}{extra}")
    lines.append("-" * 64)
    # only one step runs at a time, so the whole-job peak = the MAX step peak
    peak_vram = max((r.get("vram_peak_mib") or 0) for r in ctx.records) if ctx.records else 0
    peak_ram  = max((r.get("ram_peak_mib") or 0) for r in ctx.records) if ctx.records else 0
    lines.append(f"  {'TOTAL':<24}{'':<8}{fmt_time(total_seconds):>10}"
                 f"{_fmt_mib(peak_vram):>9}{_fmt_mib(peak_ram):>9}")
    lines.append(f"  {'(VRAM/RAM = peak per step; TOTAL = max across steps)':<40}")
    lines.append("=" * 64)
    lines.append(f"  VERDICT: {verdict}")
    if failed:
        lines.append(f"  FAILED : {', '.join(r['name'] for r in failed)}")
    if skipped:
        lines.append(f"  SKIPPED (dependency did not succeed): {', '.join(r['name'] for r in skipped)}")
    lines.append("=" * 64)

    # error tail for each failed step — this is the 'important error' part
    for r in failed:
        tail = r["output_tail"][-40:]
        lines.append("")
        lines.append(f"----- ERROR TAIL: {r['name']}  (last {len(tail)} output lines) -----")
        lines.extend("  " + l for l in tail)

    text = "\n".join(lines)
    print("\n" + text + "\n")

    # persist next to the experiment outputs (rsyncs back automatically)
    try:
        os.makedirs(model_path, exist_ok=True)
        report_txt = os.path.join(model_path, "run_report.txt")
        with open(report_txt, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        import json
        with open(os.path.join(model_path, "run_report.json"), "w", encoding="utf-8") as f:
            json.dump({
                "experiment":   exp_name,
                "plot":         cfg.get("plot"),
                "when":         datetime.datetime.now().isoformat(timespec="seconds"),
                "slurm_job_id": job,
                "renderer":     renderer,
                "git_commit":   git["commit"],
                "git_dirty":    git["dirty"],
                "use_agisoft_sfm": cfg.get("use_agisoft_sfm", False),
                "sfm_variant":  cfg.get("sfm_variant", ""),
                "verdict":      verdict,
                "total_seconds": round(total_seconds, 2),
                "steps":        ctx.records,
            }, f, indent=2)
        print(f"Run report → {report_txt}")
    except Exception as e:
        print(f"WARNING: could not write run report: {e}")

    # optional combined report for a whole sbatch run (one file across all plots)
    agg = os.environ.get("WHEAT_RUN_REPORT")
    if agg:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(agg)), exist_ok=True)
            with open(agg, "a", encoding="utf-8") as f:
                f.write(text + "\n\n")
        except Exception as e:
            print(f"WARNING: could not append to WHEAT_RUN_REPORT={agg}: {e}")

    if verdict == "SUCCESS":
        print("\nOK  PIPELINE FINISHED SUCCESSFULLY!")
    else:
        print("\n!!! PIPELINE FINISHED WITH FAILURES — see run_report.txt")


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
    # opt-in flag: when true, all stages honor the SfM cx/cy via asymmetric frustum
    pp_flag           = ["--use_principal_point"] if cfg.reconstruction.get("use_principal_point", False) else []
    # opt-in: AbsGS densification criterion (gsplat means2d.absgrad) — recovers fine wheat detail
    absgrad_flag      = ["--absgrad"] if cfg.reconstruction.get("absgrad", False) else []
    ctx               = RunContext()

    if cfg.run_train:
        _check_overwrite(model_path, cfg)
    _save_config(model_path, cfg)

    # Step 1: Vanilla 3DGS Training
    if cfg.run_train:
        run_step(ctx, "train", "1. Train", [
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
        ] + seg_dir_flag + data_device_flag + pp_flag + absgrad_flag + wandb_flag, log_file)

    # Step 2: Render from original training/test camera views (for quality check)
    if cfg.run_render:
        # wipe stale renders from any previous run — keeps next metrics step from
        # iterating leftover files that don't correspond to the current eval split
        import shutil as _shutil
        for sub in ("train", "test"):
            sub_path = os.path.join(model_path, sub)
            if os.path.isdir(sub_path):
                print(f"Clearing stale renders at {sub_path}")
                _shutil.rmtree(sub_path)
        run_step(ctx, "render", "2. Render", [
            "python", "src/reconstruction/render.py",
            "-s", dataset_path,
            "-m", model_path,
            "--resolution", resolution_str,
            "--iteration", str(cfg.reconstruction.iterations)
        ] + seg_dir_flag + data_device_flag + pp_flag, log_file, depends_on="train")

    # Step 3: Compute PSNR/SSIM/LPIPS quality metrics on test views (reads the renders from step 2)
    if cfg.run_metrics:
        run_step(ctx, "metrics", "3. Metrics", [
            "python", "src/reconstruction/metrics.py",
            "-m", model_path,
            "-s", dataset_path,   # enables the ROI + marker masked passes (COLMAP frame; Agisoft auto-skips)
        ], log_file, depends_on="render")

    # Step 4: 3D Segmentation — assign wheat head IDs to Gaussians
    if cfg.run_seg:
        seg_tee = _Tee(log_file) if log_file and cfg.log_seg_only else None
        if seg_tee:
            sys.stdout = seg_tee
            print(f"Logging Step 4 to: {os.path.abspath(log_file)}")
        run_step(ctx, "seg", "4. Segmentation", [
            "python", "src/segmentation_3d/run_3d_seg.py",
            "-s", dataset_path,
            "-m", model_path,
            "--resolution", resolution_str,
            "--eval",
            "--iou_threshold", "0.5",
            "--exp_name", cfg.segmentation_3d.exp_name,
            "--vis_max_heads", str(cfg.segmentation_3d.vis_max_heads),
        ] + seg_dir_flag + ([] if cfg.segmentation_3d.save_vis_overlay else ["--no_save_vis_overlay"])
          + ([] if cfg.segmentation_3d.use_mask_cache else ["--no_mask_cache"])
          + (["--frustum_cull"] if cfg.segmentation_3d.get("frustum_cull", False) else ["--no_frustum_cull"])
          + (["--roi_cull"] if cfg.segmentation_3d.get("roi_cull", False) else [])
          + (["--roi_buffer_m", str(cfg.segmentation_3d.get("roi_buffer_m", 0.25))])
          + (["--height_band"] if cfg.segmentation_3d.get("height_band", False) else [])
          + (["--marker_exclude"] if cfg.segmentation_3d.get("marker_exclude", False) else [])
          + (["--marker_radius_m", str(cfg.segmentation_3d.get("marker_radius_m", 0.075))])
          + (["--legacy_ground_cull"] if cfg.segmentation_3d.get("legacy_ground_cull", False) else [])
          + (["--ground_percentile", str(cfg.segmentation_3d.get("ground_percentile", 10.0))])
          + ["--seg_seed", str(cfg.segmentation_3d.seg_seed)]
          + data_device_flag + pp_flag + wandb_flag, log_file, depends_on="train")
        if seg_tee:
            seg_tee.close()
        # auto-export colored PLY right after segmentation — no separate toggle needed
        exp_dir = os.path.join(model_path, "segmentation_3d", cfg.segmentation_3d.exp_name)
        run_step(ctx, "export_ply", "4b. Export Colored PLY", [
            "python", "src/segmentation_3d/export_colored_ply.py",
            "--gaussians_ply", os.path.join(exp_dir, "gaussians.ply"),
            "--labels_path",   os.path.join(exp_dir, "all_obj_labels.pth"),
            "--output_ply",    os.path.join(exp_dir, "gaussians_colored.ply"),
            "--sh_degree",     str(cfg.reconstruction.sh_degree),
        ], log_file, depends_on="seg")

    # Step 5: Render 360 flyaround video of the segmented wheat field
    if cfg.run_render_360:
        fast_render_flag = ["--fast_render"] if cfg.fast_render_360 else []
        white_bg_flag    = ["--white_background"] if cfg.white_background_360 else []
        run_step(ctx, "render_360", "5. Render360", [
            "python", "src/viewer/render_360.py",
            "-s", dataset_path,
            "-m", model_path,
            "--render_type", "field",
            "--exp_name", cfg.segmentation_3d.exp_name,
            "--n_frames", str(cfg.n_frames),
            "--framerate", str(cfg.framerate),
            "--elevation", str(cfg.elevation),
            "--downscale", str(cfg.get("render_360_downscale", 2)),
        ] + fast_render_flag + white_bg_flag + data_device_flag + pp_flag, log_file, depends_on="seg")

    # Step 6: Evaluate 3D segmentation quality — saves overlay PNGs per camera
    if cfg.run_eval:
        run_step(ctx, "eval", "6. Eval", [
            "python", "src/segmentation_3d/eval_wheatgs.py",
            "-s", dataset_path,
            "-m", model_path,
            "--resolution", resolution_str,
            "--exp_name", cfg.segmentation_3d.exp_name,
            "--skip_train"
        ] + seg_dir_flag + data_device_flag + pp_flag, log_file, depends_on="seg")

    # Step 6b: Pixel-level 2D metrics vs manual GT masks — requires run_eval output (test/segmentation/)
    if cfg.run_eval_2d:
        run_step(ctx, "eval_2d", "6b. Eval2D", [
            "python", "src/segmentation_3d/eval_seg_2d.py",
            "-s", dataset_path,
            "-m", model_path,
            "--resolution", resolution_str,
            "--exp_name", cfg.segmentation_3d.exp_name,
        ] + data_device_flag + pp_flag, log_file, depends_on="eval")

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
        run_step(ctx, "viewer", "7. Viewer", viewer_cmd, log_file,
                 depends_on="train", cwd=viewer_dir, allow_interrupt=True)

    # end-of-run summary table + persisted report (statuses, times, error tails)
    _print_and_write_report(ctx, model_path, cfg, exp_name)


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
