"""Orchestrator for the phone-image preprocessing pipeline.

Runs the three preprocessing steps in order, with toggles to skip any of them:
    1. preprocess_uniform_size.py — center-crop to one resolution (or symlink if already uniform)
    2. run_colmap.py             — COLMAP SfM: feature extraction → matching → mapper → undistortion
    3. compare_to_agisoft.py     — per-camera translation/rotation error vs Agisoft reference (optional)

Each step is launched as a subprocess so it can be run individually too. Hydra config:
configs/preprocessing/config.yaml.

Typical usage:
    python src/preprocessing/run_preprocessing.py field=field_D plot=20250523
    python src/preprocessing/run_preprocessing.py field=field_A plot=20250618 run_compare=true
    python src/preprocessing/run_preprocessing.py plot=20250523 run_uniform=false   # already uniform/symlinked
"""

import json
import os
import shutil
import subprocess
import sys
import time

import hydra
from omegaconf import DictConfig, OmegaConf


# Folders that COLMAP creates inside {source_path}/ — wiped when clean_before_colmap=true.
# We deliberately do NOT touch input/, input_uniform/, agisoft/, video/, logs/, or summary JSONs
# (the JSONs live under logs/ which is preserved).
_COLMAP_OUTPUT_FOLDERS = ("distorted", "sparse", "images", "stereo")


def _clean_colmap_outputs(source_path):
    """Remove COLMAP output folders so a re-run starts from scratch. Prints what was removed.
    Leaves input/, input_uniform/, agisoft/, video/, logs/ alone."""
    removed = []
    for name in _COLMAP_OUTPUT_FOLDERS:
        path = os.path.join(source_path, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
            removed.append(name)
    # also remove stale per-step summaries from previous run so we don't accidentally re-read them
    for fname in ("colmap_summary.json", "compare_summary.json"):
        p = os.path.join(source_path, "logs", fname)
        if os.path.isfile(p):
            os.remove(p)
    if removed:
        print(f"[clean] removed {source_path}/{{{','.join(removed)}}}/")
    else:
        print(f"[clean] nothing to remove in {source_path}")


def _read_summary(source_path, filename):
    """Read one of the per-step summary JSONs that each child script drops in {source_path}/logs/.
    Returns None if missing (e.g. step was skipped or crashed)."""
    path = os.path.join(source_path, "logs", filename)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def fmt_time(seconds):
    """Format seconds into h:mm:ss string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def run_step(name, cmd, timings):
    """Run one pipeline step as a subprocess; abort the pipeline if it fails."""
    print(f"\n{'='*60}\n  STEP: {name}\n{'='*60}")
    print(f">>> {' '.join(cmd)}\n")
    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0
    timings[name] = elapsed
    print(f"\n>>> {name} finished in {fmt_time(elapsed)}")
    if result.returncode != 0:
        print(f"!!! ERROR: {name} exited with code {result.returncode}. Aborting pipeline.")
        sys.exit(result.returncode)


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing/config")
def main(cfg: DictConfig):
    """Drive the three preprocessing scripts based on cfg toggles. Each step shares field=...
    and plot=... so the user only sets them once at the orchestrator level."""
    print("--- preprocessing orchestrator config ---")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------------------------------")

    common_args = [f"field={cfg.field}", f"plot={cfg.plot}"]
    timings = {}
    t_start = time.perf_counter()

    # optional cleanup: nuke previous COLMAP output before step 2 so re-runs start fresh
    if cfg.run_colmap and cfg.get("clean_before_colmap", False):
        source_path = os.path.join(cfg.dataset.input_dir, str(cfg.field), str(cfg.plot))
        print(f"\n[clean_before_colmap=true] cleaning previous COLMAP output...")
        _clean_colmap_outputs(source_path)

    if cfg.run_uniform:
        run_step("1. uniform-size", [
            "python", "src/preprocessing/preprocess_uniform_size.py", *common_args,
        ], timings)

    if cfg.run_colmap:
        run_step("2. COLMAP SfM", [
            "python", "src/preprocessing/run_colmap.py", *common_args,
        ], timings)

    if cfg.run_compare:
        run_step("3. compare to Agisoft", [
            "python", "src/preprocessing/compare_to_agisoft.py", *common_args,
        ], timings)

    total = time.perf_counter() - t_start

    # Aggregate per-step JSON summaries (each child script drops one in {source_path}/logs/)
    # str() because numeric-looking plot names (e.g. 20250523) are parsed by Hydra/YAML as int
    source_path = os.path.join(cfg.dataset.input_dir, str(cfg.field), str(cfg.plot))
    s_uniform = _read_summary(source_path, "uniform_size_summary.json") if cfg.run_uniform else None
    s_colmap  = _read_summary(source_path, "colmap_summary.json")       if cfg.run_colmap  else None
    s_compare = _read_summary(source_path, "compare_summary.json")      if cfg.run_compare else None

    # Re-print each child script's full summary block at the very end. COLMAP dumps a lot of
    # output during step 2 that buries the step-1 summary, so we reproduce all three blocks
    # back-to-back here in the same boxed format the children used.

    def _emit_uniform_block(s):
        minutes, seconds = divmod(int(s["elapsed_s"]), 60)
        print("\n" + "="*50)
        print("      UNIFORM-SIZE SUMMARY")
        print("="*50)
        print(f"{'Plot:':<28} {s['field']}/{s['plot']}")
        print(f"{'Total images:':<28} {s['n_images']}")
        print(f"{'Source sizes:':<28} {s.get('source_sizes', '-')}")
        print(f"{'Target size:':<28} {s.get('target_size', '-')}")
        print(f"{'Outcome:':<28} {s['mode']}")
        if s.get("n_cropped", 0) > 0 or s.get("n_copied", 0) > 0:
            print(f"{'Copied as-is:':<28} {s.get('n_copied', 0)}")
            print(f"{'Center-cropped:':<28} {s.get('n_cropped', 0)}")
        print("-" * 50)
        print(f"{'TOTAL TIME:':<28} {minutes}m {seconds}s  ({s['elapsed_s']:.1f}s)")
        print("="*50)

    def _emit_colmap_block(s):
        minutes, seconds = divmod(int(s["elapsed_s"]), 60)
        subs = s.get("submodels", [])
        subs_str = ", ".join(f"{x['name']}={x['images']}" for x in subs) if subs else "-"
        print("\n" + "="*50)
        print("      COLMAP SUMMARY")
        print("="*50)
        print(f"{'Plot:':<28} {s['field']}/{s['plot']}")
        print(f"{'Camera model:':<28} {s['camera']}  (single_camera={s['single_camera']})")
        print(f"{'Matcher:':<28} {s['matcher']}")
        print(f"{'GPU enabled (SIFT+match):':<28} {s['gpu']}")
        print(f"{'Threads (feat+match+map):':<28} {s['num_threads']}")
        print("-" * 50)
        print(f"{'Input images:':<28} {s['input_images']}")
        print(f"{'Sub-models from mapper:':<28} {len(subs)}  ({subs_str})")
        print(f"{'Registered in largest:':<28} {s['registered']} / {s['input_images']}")
        if s['input_images'] > 0:
            print(f"{'Registration rate:':<28} {100.0*s['registered']/s['input_images']:.1f}%")
        print("-" * 50)
        print(f"{'TOTAL TIME:':<28} {minutes}m {seconds}s  ({s['elapsed_s']:.0f}s)")
        print("="*50)

    def _emit_compare_block(s):
        minutes, seconds = divmod(int(s["elapsed_s"]), 60)
        print("\n" + "="*50)
        print("      COMPARE_TO_AGISOFT SUMMARY")
        print("="*50)
        print(f"{'Plot:':<28} {s['field']}/{s['plot']}")
        print(f"{'Cameras ours:':<28} {s['n_ours']}")
        print(f"{'Cameras agisoft:':<28} {s['n_agisoft']}")
        print(f"{'Common (matched):':<28} {s['n_common']}")
        print("-" * 50)
        print(f"{'Umeyama scale (our→m):':<28} {s['scale']:.6f}")
        print(f"{'Mean translation err:':<28} {s['mean_trans_mm']:.2f} mm")
        print(f"{'Median translation err:':<28} {s['median_trans_mm']:.2f} mm")
        if s.get("mean_rot_deg") is not None:
            print(f"{'Mean rotation err:':<28} {s['mean_rot_deg']:.3f}°")
            print(f"{'Median rotation err:':<28} {s['median_rot_deg']:.3f}°")
        print("-" * 50)
        print(f"{'TOTAL TIME:':<28} {minutes}m {seconds}s  ({s['elapsed_s']:.1f}s)")
        print("="*50)

    print("\n" + "#"*60)
    print(f"#  RECAP — all per-step summaries re-printed below for {cfg.field}/{cfg.plot}")
    print("#"*60)

    if s_uniform: _emit_uniform_block(s_uniform)
    elif cfg.run_uniform: print("\n[uniform-size summary missing — step may have failed before writing JSON]")

    if s_colmap: _emit_colmap_block(s_colmap)
    elif cfg.run_colmap: print("\n[colmap summary missing — step may have failed before writing JSON]")

    if s_compare: _emit_compare_block(s_compare)
    elif cfg.run_compare: print("\n[compare summary missing — step may have failed before writing JSON]")

    # final pipeline-wide timing table
    print("\n" + "="*60)
    print("      PREPROCESSING PIPELINE — TIMING SUMMARY")
    print("="*60)
    print(f"{'Plot:':<28} {cfg.field}/{cfg.plot}")
    print(f"{'Source path:':<28} {source_path}")
    print("-" * 60)
    for name, t in timings.items():
        print(f"   {name:<28} {fmt_time(t)}")
    print(f"   {'TOTAL':<28} {fmt_time(total)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
