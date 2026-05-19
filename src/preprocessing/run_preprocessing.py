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

import os
import subprocess
import sys
import time

import hydra
from omegaconf import DictConfig, OmegaConf


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
    print(f"\n{'='*40}\n  PREPROCESSING SUMMARY\n{'='*40}")
    for name, t in timings.items():
        print(f"  {name:<25} {fmt_time(t)}")
    print(f"{'='*40}\n  {'TOTAL':<25} {fmt_time(total)}\n{'='*40}")


if __name__ == "__main__":
    main()
