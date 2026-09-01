# `scripts/` — cluster job templates

The GPU-heavy pipeline stages (3DGS training, mask generation, 3D segmentation) are
normally run as batch jobs on a SLURM cluster. This folder holds **generic, reusable
templates** for those jobs:

| Template | Stage | Conda env | Runs |
|---|---|---|---|
| `example_maskgen_job.sh` | 2 — mask generation | `wheat-maskgen` | `run_mask_generation.py` |
| `example_train_job.sh` | 3 — 3DGS reconstruction | `wheat3dgs` | `run_reconstruction.py` (train) |
| `example_seg_job.sh` | 4 — 3D segmentation | `wheat3dgs` | `run_reconstruction.py` (seg) |

## Using a template

1. Copy the template (e.g. `cp example_train_job.sh my_train_job.sh`).
2. Replace every `<PLACEHOLDER>`: `<CLUSTER_PROJECT_PATH>` (absolute path to your repo
   clone), `<YOUR_EMAIL>`, and the `FIELD` / `DATE` / `experiment_name` for your run.
3. Adjust the `#SBATCH` headers (GPU type, memory, time) and the `module load` lines to
   match your cluster — the ones shown are an ETH Euler example.
4. Submit with `sbatch my_train_job.sh`.

The pipeline itself is driven entirely by Hydra configs under `configs/` and the entry
points under `src/`; a job script just sets the environment and calls them. See the
top-level README for the full pipeline, and the per-module READMEs under `src/` for each
stage's options.

## Notes

- **Two conda envs.** Mask generation needs `wheat-maskgen` (torch ≥2.4); training and
  segmentation need `wheat3dgs` (torch 2.1.2 + the compiled CUDA submodules). Don't mix them.
- **Model weights** (YOLO/SAM checkpoints) live under `src/mask_generation/weights/` and are
  gitignored — download them separately before running mask generation.
- **Experiment names must be unique** per run, or results collide (see "Experiment Naming").
- The authors' exact per-experiment run scripts from the thesis are kept local-only (they
  hardcode a specific cluster path and account, so they aren't reusable) and are not shipped
  in this repo.
