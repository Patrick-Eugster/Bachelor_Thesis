# `scripts/` — Euler job templates

Every pipeline stage is heavy on RAM and VRAM, so we run them as SLURM batch jobs on the
ETH Euler cluster. This folder holds one **template per stage**, each running the phone
thesis configuration and carrying the GPU, memory and wall time our own runs used, so
they work as they are on Euler. The scripts we actually used are kept in the private
`docs/` submodule, since they hardcode a specific Euler path and account.

Run them in order. Mask generation and 3DGS reconstruction are independent of each other,
3D segmentation needs both.

| Template | Stage | Conda env | Runs |
|---|---|---|---|
| `example_maskgen_job.sh` | 2 — mask generation | `wheat-maskgen` | `run_mask_generation.py` |
| `example_train_job.sh` | 3 — 3DGS reconstruction | `wheat3dgs` | `run_reconstruction.py` (train, render, metrics) |
| `example_seg_job.sh` | 4 — 3D segmentation | `wheat3dgs` | `run_reconstruction.py` (seg, eval) |

## Using a template

1. Copy the template (e.g. `cp example_train_job.sh my_train_job.sh`).
2. Replace every `<PLACEHOLDER>`: `<EULER_PROJECT_PATH>` (absolute path to your repo
   clone), `<YOUR_EMAIL>`, and the `FIELD` / `DATE` / `experiment_name` for your run.
3. Adjust the `#SBATCH` headers (GPU type, memory, time) and the `module load` lines to
   match your own setup — the ones shown are what we used on Euler.
4. Submit with `sbatch my_train_job.sh`.

The pipeline itself is driven entirely by Hydra configs under `configs/` and the entry
points under `src/`, so a job script only sets the environment and calls them. See the
top-level README for the full pipeline, and the per-module READMEs under `src/` for each
stage's options.

## Notes

- **Two conda envs.** Mask generation needs `wheat-maskgen` (torch ≥2.4). 3DGS reconstruction
  and 3D segmentation need `wheat3dgs` (torch 2.1.2 + the compiled CUDA submodules). Don't mix them.
- **Model weights.** `wheat_head_detection_model.pt` has to be under
  `src/mask_generation/weights/` before mask generation runs. It is gitignored, see INSTALL.md.
- **Experiment names.** Mask generation overwrites an existing name without asking. 3DGS
  reconstruction refuses to start on a name that already exists, so pass `allow_overwrite=true`
  if needed. 3D segmentation is not guarded either, so a repeated
  `segmentation_3d.exp_name` replaces the previous run. The top-level README explains how the
  names chain between stages.
