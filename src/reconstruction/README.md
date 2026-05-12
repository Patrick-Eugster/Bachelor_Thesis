# 3DGS Reconstruction — `src/reconstruction/`

## Overview

Trains a 3D Gaussian Splatting model of a wheat plot from multi-view images and COLMAP camera calibration. The trained model is the input for 3D segmentation (step 4).

Entry point: `src/run_reconstruction.py` — configured via `configs/reconstruction_seg3d/config.yaml`.

---

## How to Run

Run from the workspace root:

```bash
python src/run_reconstruction.py
```

All pipeline steps are **off by default** — enable the ones you want in `configs/reconstruction_seg3d/config.yaml` or pass them on the CLI:

```bash
python src/run_reconstruction.py run_train=true
python src/run_reconstruction.py run_train=true run_render=true run_metrics=true
python src/run_reconstruction.py dataset=phone plot=phone01 run_train=true experiment_name=phone_test
```

---

## Experiment Names

Experiment names are split across two config files:

`configs/reconstruction_seg3d/config.yaml`:
```yaml
plot: "plot_461"             # which plot to process
experiment_name: "initial"   # name for the 3DGS training output folder
prepend_date: false          # prepends today's date: "2025-04-28_initial"
```

`configs/reconstruction_seg3d/segmentation_3d/default.yaml`:
```yaml
detection_experiment: "initial"  # which yolo_sam detection run to read masks from
exp_name: "run_1"                # name for the segmentation subfolder
```

- **`experiment_name`** — controls the top-level output folder `results/reconstruction/fip/{plot}/vanilla_3dgs/{experiment_name}/`. All training, render, and metrics outputs go here. Named experiments (not `"initial"`) warn before overwriting.
- **`detection_experiment`** — points to the mask generation run whose `bboxes/` and `masks/` are used as input for training and segmentation. Must match the folder name in `results/mask_generation/`.
- **`exp_name`** — controls the segmentation subfolder `segmentation_3d/{exp_name}/`. Lets you re-run segmentation with different settings on the same trained model without retraining.

---

## Pipeline Steps

All steps are independent toggles. Run them in order, but you can re-run any individual step without repeating earlier ones.

| Step | Toggle | Script | What it does |
|------|--------|--------|--------------|
| 1 | `run_train` | `vanilla_3dgs/train_vanilla_3dgs.py` | Train 3DGS model — 15000 iterations, L1 + SSIM loss |
| 2 | `run_render` | `render.py` | Render from training + test camera positions for visual quality check |
| 3 | `run_metrics` | `metrics.py` | Compute PSNR / SSIM / LPIPS on test views → `results.json` |
| 4 | `run_seg` | `segmentation_3d/run_3d_seg.py` | Assign 3D wheat head IDs to Gaussians (see segmentation README) |
| 4b | auto after 4 | `segmentation_3d/export_colored_ply.py` | Bake per-head HSV colors into `gaussians_colored.ply` |
| 5 | `run_render_360` | `viewer/render_360.py` | Render 360° flyaround video → `wheat_field_360.mp4` |
| 6 | `run_eval` | `segmentation_3d/eval_wheatgs.py` | Evaluate 3D segmentation quality vs SAM masks |
| 7 | `run_viewer` | `viewer/wheatgs_rendering.py` | Open interactive viser viewer at `http://localhost:8080` |

---

## Key Training Parameters

Set in `configs/reconstruction_seg3d/reconstruction/vanilla_3dgs.yaml`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `resolution` | `2` | Downscale factor — `1` = full res (may OOM on 16 GB GPU), `2` = half res (safe) |
| `sh_degree` | `3` | Spherical harmonics degree — set `0` to save ~1.3 GB VRAM (wheat has no shiny surfaces) |
| `opacity_prune_threshold` | `0.005` | Raise to `0.01` if OOM — prunes more Gaussians, no quality loss for wheat |
| `data_device_cpu` | `true` | **Keep true on 16 GB GPU** — images stay in RAM, only active render goes to VRAM |
| `densify_until_iter` | `11000` | Gaussian count peaks here — most likely OOM point if it happens |
| `densify_grad_threshold` | `0.0002` | Lower = more Gaussians = higher quality but more VRAM |

Training takes ~1.5h at `resolution: 2` on a 24 GB GPU, ~45 min at `resolution: 2` locally.

---

## Output Structure

```
results/reconstruction/fip/{plot}/vanilla_3dgs/{experiment_name}/
├── config.yaml                          ← auto-saved full config at run start
├── point_cloud/
│   ├── iteration_7000/point_cloud.ply   ← mid-training Gaussian model
│   └── iteration_15000/point_cloud.ply  ← final trained Gaussian model
├── train/                               ← renders from training camera positions
├── test/ours_15000/                     ← renders from test camera positions
├── results.json                         ← PSNR / SSIM / LPIPS scores (step 3)
├── seg_logs/{exp_name}.txt              ← full segmentation log (step 4)
└── segmentation_3d/{exp_name}/          ← segmentation outputs (see segmentation README)
```

---

## Phone Data

Phone images need COLMAP processing first:
```bash
# Place images in input_plots/phone/phone01/input/ first
python src/reconstruction/convert.py -s input_plots/phone/phone01 --camera OPENCV
```
Then run normally with `dataset=phone plot=phone01`.

---

## Logging

The full pipeline output (all steps) is logged to `seg_logs/{exp_name}.txt` inside the experiment folder. Set `log_seg_only: true` (default) to only log the segmentation step — training output stays terminal-only since it's already verbose.
