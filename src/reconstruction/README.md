# 3DGS reconstruction — `src/reconstruction/`

This stage trains a 3D Gaussian Splatting (3DGS) model of a wheat plot from its
multi-view images and the COLMAP camera calibration. The trained model is what the 3D
segmentation stage later lifts the 2D masks into, so 3DGS reconstruction always runs
before 3D segmentation.

This folder holds the training-side scripts (`train_vanilla_3dgs.py`, `render.py`,
`metrics.py`, plus a couple of small helpers). You do not launch them directly. You run the orchestrator
`src/run_reconstruction.py`, which sits at the top of `src/`, and it launches each
step as its own process.

## How to run

Run everything from the workspace root. You pick the dataset with a single switch,
`profile=phone` (the default) or `profile=fip`, and the profile sets the right
dataset, defaults, and per-stage values for you.

All steps are **off by default**. You turn on the ones you want with `run_*` flags:

```bash
# train only (phone default)
python src/run_reconstruction.py run_train=true

# train, render the views, and compute metrics
python src/run_reconstruction.py run_train=true run_render=true run_metrics=true

# a single FIP plot
python src/run_reconstruction.py profile=fip plot=plot_463 run_train=true
```

Any config value can be overridden on the command line, so a specific plot, session,
or run name is just an extra argument:

```bash
python src/run_reconstruction.py run_train=true plot=field_A date=20250627 experiment_name=my_run
```

## Steps the orchestrator runs

The orchestrator `src/run_reconstruction.py` drives the whole 3DGS reconstruction and 3D
segmentation flow. Each step is a separate process, and each one reuses the output
of the steps before it, so run them in order the first time. After that you can
re-run any single step on the already-trained model.

**3DGS reconstruction steps** (this stage, documented below):

| Step | Toggle | Script | What it does |
|------|--------|--------|--------------|
| 1 | `run_train` | `reconstruction/vanilla_3dgs/train_vanilla_3dgs.py` | Train the 3DGS model of the plot |
| 2 | `run_render` | `reconstruction/render.py` | Produce 2D images from the trained 3D model, one per training and test camera, so the result can be looked at and scored |
| 3 | `run_metrics` | `reconstruction/metrics.py` | Score those test renders against the real images (PSNR, SSIM, LPIPS etc.) |

**3D segmentation and viewer steps** (documented in the [3D segmentation README](../segmentation_3d/README.md)):

| Step | Toggle | Script | What it does |
|------|--------|--------|--------------|
| 4 | `run_seg` | `segmentation_3d/run_3d_seg.py` | Give every Gaussian a 3D wheat head ID |
| 5 | `run_render_360` | `viewer/render_360.py` | Render a 360 degree flyaround video of the plot |
| 6 | `run_eval` | `segmentation_3d/eval_wheatgs.py` | Score the 3D segmentation quality |
| 6b | `run_eval_2d` | `segmentation_3d/eval_seg_2d.py` | Score the 2D masks per pixel against the manual ground truth |
| 7 | `run_viewer` | `viewer/wheatgs_rendering.py` | Open the interactive browser viewer |

Steps 4, 6, and 6b are the 3D segmentation stage, and steps 5 and 7 use the tools in
[src/viewer/](../viewer/). Step 4 (`run_seg`) also exports a per-head colored PLY right
after it finishes, and step 6b depends on step 6, so run them together.

## The three 3DGS reconstruction steps

- **Train** (`run_train`) optimizes a cloud of 3D Gaussians so that rendering them
  from each camera reproduces the input images. It starts from the sparse SfM points
  and refines and densifies the Gaussians over the training iterations. The model is
  saved as a point cloud partway through (`point_cloud/iteration_7000/`) and at the end
  (`point_cloud/iteration_15000/`).

- **Render** (`run_render`) produces 2D images from the trained 3D model, one per
  training and test camera, so you can look at the 3DGS reconstruction quality.

- **Metrics** (`run_metrics`) reads those test renders and computes PSNR, SSIM, LPIPS etc.
  against the real images, writing the scores to `results.json`.

## Experiment names

Set in `configs/reconstruction_seg3d/config.yaml`:

```yaml
experiment_name: "thesis_baseline"   # name of the output folder
prepend_date: false                  # true prepends today's date, e.g. 2025-04-28_thesis_baseline
allow_overwrite: false               # a named run that already exists is refused, not overwritten
```

`thesis_baseline` is the default name in mask generation, 3DGS reconstruction, and 3D
segmentation. When the names match, the stages find each other's outputs and chain
automatically. They may also differ, since 3D segmentation picks its mask input by name
(`segmentation_3d.mask_gen_experiment`), so one mask generation run can feed several
3DGS reconstruction and 3D segmentation runs without being recomputed.
A named run that already exists on disk is **refused** with a hard error so finished
runs are never clobbered. Set `allow_overwrite=true` to overwrite on purpose. Leave
`experiment_name` empty (`""`) for an auto timestamp.

## Key training parameters

Set in `configs/reconstruction_seg3d/reconstruction/vanilla_3dgs.yaml`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `iterations` | `15000` | Total number of training iterations. |
| `resolution` | `1` | Downscale factor for the training images. `1` is full resolution and is the pipeline default. |
| `sh_degree` | `3` | Spherical harmonics degree, which controls view-dependent color. |
| `opacity_prune_threshold` | `0.005` | Gaussians below this opacity get pruned. Raise to `0.01` to prune more and save VRAM. |
| `data_device_cpu` | `true` | Keep the training images in RAM instead of VRAM. Strongly recommended on a 16 GB GPU. |
| `densify_grad_threshold` | set by profile | Gradient threshold for splitting and cloning Gaussians. Lower means more Gaussians, so finer detail but more VRAM. FIP uses `0.0008` and phone `0.0002`, because FIP turns on AbsGS (see `absgrad`) whose gradients are larger. |
| `densify_until_iter` | `11000` | Stop adding new Gaussians after this iteration. The Gaussian count peaks around here. |
| `absgrad` | set by profile | AbsGS densification, which can help preserve finer wheat detail. FIP `true`, phone `false`. When `true` you must raise `densify_grad_threshold`. |
| `use_principal_point` | `true` | Uses COLMAP's off-center principal point through an asymmetric frustum. A big quality gain on FIP. On phone it does nothing, since COLMAP re-centers the principal point. |

These are the parameters you are most likely to change, not the full list. The config file
holds the rest.

## Thesis baseline configuration

The two profiles, `profile=fip` and `profile=phone`, carry the configuration we settled
on for the thesis, so the defaults already match what we actually ran. For 3DGS
reconstruction that is the gsplat engine, the principal point accounted for, 15k
iterations, and resolution 1. FIP additionally turns on AbsGS densification, which can
help preserve finer detail (with the raised gradient threshold), and phone uses the
standard densification threshold.

## Outputs

```
results/reconstruction/<dataset>/<plot>/vanilla_3dgs/<experiment_name>/
├── config.yaml                          ← the full config, auto-saved at run start
├── point_cloud/
│   ├── iteration_7000/point_cloud.ply   ← mid-training model
│   └── iteration_15000/point_cloud.ply  ← final trained model
├── train/                               ← renders from the training cameras (step 2)
├── test/                                ← renders from the test cameras (step 2)
├── results.json                         ← PSNR / SSIM / LPIPS on the test views (step 3), plus per_view.json
├── run_report.txt                       ← per-step OK / FAIL / SKIP with timings for the run
├── seg_logs/                            ← 3D segmentation logs
└── segmentation_3d/<exp_name>/          ← 3D segmentation outputs (check 3D segmentation README)
```

`<dataset>` is `fip` or `phone`, and phone additionally nests `<field>/<session>` in
the plot path. Only the main outputs are shown here. Each experiment folder also holds
extra visualizations and logs.

## Phone preprocessing

Phone data must run the SfM stage first to build `images/` and `sparse/0/` before any
3DGS reconstruction step. See [src/preprocessing/README.md](../preprocessing/README.md).
FIP plots come pre-calibrated from Agisoft and skip preprocessing entirely.

## Subfolders

- `vanilla_3dgs/` — the 3DGS trainer (`train_vanilla_3dgs.py`).
- `compare_renderers.py`, `vis_cam.py` — small helper scripts for A/B renderer checks
  and camera visualization.

## Optional analysis helpers

[`src/analysis/`](../analysis/) holds standalone scripts that are not part of a run. The one most
useful here is `collect_experiment_results.py`, which harvests every 3DGS reconstruction, 3D
segmentation and evaluation run into a single results table. Read the notes at the top of
[`src/analysis/README.md`](../analysis/README.md) first, since it needs its output path adjusted
before it runs.

Detailed write-ups on the 3DGS reconstruction flags live in the private `docs/` folder.
