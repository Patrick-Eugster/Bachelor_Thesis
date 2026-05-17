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
python src/run_reconstruction.py dataset=phone plot=field_A date=20250618 run_train=true experiment_name=phone_test
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
| 1 | `run_train` | `vanilla_3dgs/train_vanilla_3dgs.py` | Train 3DGS model — L1 + SSIM loss |
| 2 | `run_render` | `render.py` | Render from training + test camera positions |
| 3 | `run_metrics` | `metrics.py` | Compute PSNR / SSIM / LPIPS on test views → `results.json` |
| 4 | `run_seg` | `segmentation_3d/run_3d_seg.py` | Assign 3D wheat head IDs to Gaussians |
| 4b | auto after 4 | `segmentation_3d/export_colored_ply.py` | Bake per-head HSV colors into `gaussians_colored.ply` |
| 5 | `run_render_360` | `viewer/render_360.py` | Render 360° flyaround video → `wheat_field_360.mp4` |
| 6 | `run_eval` | `segmentation_3d/eval_wheatgs.py` | Evaluate 3D segmentation quality vs SAM masks |
| 7 | `run_viewer` | `viewer/wheatgs_rendering.py` | Open interactive viser viewer at `http://localhost:8080` — in the GUI set render res to `2560 × 1920` for good quality |

### Input / Output per step

| Step | Input | Output |
|------|-------|--------|
| 1 | `images/` + `sparse/` + `bboxes/` | `point_cloud/iteration_{N}/point_cloud.ply` |
| 2 | `point_cloud/iteration_{N}/point_cloud.ply` | `train/` + `test/ours_{N}/` |
| 3 | `test/ours_{N}/` + `images/` | `results.json` |
| 4 | `point_cloud/iteration_{N}/point_cloud.ply` + `masks/` | `segmentation_3d/{exp_name}/gaussians.ply` + `2DSeg/` + `ply/` |
| 4b | `segmentation_3d/{exp_name}/gaussians.ply` | `segmentation_3d/{exp_name}/gaussians_colored.ply` |
| 5 | `segmentation_3d/{exp_name}/gaussians.ply` | `segmentation_3d/{exp_name}/3DSeg/wheat_field_360.mp4` |
| 6 | `segmentation_3d/{exp_name}/2DSeg/` + `gaussians.ply` | `train/overlay/` + `train/segmentation/` + `test/overlay/` + `test/segmentation/` |
| 7 | `segmentation_3d/{exp_name}/gaussians.ply` + `sparse/` + `images/` | — (interactive) |

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

## Phone Data — COLMAP via `convert.py`

Phone data uses a two-level folder structure: `input_plots/phone/{field}/{date}/`.

FIP plots come with COLMAP-format camera calibration already done (via Agisoft Metashape in the original paper). Phone images do NOT — they're just JPGs with no camera poses. **COLMAP must be run first** to recover camera positions and produce the sparse 3D structure that 3DGS needs as input.

### What `convert.py` does

It wraps four COLMAP commands into one Python script:

| Step | COLMAP command | What it does |
|------|----------------|--------------|
| 1 | `feature_extractor` | Detects SIFT features in every image |
| 2 | `sequential_matcher` or `exhaustive_matcher` | Finds matching features across image pairs |
| 3 | `mapper` | Runs Structure-from-Motion (SfM) — recovers camera positions + sparse 3D points |
| 4 | `image_undistorter` | Removes lens distortion → final `images/` + `sparse/0/` ready for 3DGS |

Output is laid out exactly as 3DGS expects:
```
{source_path}/
├── input/                    ← your raw images (placed here before running)
├── images/                   ← undistorted images, used by 3DGS training
├── sparse/0/                 ← camera poses + 3D points (cameras.bin, images.bin, points3D.bin)
├── distorted/                ← intermediate working files (can be deleted after)
├── stereo/                   ← created by undistorter, not used by our pipeline
└── logs/colmap.log           ← full COLMAP output, saved automatically
```

### How to run

Place images in `input_plots/phone/field_A/20250618/input/` first, then:

```bash
python src/reconstruction/convert.py -s input_plots/phone/field_A/20250618
```

After it finishes, the data is ready for 3DGS:
```bash
python src/run_reconstruction.py dataset=phone plot=field_A date=20250618 run_train=true
```

### CLI options

All options have defaults so you can usually run with just `-s`:

| Flag | Default | Meaning |
|------|---------|---------|
| `-s` / `--source_path` | required | Folder containing `input/` with your raw images |
| `--camera` | `SIMPLE_PINHOLE` | Camera model — see table + test results below. Default chosen empirically: only model that worked on phone wheat data |
| `--matcher` | `sequential` | `sequential` (low RAM, ordered images) or `exhaustive` (any order, high RAM) |
| `--sequential_overlap` | `25` | Sequential only: how many next images each image matches against |
| `--num_threads` | `8` | Threads for SIFT extraction + matching — **lower = less RAM**. See RAM table below. Set `-1` for all cores |
| `--no_gpu` | `True` | Use CPU for SIFT (your COLMAP install is CPU-only anyway) |
| `--skip_matching` | `False` | Skip steps 1–3 if you already have a working `distorted/sparse/0/` |
| `--resize` | `False` | Also create downscaled `images_2/`, `images_4/`, `images_8/` folders |

### RAM usage and `--num_threads`

COLMAP's CPU SIFT extracts features in parallel — each thread loads a different image at full resolution and builds its own scale-space pyramid. So **RAM scales linearly with thread count**. With 11 MP phone images and ~2 GB working set per thread (image + pyramid + feature buffers), the rough budget is:

| `--num_threads` | Approx RAM (feature extraction step) | Speed |
|---|---|---|
| `-1` (all 12 cores) | ~25-29 GB ❌ fills 35 GB WSL2 | Fastest |
| `8` (default) | ~16-20 GB ⚠️ | ~1.5× slower |
| `6` | ~12-15 GB ✅ | ~2× slower |
| `4` | ~8-10 GB ✅✅ | ~3× slower |
| `2` | ~4-5 GB ✅✅✅ | ~6× slower |

Feature extraction is typically 1-2 min of the total ~8 min runtime, so even 4 threads only adds 3-4 min to the total. **Pick the highest thread count your RAM tolerates.**

### Camera models

| Model | Params | When to use |
|-------|--------|-------------|
| `SIMPLE_PINHOLE` | f, cx, cy | **Best for wheat / phone — see test results below.** Single focal length, no distortion |
| `PINHOLE` | fx, fy, cx, cy | Two focal lengths, no distortion. Works for richer scenes |
| `SIMPLE_RADIAL` | f, cx, cy, k1 | Single radial distortion param — middle ground |
| `OPENCV` | fx, fy, cx, cy + k1, k2, p1, p2 | Full radial + tangential distortion — works for buildings / varied geometry, **fails for repetitive vegetation** |
| `OPENCV_FISHEYE` | fx, fy, cx, cy, k1–k4 | Fisheye / ultra-wide lenses only |
| `FULL_OPENCV` | 12 params | Overkill, rarely useful |

Higher-parameter models try to fit more lens distortion but need many well-distributed feature matches to converge. Wheat fields are mostly repetitive vegetation → weak feature constraints → the optimizer can't uniquely solve the extra distortion params → reconstruction fails. Less is more here.

### Empirical test — phone data on plot `colmap_test` (93 images, exhaustive matching, 8 threads)

| Model | Registered | Notes |
|-------|------------|-------|
| `SIMPLE_PINHOLE` | **63/93** ✅ | Only model that produced a usable reconstruction |
| `PINHOLE` | 2/93 ❌ | Adding even just `fy` as a separate parameter breaks it |
| `OPENCV` | 2/93 ❌ | Distortion params can't be constrained on repetitive vegetation |

**Conclusion:** for phone images of wheat fields, **always use `SIMPLE_PINHOLE`**. The original phone dataset shipped with this codebase also uses it — the result was reproduced empirically, not just inherited.

For datasets with more geometric variety (buildings, structured scenes, non-vegetation outdoor), try OPENCV first — it gives better reconstruction quality when it converges.

### Sequential vs exhaustive matcher

- **`sequential`** (default) — matches each image only against the next `sequential_overlap` images. **Use this for walk-through / video sequences** (phone images taken while walking). Low RAM, fast. Default overlap of 25 means each image is matched against the next 25 in sequence.
- **`exhaustive`** — matches every image against every other image. O(N²) pairs — RAM-intensive for 100+ images. Only use for **unordered** image sets where you don't know which images are nearby.

For 100+ phone images, **exhaustive matching can fill 30+ GB of RAM**. Always use sequential for phone walk-throughs.

### Logging and timing

The script prints each step with timing and saves the full COLMAP output to `{source_path}/logs/colmap.log`:

```
Step 1/3: Feature extraction...
  Feature extraction done in 142.3s
Step 2/3: Feature matching (sequential)...
  Feature matching done in 87.1s
Step 3/3: Mapping (SfM + bundle adjustment)...
  Mapper done in 218.4s
Undistorting images...
  Undistortion done in 23.7s
Done. Total time: 7.9 min (471s)
```

### Common issues

- **"Single camera specified, but images have different dimensions"** — some phone images have slightly different resolutions (e.g. HDR vs non-HDR). The script no longer passes `--single_camera 1`, so COLMAP creates one camera per dimension group. No action needed.
- **Only some images end up in `images/`** — COLMAP couldn't link all images into one connected reconstruction and only the largest sub-model was undistorted. Check `distorted/sparse/` — if there are folders `0/`, `1/`, `2/` etc., your images don't have enough overlap. Try a higher `--sequential_overlap` or take photos with more overlap.
- **"Finding good initial image pair" appears multiple times in the log** — same as above, indicates disconnected sub-reconstructions.

---

## Logging

The full pipeline output (all steps) is logged to `seg_logs/{exp_name}.txt` inside the experiment folder. Set `log_seg_only: true` (default) to only log the segmentation step — training output stays terminal-only since it's already verbose.
