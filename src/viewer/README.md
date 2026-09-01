# `src/viewer/` — interactive viser viewer + 360° flyaround video

Two ways to *look* at a segmented 3DGS wheat plot after the pipeline has run:

| Tool | File | What it gives you | Interactive? |
|---|---|---|---|
| **viser viewer** (full) | `wheatgs_rendering.py` | live 3D viewer in a browser, wheat heads colored by 3D ID | ✅ yes — orbit/zoom |
| **render_360** | `render_360.py` | an `.mp4` flyaround video of the colored plot | ❌ batch render |
| plain viewer (single) | `singlewheat_rendering.py` | live viewer, no per-head colors | ⚠️ **currently broken** (see bottom) |

Both the full viewer and render_360 consume the **step-4 segmentation output** — they do **not** depend on each
other. You can run either one directly on a seg result.

---

## Do I need Euler? No — run these LOCALLY.

- **viser** is an interactive server (renders live on the GPU, you click around in a browser). It has to run on
  your local machine — you can't meaningfully interact with it on the batch cluster.
- **render_360** is a batch render, but Euler's env is missing `libopenh264.so.5` so the final mp4 stitch fails
  there (frames render, video doesn't). So render it locally too.
- Both just **render** a trained model (forward pass only, no training/seg optimization), so they fit in local
  VRAM fine — this is *not* the "never run full seg locally" case.

So the normal flow is: **train + seg on Euler → rsync results local → view / render_360 locally.**

---

## What files each needs (must be present locally after rsync)

Everything lives under the experiment's model dir:
`results/reconstruction/<dataset>/<plot>/vanilla_3dgs/<experiment_name>/`

- **viser (full):**
  - `segmentation_3d/<exp_name>/gaussians.ply`   — the fine-tuned scene model (falls back to
    `point_cloud/iteration_<N>/point_cloud.ply` if the seg ply is missing)
  - `segmentation_3d/<exp_name>/all_obj_labels.pth`  — the per-Gaussian head-ID labels (written by step 4)
  - the dataset's `sparse/0/` + `images/`  (for the camera trajectory + backgrounds)
- **render_360:**
  - `segmentation_3d/<exp_name>/gaussians.ply` + the per-head `segmentation_3d/<exp_name>/ply/wh_*.ply`
    (it rebuilds `all_obj_labels.pth` from those if it's absent)

> **rsync check:** make sure your pull included the whole `segmentation_3d/<exp_name>/` folder
> (`gaussians.ply`, `all_obj_labels.pth`, and `ply/`). If viser complains it can't find labels, that folder
> didn't come across.

---

## Run the viser viewer

### Option A — via the orchestrator (recommended: it resolves all the paths for you)

Use the **same** `plot` / `date` / `experiment_name` / `segmentation_3d.exp_name` you trained+segged with.
All `run_*` toggles default to **false** in `configs/reconstruction_seg3d/config.yaml`, so you only need to
turn the viewer on — no need to spell out the other steps:

```bash
# phone — a real, working example (field_A/20250715, the phone_sahi run, seg_cull_v3 = known-good seg)
python src/run_reconstruction.py dataset=phone plot=field_A date=20250715 \
  experiment_name=phone_sahi segmentation_3d.exp_name=seg_cull_v3 \
  run_viewer=true
```

```bash
# FIP equivalent (no date= for fip)
python src/run_reconstruction.py plot=plot_461 \
  experiment_name=<recon_exp> segmentation_3d.exp_name=<seg_exp> \
  run_viewer=true
```

> **Phone gotcha:** `plot=` is the *field* and `date=` is the session — the path is
> `input_plots/phone/<plot>/<date>/`. `experiment_name` must match the training run and
> `segmentation_3d.exp_name` the seg run, or it won't find `gaussians.ply` / `all_obj_labels.pth`.

Steps toggled off are assumed already-present on disk, so `run_viewer=true` alone works on a rsynced result.
It prints `Open http://localhost:8080` — open that in your browser. **Ctrl+C** in the terminal stops it.

Handy toggles (in `configs/reconstruction_seg3d/config.yaml`):
- `viewer_port: 8080`   — change if 8080 is busy
- `viewer_type: "full"` — colored per-head viewer (keep this; `"single"` is broken)
- `fast_viewer: true`   — pre-bake flat head colors into the Gaussians (fast). `false` = per-frame overlay
  colors (slower, prettier).

### Option B — standalone (run from the viewer dir)

```bash
cd src/viewer
python wheatgs_rendering.py \
  --input_ply   <model>/segmentation_3d/<seg_exp>/gaussians.ply \
  --labels_path <model>/segmentation_3d/<seg_exp>/all_obj_labels.pth \
  --colmap_path <dataset>/sparse/0 \
  --images_path <dataset>/images \
  --port 8080 --sh_degree 3 --fast_render
```
where `<model>` = `results/reconstruction/<dataset>/<plot>/vanilla_3dgs/<recon_exp>` and `<dataset>` =
`input_plots/<dataset>/<plot>`. (`--fast_render` = the `fast_viewer` flat-color mode.)

---

## Run the 360° flyaround video

### Option A — orchestrator

```bash
# phone — a real, working example (same session/exps as the viewer above)
python src/run_reconstruction.py dataset=phone plot=field_A date=20250715 \
  experiment_name=phone_sahi segmentation_3d.exp_name=seg_cull_v3 \
  run_render_360=true
```
Output: `.mp4` in the experiment's `segmentation_3d/<seg_exp>/3DSeg/` folder. Toggles:
`n_frames: 200`, `framerate: 20`, `elevation: 45`, `fast_render_360: true`, `white_background_360: true`,
`render_360_downscale: 2` (video res = training image ÷ this; **1 = full resolution**, 2 = half).

### Option B — standalone

```bash
python src/viewer/render_360.py \
  -s <dataset> -m <model> --render_type field --exp_name <seg_exp> \
  --n_frames 200 --framerate 20 --elevation 45 --fast_render --white_background
```

`--fast_render` does one colored render per frame instead of N_heads flashsplat renders (~N_heads× faster,
flat colors). Drop it for per-head-overlay colors (much slower).

---

## Notes / gotchas

- **viser ≠ render_360.** Neither needs the other; both just need the step-4 seg output. View first, render the
  video whenever.
- **Euler mp4 is broken** (`libopenh264.so.5` missing) — render_360 frames render but the stitch fails there.
  Do the video locally. (Don't `conda install ffmpeg` into the torch-2.1.2 env — it can pull an incompatible
  stack.)
- **360 orbit axis (phone):** phone COLMAP frames aren't gravity-aligned, so a naive world-Z orbit loops
  the camera *over the top* of the plot instead of circling it. `render_360` now estimates the scene up
  from the camera centers (plane-fit) and orbits around that → proper turntable. FIP (gravity-aligned,
  up ≈ world-Z) is detected and keeps the original path byte-identical. It prints `Scene up axis: …` so
  you can see which path it took.
- **Colored render labels:** `render_360` uses the step-4 `all_obj_labels.pth` (each Gaussian in exactly
  ONE head — a clean partition). It only rebuilds labels from the per-head `ply/wh_*.ply` as a *fallback*
  when that file is missing, and never overwrites it. (Earlier it always rebuilt from the plys, which store
  each head's RAW pre-overlap-resolution set → a Gaussian could land in ~30 heads → chaotic colors, markers
  colored. Fixed. If you ran the old version, its `all_obj_labels.pth` was overwritten — restore it from a
  byte-identical sibling seg run, or re-run seg.)
- **Port busy:** change `viewer_port` (orchestrator) or `--port` (standalone).
- **`sh_degree` must match training** (default 3; if you trained with `sh_degree=0`, pass that).
- **`singlewheat_rendering.py` (`viewer_type: "single"`) is broken** — the plain post-step-1 viewer, unused
  since `viewer_type: "full"` is the default. Broken by multiple API drifts: old nerfview `img_wh`, old
  `Camera` constructor args, dropped `load_ply(remove_features_rest=…)` kwarg, hardcoded `.bin` COLMAP format,
  and no `--colmap_path`/`--images_path` passed by `run_reconstruction.py`. Use `viewer_type: "full"`.
- **FIP principal-point:** FIP models are trained `use_principal_point=true`; the viewer reads COLMAP
  intrinsics directly so a small shift is possible, but it's visualization-only. Phone pp ≈ center → no-op.
