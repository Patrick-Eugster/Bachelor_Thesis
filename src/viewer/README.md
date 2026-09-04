# Viewer — `src/viewer/`

This folder holds the two ways to look at a finished 3D segmentation. The viser viewer opens
the 3D model in your browser and lets you orbit around it, and the 360 render creates a
flyaround video of that same model. Both read the result of the 3D segmentation stage, and
neither needs the other, so you can run either one on its own.

## How to run

Both run through the orchestrator `src/run_reconstruction.py` (at the top of `src/`), which
resolves all the paths for you. Use the same names you trained and segmented with, and pick
the dataset with `profile=phone` (the default) or `profile=fip`:

```bash
# open the interactive viewer
python src/run_reconstruction.py run_viewer=true

# render the 360 flyaround video
python src/run_reconstruction.py run_render_360=true

# a FIP plot, with the experiment names spelled out
python src/run_reconstruction.py profile=fip plot=plot_461 \
  experiment_name=thesis_baseline segmentation_3d.exp_name=thesis_baseline \
  run_viewer=true
```

The viewer prints `Open http://localhost:8080` when it is ready. Press Ctrl+C in the terminal
to stop it. The 360 render writes `wheat_field_360.mp4` next to the 3D segmentation result, and
the [3D segmentation README](../segmentation_3d/README.md) describes the rest of that folder.

You can also call either script directly, which is useful when you want to point at a result
by hand:

```bash
cd src/viewer
python wheatgs_rendering.py \
  --input_ply   <model>/segmentation_3d/<seg_exp>/gaussians.ply \
  --labels_path <model>/segmentation_3d/<seg_exp>/all_obj_labels.pth \
  --colmap_path <plot>/sparse/0 \
  --images_path <plot>/images \
  --port 8080 --sh_degree 3 --fast_render
```

```bash
python src/viewer/render_360.py -s <plot> -m <model> \
  --render_type field --exp_name <seg_exp> \
  --n_frames 200 --framerate 20 --elevation 45 --fast_render --white_background
```

Here `<model>` is `results/reconstruction/<dataset>/<plot>/vanilla_3dgs/<experiment_name>/`
and `<plot>` is the plot folder under `input_plots/`.

## What it needs on disk

Both tools read the trained model and the 3D segmentation result. From the 3D segmentation
folder they need `gaussians.ply` and `all_obj_labels.pth`. From the plot folder they need
`sparse/0` and `images/` for the cameras. If there is no 3D segmentation result yet, the
viewer falls back to the plain trained point cloud, so it still opens after training alone,
just without head colors.

## Settings

Set in `configs/reconstruction_seg3d/config.yaml`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `viewer_type` | `full` | `full` is the colored per-head viewer. `single` is the plain one and does not work, see the notes below. |
| `fast_viewer` | `true` | Bake flat head colors once at startup, which is much faster. `false` recomputes the colors every frame, which looks nicer but is slow. |
| `viewer_port` | `8080` | Port the browser viewer listens on. |
| `n_frames` | `200` | How many frames the 360 video has. |
| `framerate` | `20` | Frames per second of the video. |
| `elevation` | `45` | Camera angle above the plot, in degrees. |
| `fast_render_360` | `true` | One colored render per frame instead of one per head, which is far faster. |
| `white_background_360` | `true` | White background in the video instead of black. |
| `render_360_downscale` | `2` | Video resolution is the training image size divided by this. `1` is full resolution. |
| `render_360_distance_mult` | `2.0` | How far the orbit sits from the plot. Higher is further out. |

These are the settings you are most likely to change, not the full list. The config file holds
the rest.

## Notes and limits

- The 360 render is very memory hungry on phone data, where a large plot can use up to about
  90 GB of RAM.
- On phone the orbit axis is estimated from the camera positions, because phone captures are
  not gravity aligned and a plain vertical orbit would loop over the top of the plot instead
  of circling it. The script prints which axis it picked.
- The video and the viewer do not use the same head colors. The video uses the high-contrast
  palette, while the viewer has its own plain color ramp.
- The 360 render uses the head labels saved by the 3D segmentation, and only rebuilds them
  from the per-head files in `ply/` when that file is missing.
- `singlewheat_rendering.py` (`viewer_type: "single"`) does not work. Use `full`.
- If the port is already taken, change `viewer_port`, or `--port` when calling the viewer
  directly.

## Optional

`recolor_heads_contrast.py` in [`src/analysis/`](../analysis/) recolors a 3D segmentation PLY
with the high-contrast palette. It has to be run separately.

## Scripts

- `wheatgs_rendering.py` — the interactive viser viewer.
- `render_360.py` — the 360 flyaround video.
- `singlewheat_rendering.py` — the plain viewer, kept but not working.
- `run.sh` — a legacy helper script, no longer used.
