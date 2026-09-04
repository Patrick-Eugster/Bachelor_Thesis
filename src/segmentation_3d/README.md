# 3D segmentation — `src/segmentation_3d/`

This stage is the 3D instance segmentation. It gives every Gaussian a consistent 3D
wheat head ID, so each wheat head is one instance with a single ID across all the views.
It takes the trained 3DGS model from the 3DGS reconstruction stage and the 2D masks from
the mask generation stage, and works out which Gaussians belong to which head.

This folder holds the 3D segmentation scripts. You do not launch them directly. You run
the orchestrator `src/run_reconstruction.py` (at the top of `src/`) with `run_seg=true`,
and it launches `run_3d_seg.py` for you. As with the other stages, you pick the dataset
with `profile=phone` (the default) or `profile=fip`.

## How it works

3D segmentation processes one 2D mask at a time (one wheat head in one camera view) and
repeats an iterative match-and-fine-tune loop:

1. **Lift to 3D** — a FlashSplat solver lifts this one 2D mask into 3D, picking the
   Gaussians that belong to it in that one view.

2. **Project to all views** — render those lifted Gaussians into every other camera.

3. **Match across views** — compare the projection against the 2D masks in the other
   views, and in each view keep the best overlap when its IoU is above the fine-tune
   matching threshold.

4. **Fine-tune** — re-optimize the 3D assignment using the matched views, then keep
   looking for more matching views and re-optimizing until no new ones are found.

5. **Save** — write the head's Gaussians and record its 2D projection in each camera.

This repeats until every mask is handled, and each accepted head gets a unique integer
ID. If a newly segmented head's Gaussians largely overlap an already-assigned head, it
updates that head instead of creating a new one.

## How to run

3D segmentation needs two things in place first: the 2D masks from the mask generation
stage, and the trained model from the 3DGS reconstruction stage. So run mask generation
and training before it, or run training in the same command. All steps are off by default
and turned on with `run_*` flags:

```bash
# segment, reusing the already-trained model (phone default)
python src/run_reconstruction.py run_seg=true

# segment, then evaluate
python src/run_reconstruction.py run_seg=true run_eval=true run_eval_2d=true

# a FIP plot
python src/run_reconstruction.py profile=fip plot=plot_461 run_seg=true
```

The segmentation settings live under the `segmentation_3d` config group, so override
them with that prefix:

```bash
python src/run_reconstruction.py run_seg=true segmentation_3d.exp_name=run_2
python src/run_reconstruction.py run_seg=true segmentation_3d.iou_threshold=0.5
```

## Configuration

Set in `configs/reconstruction_seg3d/segmentation_3d/default.yaml`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `exp_name` | `thesis_baseline` | Output subfolder inside `segmentation_3d/`. Change it to re-run on the same trained model without overwriting the old result. |
| `iou_threshold` | `0.6` | The fine-tune matching threshold. Higher is stricter, so fewer matches are accepted. |
| `detection_method` | `yolo_sam_v1` | Which mask-generation detector to read from (the detector subfolder). |
| `mask_gen_experiment` | `thesis_baseline` | Which mask-generation run to read from (the experiment subfolder under that detector). |
| `use_mask_cache` | `true` | Decode each mask once into a small cached crop, a speedup when there are many masks. Lossless. |
| `frustum_cull` | `true` | Skip rendering a head into cameras it does not project into. Lossless. |
| `save_vis_overlay` | `true` | Save a per-head overlay image, useful for checking the result. |
| `vis_max_heads` | `10` | How many heads get an overlay. `0` saves one for every head, which can be hundreds or even a few thousand images depending on the plot. |
| `contrast_palette` | `true` | High-contrast head colors in `gaussians_colored.ply` and the 360 video. Neighbouring heads get clearly different colors. `false` gives the old plain hue ramp, where the colors of touching heads look almost the same. |
| `seg_seed` | `0` | Fixed seed for the order the masks are processed in. The result depends on that order, so a fixed seed is what makes a run reproducible. |

These are the parameters you are most likely to change, not the full list. The config file
holds the rest.

Together, `detection_method` and `mask_gen_experiment` point to the mask folder
`results/mask_generation/<dataset>/<plot>/<detection_method>/<mask_gen_experiment>/`, so both
are needed to identify which masks to read. They are separate from the 3DGS reconstruction's
own `experiment_name`. So the names can match, in which case the shared `thesis_baseline`
chains the stages automatically, or they can differ, which lets one mask-generation run
feed several 3D segmentation runs without recomputing the masks.

### `use_principal_point` must match the trained model

3D segmentation must render with the same `use_principal_point` value the model was
trained with, or every projection shifts by a few pixels and matching falls apart. The
default is `true` for both datasets and a train-then-segment run is always consistent, so
this only needs care when you segment a pre-trained model in a separate run and change the
flag in between.

### Ground cull and region-of-interest options

A ground cull always runs and cannot be turned off. It drops the ground below the plot so
it is not gathered into a head. Which cull runs depends on the plot. When the triangulated
markers from preprocessing are present, as they are on phone, it fits the marker plane and
cuts only the lowest `ground_percentile` percent by height, which keeps every head whole.
Without that file, as on FIP, it falls back to the older cut at the mean height, which
works there because the heads are the topmost part of an overhead capture.

On top of that, three optional filters restrict the 3D segmentation to the plot region of
interest and drop the coded markers from the result (`roi_cull`, `height_band`,
`marker_exclude`). These are phone quality options, off by default, and they also need the
triangulated markers. See the comments in
`configs/reconstruction_seg3d/segmentation_3d/default.yaml`.

## Outputs

```
results/reconstruction/<dataset>/<plot>/vanilla_3dgs/<experiment_name>/segmentation_3d/<exp_name>/
├── gaussians.ply           ← all segmented Gaussians, with head IDs
├── gaussians_colored.ply   ← same, with a color per head baked in (for the viewer)
├── all_obj_labels.pth      ← the head ID of every Gaussian
├── results.csv             ← per head: ID, source mask, view count, Gaussian count
├── seg_summary.json        ← run totals, including the predicted head count for the plot
├── 2DSeg/                  ← the 2D projection of every head per camera (read by the evaluation and the 360 step)
├── ply/                    ← one PLY per head (wh_0001.ply, and wh_0042_b.ply for an overlap update)
├── img/                    ← overlay images for the first vis_max_heads heads (only if save_vis_overlay is on)
├── eval_2d/                ← output of the 2D evaluation (run_eval_2d)
└── wheat_field_360.mp4     ← 360 flyaround video of the segmented field (run_render_360)
```

`<dataset>` is `fip` or `phone`, and phone additionally nests `<field>/<session>` in the
plot path. The 360 step also writes its single frames into a `3DSeg/` folder while it
renders, then deletes that folder once the video is encoded. Only the main outputs are
shown here, and the folder holds a few smaller files besides, such as the run metadata.

## Evaluation

Two evaluation steps run through the same orchestrator.

- **Step 6** (`run_eval`, `eval_wheatgs.py`) renders the 3D segmentation back to 2D and
  writes an overlay image and a binary segmentation mask per camera. Step 6b needs this
  output, so run them together the first time.

- **Step 6b** (`run_eval_2d`, `eval_seg_2d.py`) scores those 2D masks per pixel against
  the manual ground-truth masks in `input_plots/<dataset>/<plot>/manual_label/`, and also
  reports a per-view head count (predicted heads vs ground truth). It writes the scores
  and a color-coded image to `eval_2d/`.

```bash
python src/run_reconstruction.py run_eval=true run_eval_2d=true
```

**The `eval_seg_2d` pixel scores are computed on collapsed binary masks, so the built-in
step is not a per-instance evaluation.** Both the ground truth and the prediction are
single masks where every wheat head pixel is 1 and background is 0, no matter how many
heads are present. The scores say whether the pipeline found wheat head area in the right
places, not whether it told each individual head apart. Two runs can score the same even
if one separates the heads cleanly and the other merges them into one blob. To measure
that, use the per-instance evaluation below.

Both FIP and phone have manual ground-truth masks. FIP ships one annotated camera per
plot from the original authors, and the phone masks were made with the
[ground-truth tool](../mask_generation/gt_tool/README.md).

### Per-instance evaluation (phone)

The built-in `eval_seg_2d` is union-based, but a true per-instance evaluation is available
for phone as a standalone analysis script, `src/analysis/phone_seg_instance_eval.py` (with
the pixel companion `phone_seg_cpu_eval.py`). It matches each predicted head in the 2D
projection one-to-one against the phone per-head ground-truth instance masks
(`manual_label/<stem>_sets/`, from the [ground-truth tool](../mask_generation/gt_tool/README.md))
and reports head-count precision/recall/F1, matched-mask IoU, and merge/split counts, so
it catches the over-merging the union score cannot. It runs on its own, not through the
orchestrator, and is described in [src/analysis/README.md](../analysis/README.md). FIP has
union ground truth only, so this is phone-only.

These two scripts stay in `src/analysis/` and were not moved into this folder, so the
other archived scripts that import them do not break. Each one has a hardcoded list of the
runs it scores. Point that list at your own segmentation runs before you run it. The
output path also goes into the private `docs/analysis_results/` folder and is written
near the top of the file, so it needs to be adjusted as well.

## Viewing the result

Two more steps show the segmented field, both run through the same orchestrator and both
built on the tools in [src/viewer/](../viewer/):

- **Step 5** (`run_render_360`) renders a 360 degree flyaround video of the per-head
  colored field to `segmentation_3d/<exp_name>/wheat_field_360.mp4`. This step is very memory
  hungry on phone, where a large plot can use up to about 90 GB of RAM.
- **Step 7** (`run_viewer`) opens the colored segmentation in an interactive browser
  viewer.

```bash
python src/run_reconstruction.py run_render_360=true
```

## Scripts

- `run_3d_seg.py` — the match-and-fine-tune loop (run with `run_seg`).
- `export_colored_ply.py` — bakes a color per head into `gaussians_colored.ply`, run
  automatically after `run_seg`.
- `eval_wheatgs.py` — the `run_eval` step.
- `eval_seg_2d.py` — the `run_eval_2d` step.
- `seg_roi.py` — internal helper for the ground cull and the region-of-interest options.
- `headcount_gt.py`, `seg_head_size_stats.py` — optional diagnostics, not part of a
  normal run.

Detailed write-ups on the 3D segmentation internals live in the private `docs/` folder.
