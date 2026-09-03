# Mask Generation — `src/mask_generation/`

For each image, this stage puts a box around each wheat head and segments the head
inside the box into a 2D binary mask. A detector produces the boxes and SAM (Segment
Anything) turns them into the masks. The boxes and masks are written per image and are
the 2D input that 3D segmentation lifts into the 3DGS reconstruction.

Everything runs through one entry point, `run_mask_generation.py`. It picks a detector
by `method=`, runs it to produce the boxes, and then runs the shared SAM step that turns
those boxes into masks:

```bash
python src/mask_generation/run_mask_generation.py
```

This is a per-dataset stage, so it uses the same `profile=phone` (default) or
`profile=fip` switch as 3DGS reconstruction and 3D segmentation. The profile picks the
dataset and the thesis-best detector settings for that dataset in one go.

## Detectors

You choose which detector produces the boxes with `method=`, and `run_mask_generation.py`
runs the one you name. The default is `yolo_sam_v1`, the thesis baseline for both datasets.

| `method=` | What it is | Masks |
|-----------|------------|-------|
| `yolo_sam_v1` (default) | YOLOv5 wheat-head detector, then the shared SAM step | SAM |
| `sahi_yolo_sam` | The same YOLOv5 detector run tiled with SAHI, so small heads in large images are found | SAM |
| `yolo11_sam` | A YOLOv11 wheat-head detector, then the shared SAM step | SAM |
| `yolo11_seg` | A YOLOv11 instance-segmentation model that outputs a mask per head directly | its own |

The first three produce boxes and hand them to the shared SAM step. `yolo11_seg` outputs a
mask per head directly, so it skips the shared SAM step. The profiles compose
`yolo_sam_v1`, and you switch to another detector by overriding the method:

```bash
python src/mask_generation/run_mask_generation.py method=sahi_yolo_sam
```

## How to run

```bash
# phone (default profile), every session in the dataset
python src/mask_generation/run_mask_generation.py

# fip, every plot in the dataset
python src/mask_generation/run_mask_generation.py profile=fip

# one phone session only
python src/mask_generation/run_mask_generation.py dataset.plot_glob=field_A/20250715

# one fip plot only
python src/mask_generation/run_mask_generation.py profile=fip dataset.plot_glob=plot_461

# a different detector on the phone default
python src/mask_generation/run_mask_generation.py method=sahi_yolo_sam
```

By default the stage processes every plot or session in the dataset. Pass
`dataset.plot_glob` to target a single one. Any config value can be overridden on the
command line, so a plot, a run name, or a threshold is just an extra argument.

## Key options

Set these in `configs/mask_generation/config.yaml` and the method files under
`configs/mask_generation/method/`, or override them on the command line.

| Option | Default | Meaning |
|--------|---------|---------|
| `profile` | `phone` | Dataset and per-dataset best settings. `phone` or `fip` |
| `method` | `yolo_sam_v1` | Which detector runs. See the table above |
| `dataset.plot_glob` | all | Which plots or sessions to process. Set one to target it, e.g. `field_A/20250715` |
| `experiment_name` | `thesis_baseline` | Names the output folder. 3D segmentation reads these masks by that name, see the outputs section |
| `roi.enabled` | `true` | Marker region of interest. On for phone, does nothing on FIP (see below) |
| `method.only_yolo` | `false` | Skip SAM and write only the boxes. Handy for box-level metrics |
| `method.sam_backend` | set by profile | Which SAM turns boxes into masks. Phone uses `sam2`, FIP uses `sam1`. Also accepts `sam3` |
| `method.sam_crop_mode` | set by profile | How much of the image SAM encodes per head. Phone uses `per_tile`, FIP uses `full_frame`. Also `per_head` |

These are the options you are most likely to change, not the full list. The config files
hold the rest.

The phone profile already sets the values we settled on for phone data, so a plain
`profile=phone` run uses full-resolution 4032 px input, per-tile SAM2, and a strict 0.70 confidence threshold. The FIP profile keeps the settings of 1280 px input,
full-frame SAM1, and a 0.35 confidence threshold.

## Region of interest (phone)

Phone images catch neighbouring plots and blurry corners, so mask generation can restrict
detection to the plot area bounded by the ground markers. With `roi.enabled=true` (the
default) it builds a polygon from the triangulated markers, greys out everything outside
it, and drops boxes that fall outside the plot.

This is a phone feature and needs the preprocessing marker step to have run first
(`run_markers=true`, see [`../preprocessing/markers/README.md`](../preprocessing/markers/README.md)),
since the polygon comes from `logs/marker_points3d.json`. On FIP there are no phone
markers, so the region of interest does nothing and the output is unchanged. To
turn it off anywhere, pass `roi.enabled=false`. The builder is
[`roi_mask.py`](roi_mask.py), and you can preview a polygon before a run:

```bash
python src/mask_generation/roi_mask.py input_plots/phone/field_A/20250715 --buffer_px 80
```

## Outputs

Written under the dataset's mask-generation result folder, one tree per detector:

```
results/mask_generation/<dataset>/<plot>/<method>/<experiment_name>/
├── bboxes/     ← one .pt of wheat-head boxes per image
└── masks/      ← one binary mask PNG per head
```

For phone `<plot>` is `<field>/<session>`, and for FIP it is the plot name. The stage also
writes visual overlays and a per-run summary next to these, and prints a boxed summary
report at the end with the head counts and timing.

3D segmentation reads its masks from that folder by name, through
`segmentation_3d.detection_method` (the `<method>` part) and
`segmentation_3d.mask_gen_experiment` (the `<experiment_name>` part). Their defaults are
`yolo_sam_v1` and `thesis_baseline`, which is what this stage writes by default, so a
plain run chains automatically. They are
separate settings from this stage's own `experiment_name`, so the names may also differ,
which lets one mask generation run feed several 3D segmentation runs without being
recomputed.

## Ground-truth labeling tool

[`gt_tool/`](gt_tool/) is an interactive tool built for this thesis to create pixel-mask
ground truth for phone wheat heads. It serves a small browser page where you click points
on a head and SAM2.1 segments it, with brush and polygon tools for the heads SAM cannot
get on its own. It writes the per-head ground-truth masks that the evaluation reads. Run
it locally:

```bash
python -m mask_generation.gt_tool.server   # then open http://localhost:8000
```

See [`gt_tool/README.md`](gt_tool/README.md) for the full workflow and controls.

## Model weights

The YOLO and SAM checkpoints are not shipped with the repository. They go in
`src/mask_generation/weights/` (gitignored). See [`../../INSTALL.md`](../../INSTALL.md) for
which files to download and where to put them.

## Subfolders

- [`yolo_sam_v1/`](yolo_sam_v1/) — the default YOLOv5 detector plus the shared SAM step,
  with a deeper look at how the detection runs.
- [`sahi_yolo_sam/`](sahi_yolo_sam/), [`yolo11_sam/`](yolo11_sam/),
  [`yolo11_seg/`](yolo11_seg/) — the alternative detectors selected by `method=`.
- [`sam_v1/`](sam_v1/) — the shared SAM step every box-producing detector hands off to.
- [`evaluation/`](evaluation/) — scores boxes and masks against the ground truth.
- [`gt_tool/`](gt_tool/) — the ground-truth labeling tool described above.
- [`yolov5/`](yolov5/) — the vendored YOLOv5 code the detectors build on.

## Optional analysis helpers

[`src/analysis/`](../analysis/) holds standalone scripts that are not part of a run but are useful
for this stage. They cover confidence-threshold sweeps and mask scoring, a plain-YOLO versus SAHI
comparison, and preparing the seed boxes and masked images for ground-truth labeling. Read the
notes at the top of [`src/analysis/README.md`](../analysis/README.md) first, since most of them
need their output path adjusted before they run.

> The deeper mask-generation write-ups (the SAHI tiling study, the SAM-backend comparison,
> the region-of-interest study) live under `docs/`, a private working-notes submodule that
> is not part of a public clone.
