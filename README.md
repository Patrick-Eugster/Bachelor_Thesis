# Phone-Wheat3DGS

Phone-Wheat3DGS is a smartphone adaptation of **Wheat3DGS**, a pipeline for 3D
reconstruction and instance segmentation of wheat heads in a field plot. Given
many overlapping images of a wheat plot, it reconstructs the plot with
3D Gaussian Splatting (3DGS), segments every individual wheat head in 3D, and gives
each head a consistent ID across all views. The original pipeline was built for a
specialized FIP (Field Imaging Platform) camera rig, and this work adapts the full
pipeline so that it also runs on images captured with a consumer smartphone.

This code accompanies the bachelor thesis: *3D Wheat Head Instance Segmentation from
Smartphone Sideview Captures*.

> **First time here?** Install the environment before running anything — see
> **[INSTALL.md](INSTALL.md)**.

## Pipeline overview

The pipeline has four stages. Each stage has its own entry point under `src/`, is
configured through Hydra files under `configs/`, and writes into `results/`. The
stages run in order, and each one consumes the previous stage's output, so run them
in sequence (for example, 3D segmentation needs the masks from stage 2 and the
trained model from stage 3).

1. **SfM (preprocessing)** — `src/preprocessing/run_preprocessing.py`
   Phone only: it crops the images to one uniform size and runs COLMAP structure
   from motion (ALIKED + LightGlue) to recover the camera poses and calibration,
   together with a sparse point cloud. FIP plots are already processed by Agisoft, so they skip this stage.
   → [src/preprocessing/README.md](src/preprocessing/README.md)

2. **Mask generation** — `src/mask_generation/run_mask_generation.py`
   Detects wheat-head boxes and segments each head into a 2D binary mask in every
   image, using a YOLO/SAHI detector followed by SAM.
   → [src/mask_generation/README.md](src/mask_generation/README.md)

3. **3DGS reconstruction** — `src/run_reconstruction.py`
   Trains a 3D Gaussian Splatting model of the plot with the gsplat engine.
   → [src/reconstruction/README.md](src/reconstruction/README.md)

4. **3D segmentation** — part of `src/run_reconstruction.py`
   Lifts each view's masks to 3D with FlashSplat, matches them across views, and
   assigns every wheat head a single consistent 3D ID.
   → [src/segmentation_3d/README.md](src/segmentation_3d/README.md)

Stages 3 and 4 are both handled by the reconstruction orchestrator
(`src/run_reconstruction.py`). It has eight steps, train, render, metrics, 3D
segmentation, 360 video, evaluation, 2D evaluation, and the viewer. Every step is
**off by default**, so you switch on the ones you want with `run_*` flags (for
example `run_train=true`).

## Repo layout

```
Phone-Wheat3DGS/
├── README.md               ← this file
├── INSTALL.md              ← how to install
├── pyproject.toml          ← pip package (pip install -e .) + [maskgen] extra
├── environment.yml         ← conda environment for training and segmentation
├── configs/                ← Hydra config files — edit these, not the scripts
├── scripts/                ← example SLURM job templates (see scripts/README.md)
├── src/
│   ├── run_reconstruction.py   ← Stage 3+4 orchestrator (train → seg → eval)
│   ├── preprocessing/          ← Stage 1: SfM
│   │   ├── run_preprocessing.py
│   │   └── markers/            ← coded-marker steps (+ markers/legacy/)
│   ├── mask_generation/        ← Stage 2: YOLO/SAHI + SAM
│   ├── reconstruction/         ← 3DGS training internals
│   ├── segmentation_3d/        ← FlashSplat 3D segmentation + evaluation
│   ├── gaussians/              ← shared 3DGS primitives (renderer, CUDA)
│   ├── wheat_utils/            ← shared helpers (paths, eval split)
│   ├── viewer/                 ← viser viewer + render_360.py
│   ├── analysis/               ← analysis + QA helper scripts
│   ├── phenotyping/            ← length/width/volume (see caveats)
│   └── hydra_plugins/          ← Hydra config-path plugin (internal)
├── input_plots/            ← input data (not in the repo)
└── results/                ← pipeline outputs (not in the repo)
```

## Where to read more

Every stage folder has its own README with how to run that stage, what it needs on
disk, what it writes, and the settings you are most likely to change. Start there
when you want to change how a stage behaves.

| README | What it covers |
|--------|----------------|
| [src/preprocessing/](src/preprocessing/README.md) | Stage 1: uniform cropping and COLMAP SfM for phone sessions. |
| [src/preprocessing/markers/](src/preprocessing/markers/README.md) | The coded ground markers: detection, triangulation, and what they are used for. Off by default. |
| [src/mask_generation/](src/mask_generation/README.md) | Stage 2: the detectors, the SAM step, and the mask granularity options. |
| [src/reconstruction/](src/reconstruction/README.md) | Stage 3: 3DGS training, plus the eight steps of the orchestrator that runs stages 3 and 4. |
| [src/segmentation_3d/](src/segmentation_3d/README.md) | Stage 4: the match-and-fine-tune loop, its outputs, and the two evaluation steps. |
| [src/viewer/](src/viewer/README.md) | The browser viewer and the 360 flyaround video. |
| [src/analysis/](src/analysis/README.md) | Standalone analysis scripts, not part of a run. Read the notes at its top before running one. |

## Data layout

`input_plots/` and `results/` are not in the repository. This is how the pipeline
expects them to be arranged.

**`input_plots/` — the input data:**

```
input_plots/
├── fip/                        ← provided (from the Wheat3DGS repo)
│   └── <plot>/                 ← one folder per plot (e.g. plot_461)
│       ├── images/             ← the plot images
│       ├── sparse/             ← COLMAP SfM (poses + points)
│       └── manual_label/       ← ground truth (for evaluation)
│
└── phone/
    └── <field>/<session>/      ← one folder per field and session date
        ├── input/              ← raw captures (not released yet)
        ├── manual_label/       ← ground truth (not released yet)
        ├── input_uniform/      ← generated by preprocessing
        ├── images/             ← generated: undistorted images the pipeline uses
        └── sparse/0/           ← generated: COLMAP SfM
```

For FIP the data comes ready to use and skips SfM. For phone, only `input/` and
`manual_label/` are inputs and preprocessing generates the rest. The phone data itself
is not released yet, see the caveats at the end.

**`results/` — created by the pipeline:**

```
results/
├── mask_generation/
│   ├── fip/<plot>/<method>/<mask_exp>/
│   │   ├── bboxes/             ← wheat-head boxes
│   │   └── masks/              ← per-head 2D masks
│   │
│   └── phone/<field>/<session>/<method>/<mask_exp>/
│       ├── bboxes/
│       └── masks/
│
└── reconstruction/
    ├── fip/<plot>/vanilla_3dgs/<recon_exp>/
    │   ├── point_cloud/               ← trained 3DGS model
    │   ├── train/  test/              ← rendered views
    │   ├── results.json               ← reconstruction metrics
    │   └── segmentation_3d/<seg_exp>/
    │       ├── gaussians_colored.ply  ← 3D model colored per head
    │       ├── all_obj_labels.pth     ← per-Gaussian head IDs
    │       └── eval_2d/               ← 2D evaluation results
    │
    └── phone/<field>/<session>/vanilla_3dgs/<recon_exp>/
        ├── point_cloud/
        ├── train/  test/
        ├── results.json
        └── segmentation_3d/<seg_exp>/
```

These folders appear automatically as each stage runs. Only the main outputs are
shown here. Each experiment folder also holds extra files and folders, such as
visualizations, the saved config, and logs. The three experiment names are separate
settings and may be the same or different, see the Quickstart below.

## Quickstart

Stages 2 to 4 support two datasets, **phone** and **fip**, and each one has its own
configuration bundled into a Hydra *profile*. You pick the dataset with a
single switch, `profile=phone` (the default) or `profile=fip`, and the profile sets
the right dataset, defaults, and per-stage values for you. Stage 1 has no profile,
since SfM only runs on phone data, and it takes the session directly with `field=`
and `plot=`.

Every stage writes into a folder named by `experiment_name`, and `thesis_baseline` is
the default name in mask generation, 3DGS reconstruction, and 3D segmentation. When the
names match, the stages find each other's outputs and chain automatically. They may also
differ, since 3D segmentation picks its mask input by name
(`segmentation_3d.mask_gen_experiment`), so one mask generation run can feed several
3DGS reconstruction and 3D segmentation runs without being recomputed.

### Stage 1 — SfM (phone only)

Recover the camera poses and calibration for one phone session:

```bash
python src/preprocessing/run_preprocessing.py field=field_A plot=20250715
```

### Stage 2 — Mask generation

Detect and segment the wheat heads:

```bash
python src/mask_generation/run_mask_generation.py                 # phone (default)
python src/mask_generation/run_mask_generation.py profile=fip     # fip
```

By default this runs on every plot or session in the dataset. Add
`dataset.plot_glob` to target a single one:

```bash
# one phone session
python src/mask_generation/run_mask_generation.py dataset.plot_glob=field_A/20250715
# one FIP plot
python src/mask_generation/run_mask_generation.py profile=fip dataset.plot_glob=plot_461
```

To use a different detector, override the composed method (here, SAHI):

```bash
python src/mask_generation/run_mask_generation.py method=sahi_yolo_sam
```

### Stages 3 + 4 — 3DGS reconstruction and 3D segmentation

The orchestrator's steps are all off by default, so enable the ones you want with
`run_*` flags. You can combine steps in one command, or run them one at a time:

```bash
# train, segment and evaluate in a single command (phone default)
python src/run_reconstruction.py run_train=true run_seg=true run_eval=true

# or one at a time — 3D segmentation reuses the already-trained model
python src/run_reconstruction.py run_train=true                 # train only
python src/run_reconstruction.py run_seg=true run_eval=true     # then segment + eval
python src/run_reconstruction.py profile=fip run_train=true run_seg=true  # fip
```

Any config value can be overridden on the command line, so a specific plot, session,
or run name is just an extra argument:

```bash
python src/run_reconstruction.py run_train=true profile=fip plot=plot_463
python src/run_reconstruction.py run_train=true plot=field_A date=20250627 experiment_name=my_run
```

### Full pipeline for one session

Run one session (`field_A` / `20250715`) through all stages. The argument names
differ between stages for historical reasons, so follow the comments:

```bash
# 1. SfM (phone only) — here plot is the session date
python src/preprocessing/run_preprocessing.py field=field_A plot=20250715

# 2. mask generation
python src/mask_generation/run_mask_generation.py dataset.plot_glob=field_A/20250715

# 3+4. 3DGS reconstruction and 3D segmentation — here plot is the field, date the session
python src/run_reconstruction.py run_train=true run_seg=true plot=field_A date=20250715
```

## Thesis baseline configuration

The two profiles carry the `thesis_baseline` configuration we settled on for the
thesis, so the defaults already match what we actually ran:

- **SfM**: ALIKED + LightGlue front end, OPENCV camera model, exhaustive matching.
  - **Markers**: the coded-marker steps in preprocessing are **off by default**
    (`run_markers=false`). When turned on, they already have their good settings wired in.

- **Mask generation**: YOLOv5 detection followed by SAM. Phone runs YOLOv5 at full
  resolution (4032 px) with a strict 0.70 confidence threshold, and SAM2 at per-tile
  granularity. Per-head granularity scores slightly higher but runs much slower, so
  per-tile is the default. FIP runs YOLOv5 at 1280 px with a 0.35 confidence
  threshold, and SAM1 at the default full-frame granularity.

- **Reconstruction**: gsplat engine, principal point honored, 15k iterations,
  resolution 1. FIP additionally uses AbsGS densification for finer detail. Phone
  uses the standard densification threshold.

- **3D segmentation**: fine-tune match IoU threshold 0.6, all runtime
  optimizations on (including frustum culling).

## Caveats

- **Phenotyping is not adapted.** `src/phenotyping/` (length, width, volume) is
  carried over from the original Wheat3DGS and was not part of this phone adaptation,
  so it is untested here and may not run as-is.

- **The marker step is specific to our capture setup.** The coded-marker pipeline
  under `src/preprocessing/markers/` expects our field's markers and survey files.
  It is off by default and not needed for a normal run.

- **Phone mask generation uses a marker-based ROI tuned to our field.** `roi.enabled` is
  on by default, and it restricts detection to our region of interest. The region is built
  from the triangulated ground markers, so it needs the preprocessing marker steps
  (`run_markers=true`) to have run first. Those are off by default, so on a plain run, and
  on any data without our markers, the ROI falls back to covering the whole image and mask
  generation runs as normal. On FIP it does nothing either way. You can also force it off
  with `roi.enabled=false`.

- **Datasets are not included.** The phone images and their labels are not released
  yet and may follow later. The FIP data is available from the original Wheat3DGS
  repository ([github.com/zdwww/Wheat-3DGS](https://github.com/zdwww/Wheat-3DGS)). The
  ground-truth evaluation steps need that data.

- **Match ground truth to the undistorted images.** The SfM undistortion step crops
  the images to a different size. Ground truth labeled on the original images will not
  match that size, so some evaluation steps might not be valid. The ground truth
  should be labeled on the same images the pipeline uses, not the original captures.

- **Private submodules are skipped on a public clone.** The `docs/` and `thesis/`
  folders are private submodules. A plain `git clone` skips them cleanly, and the
  pipeline still runs without them.

- **Model weights are not shipped.** The YOLO and SAM checkpoints under
  `src/mask_generation/weights/` are not included in the repository — see
  [INSTALL.md](INSTALL.md) for how to get them.
