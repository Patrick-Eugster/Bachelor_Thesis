# `src/analysis/` — optional analysis & QA helpers

These are small, standalone scripts for **inspecting and sanity-checking** the pipeline's output —
they are *not* part of a normal run. You never need them to go from images to a segmented 3D model;
reach for one when you want to compare two runs, score quality, visualise a step, or prepare
ground-truth labels. Each script is run on its own from the repo root (most take `--help`), and the
ones that save numbers/plots write into `docs/analysis_results/`.

> Note: this folder used to hold ~130 thesis-specific figure/table/diagnostic scripts. Those have been
> moved out to keep the public repo clean; only the reusable helpers below remain. (The rest are kept
> for provenance in the project's private notes.)

---

## SfM & reconstruction

- **analyze_sfm_connectivity.py** — score how well a COLMAP run used and linked the input images.
  Tells you whether the reconstruction actually tied all your photos into one connected model, or
  quietly dropped/split some. A quick health check before you trust an `sparse/` model.

- **analyze_sparseness.py** — measure how "sparse" a reconstruction is (multi-view overlap + camera
  angular diversity). Sparseness isn't just how many photos you took — it's how many cameras see each
  point and from how many directions. The numbers help decide which densification setting suits a
  dataset.

- **run_hloc_sfm.py** — run an alternative SfM front-end (hloc: SuperPoint+LightGlue / LoFTR) and
  write a COLMAP model. Lets you A/B a different feature matcher against the default COLMAP pipeline on
  the same session, to see which gives better camera poses.

## Mask generation (detection + SAM)

- **sweep_conf_mask_ap.py** — sweep the detector's confidence threshold and score the resulting masks
  as instance segmentation (precision / recall / F1 / AP). This is the core mask-quality scorer: point
  it at an experiment and it measures how good the per-head masks are against ground truth across
  confidence levels.

- **plot_conf_sweep.py** — plot and aggregate the JSON files that `sweep_conf_mask_ap.py` writes. Turns
  the raw sweep numbers into readable F1/precision-recall curves so you can pick a good operating point
  at a glance.

- **score_sam_masks.py** — directly score a set of SAM masks against ground truth (IoU / precision /
  recall / F1). Handy for a straight A/B between mask variants (e.g. SAM1 vs SAM2 vs SAM3) without
  running the whole evaluation.

- **compare_yolo_sahi.py** — for one session, compare plain-YOLO vs SAHI-tiled detection box counts and
  plot the confidence distribution. Shows where tiling finds extra (usually small) heads and gives a
  tuning curve of "kept boxes vs threshold" to help set the confidence cut-off.

- **compare_bboxes.py** — diff two `bboxes/` output folders. Use it to confirm a refactor or config
  change didn't alter the detector's output — a few single-box differences are normal (GPU
  non-determinism near the threshold), a large gap means something really changed.

- **viz_sahi_tiles.py** — draw the SAHI tile grid on an image. Purely visual: you see how many tiles
  there are and how much they overlap, which helps sanity-check the tiling settings on a new image
  size.

- **run_yolo11_seg.py** — standalone runner for a YOLO11 instance-segmentation model. A self-contained
  way to try a YOLO11-seg checkpoint on images outside the main mask-generation entry point.

## 3D segmentation

- **compare_seg_runs.py** — check whether two `segmentation_3d` runs produced the *same* result, by
  hashing their outputs. Deterministic yes/no (with an exit code), so you can verify in a script that a
  code change left the segmentation byte-for-byte unchanged instead of eyeballing it.

- **seg_head_size_map.py** — top-down map of head centroids coloured by size. Shows *where* the
  suspiciously large (likely over-merged) heads sit versus normal ones, so a segmentation problem can
  be located spatially in the plot.

- **seg_roi_keep_fraction.py** — check the segmentation's ROI/marker cull on a trained model *without*
  running the (GPU-heavy) segmentation. Reports how many Gaussians survive the region-of-interest and
  marker-exclusion masks, so you can confirm the cull isn't clipping away real wheat heads.

- **recolor_heads_contrast.py** — recolour a segmentation PLY with a high-contrast palette. Neighbouring
  heads get visibly different colours, which makes it much easier to see individual heads when
  inspecting the coloured point cloud.

## Ground-truth labeling prep

- **make_gt_box_seeds.py** — seed ground-truth labeling with the model's own boxes. So a human corrects
  existing boxes instead of drawing every head from scratch — much faster labeling.

- **make_gt_labeling_images.py** — make marker-ROI-masked copies of the images chosen for labeling.
  Greys out everything outside the region of interest so the annotator focuses on the plot area and
  isn't distracted by background.

- **make_cvat_seed_zip.py** — package seed boxes into a CVAT-ready import zip. Bundles the seeded boxes
  in the format the CVAT annotation tool expects, so you can load them straight into a labeling task.

## Experiment bookkeeping

- **aggregate_maskgen_grid.py** — pool the mask-generation grid's evaluation JSONs into one ranked
  table. When you've run a grid of detector/SAM settings, this collects all the per-cell scores and
  ranks them so the best configuration is easy to spot.
