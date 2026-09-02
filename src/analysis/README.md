# `src/analysis/` — optional analysis & QA helpers

> ## ⚠️ Read this before running any script here
>
> **These scripts were written for our own runs and most of them need adjusting before they work
> for you.**
>
> - **Output paths.** Most write their results into `docs/analysis_results/`, which is a private
>   submodule folder that a public clone does not have. The output path needs to be adjusted before
>   you run one. Some scripts take an `--out` option, others have the path written near the top of
>   the file.
> - **Hardcoded runs.** Several carry a built-in list of the sessions or experiments they score.
>   Point that list at your own data first.
> - **Local reference data.** The marker and SfM geometry evaluations read survey coordinates and
>   tape measurements that are not part of this repo, so they only run where that data is present.
>
> Open a script and read its top before you run it. None of them are needed for a normal pipeline
> run.

These are small, standalone scripts for **inspecting and sanity-checking** the pipeline's output —
they are *not* part of a normal run. You never need them to go from images to a segmented 3D model.
Reach for one when you want to compare two runs, score quality, visualise a step, or prepare
ground-truth labels. Each script is run on its own from the repo root (most take `--help`), and the
ones that save numbers/plots write into `docs/analysis_results/` by default.

> Note: this folder used to hold ~130 thesis-specific figure/table/diagnostic scripts. Most have been
> moved out to keep the public repo clean. (The rest are kept
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
  the same session, to see which gives better camera poses. This one needs the `hloc` package
  installed separately, since it is not part of the pipeline's own requirements, and it crashes on
  startup without it.

- **fip_principal_point_offset.py** — measure how far each camera's principal point sits from the image
  center, across a dataset. This is where the "off-center by up to ~90 px" figure comes from, and it
  reads the COLMAP `sparse/0` cameras.

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

- **make_gt6_manifest.py** — build the small ground-truth manifest (the labeled images plus their
  per-head instance masks and boxes) that `sweep_conf_mask_ap.py` reads. Re-run it when the ground-truth
  set changes.

## 3D segmentation

- **phone_seg_instance_eval.py** — per-instance (per-head) evaluation of the phone 3D segmentation. Each
  predicted head in the 2D projection is Hungarian-matched one-to-one against the manual per-head
  ground-truth instance masks, and it reports head-count precision/recall/F1, matched-mask IoU, and
  merge/split counts. This is the per-instance seg-quality metric behind the phone results.

- **phone_seg_cpu_eval.py** — pixel evaluation of the phone 3D segmentation (IoU/precision/recall/F1 of
  the wheat-head area) against the ground-truth mask, and the shared run registry that the instance eval
  reads. The companion binary-foreground scorer to the instance eval.

- **fip_seg_marker_excluded_eval.py** — recompute the FIP 3D segmentation 2D metrics with the coded-marker
  disks cut out, so the score is not helped or hurt by the markers.

- **fip_seg_marker_masked_eval.py** — the same marker-excluded FIP seg score, but it finds the marker
  plates by detecting their white blobs instead of using the survey disks.

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

## Markers & SfM geometry

These evaluate the coded-marker geometry and the SfM pose quality against physical ground truth. They read
the private supervisor reference data (survey coordinates and tape measurements), so they run only where
that reference tree is present.

- **rescore_models_geometry.py** — score different SfM front-ends by triangulating the coded markers
  through each model's poses and comparing all pairwise marker distances to the physical survey and tape
  ground truth, in centimetres. The "which SfM front-end is best" evaluation.

- **eval_marker_geometry_gt.py** — detection-free marker geometry error: triangulate the verified marker
  pins through our poses and compare the pairwise distances to survey and tape.

- **eval_marker_detection.py** — marker-detector accuracy in 2D (recall, precision, localization error in
  pixels) against hand-verified marker pins, per session.

- **marker_geometry_gt.py** — validate the triangulated marker geometry against the surveyed XYZ and tape
  measurements using all pairwise marker distances.

- **compare_sfm_models_markers.py** — compare SfM models by marker reprojection error against the Agisoft
  reference, with a shared triangulation so the comparison is fair.

- **marker_cross_session_repeatability.py** — measure how repeatable the triangulated marker positions are
  across sessions, the source of the marker repeatability number.

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

- **collect_experiment_results.py** — harvest every reconstruction, 3D segmentation, and evaluation run
  into one master results table (CSV and markdown), reading each run's saved config and metrics. A
  pipeline-wide results tracker for when you have many experiments.
