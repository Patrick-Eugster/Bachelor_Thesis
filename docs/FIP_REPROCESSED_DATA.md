# FIP single-row 2024 reprocessed data — what it is + 3 planned experiments

New supervisor drop: `demoanlage2025_v0/FIP_single_row_exp_2024_reprocessed/` (gitignored, local-only).
Reprocessed FIP 2024 single-row plots in Agisoft (Metashape ≥2.1.0 with automatic COLMAP export).
**Not yet wired into the pipeline** — this doc records the contents and the experiments it unlocks.

## Contents

- **7 plots** `plot_461 … plot_467`. Each has `colmap_reprocessed/` with **4 variants**, every one a ready
  `images/` + `sparse/0/` (Agisoft COLMAP **.txt** export, **36 cameras**):
  `undistorted_jpg`, `undistorted_png`, `distorted_jpg`, `distorted_png`.
- **`wheat_head_counts.xlsx`** — manual GT head counts **per row** (row1…row6) per plot. Plot totals:
  461=776, 462=764, 463=807, 464=693, 465=707, 466=729, 467=808. Comments note boundary heads excluded.
- **`distances.json`** — manual **tape distances** per plot between markers (`1-2`, `2-3`, `1-3`) in cm.
- **`<plot>/marker_projections.csv`** — Agisoft marker 2D pixels (cols `Marker,Camera,X,Y,Pinned`), in
  **undistorted**-image space (use `cv2.undistortPoints` for the distorted variants).

### Markers are DIFFERENT from phone
FIP markers are **`target 1/2/3`** (plot_467 also `target 8`), only **3 per plot** — **not** the phone
12-bit coded IDs `{77,85,89,101,105,113}`, and they look non-coded. So **our CCT detector
(`detect_markers_v8_cct`) does NOT apply to FIP** (different IDs, no decodable manifest). FIP is also 36
**overhead** views, not a continuous walk, so the open-walk pose-drift story
([PHONE_SFM_POSE_ACCURACY.md](PHONE_SFM_POSE_ACCURACY.md)) doesn't transfer.

### Practical gotchas
- **1128 `*:Zone.Identifier` files** (Windows ADS from the WSL copy) litter the folder — delete before use
  (`find … -name '*Zone.Identifier*' -delete`); harmless but clutters globs.
- **jpg vs png filenames differ**: jpg has an inserted `_6_` (`FPWW036_SR0461_6_FIP2_cam_04`) vs png
  (`FPWW036_SR0461_FIP2_cam_11`). Both end in `_cam_NN` so the eval-split regex works, but cross-variant
  matching must key on `_cam_NN`, not the full name.

## Planned experiment 1 — head count vs GT (prep DONE, compare TODO)
Compare our **predicted plot head count** against the xlsx GT total per plot. Granularity matters: the xlsx
is a **plot-level total** (Σ of 6 rows), while `eval_seg_2d.py` does **per-image mask-level** counts. So the
comparison point is the **number of unique 3D head instances** from `run_3d_seg.py` (= `num_wheat_head`),
not `eval_seg_2d`.

> ### ⚠️ REPORT CAVEAT — the xlsx count and our count are NOT 1:1 comparable
> The two numbers count over **different boundaries**, so a perfect pipeline still won't match the xlsx
> exactly — do **not** present this as a direct accuracy figure:
> - **xlsx GT** = heads the human judged clearly **inside** the plot, with **ambiguous boundary heads
>   between adjacent plots deliberately excluded** (the per-plot comments say so, e.g. *"8 heads between
>   461 and 460 not counted"*). So the GT undercounts the true physical heads by a handful of seam heads.
> - **Our count** = distinct 3D head instances inside the **reconstructed volume** — a region defined by
>   what the 36 cameras saw and what reconstructed well, **not** by the human's plot boundary. It may
>   include seam heads the human excluded, or drop poorly-covered edge heads.
>
> ⇒ Use it as a **relative / correlation** check (do our counts *track* the GT across the 7 plots? is the
> offset roughly constant?), **not** an exact per-plot accuracy number. State this caveat explicitly in the
> report so the comparison isn't over-read.

**Prep built this session:**
- `run_3d_seg.py` now writes **`segmentation_3d/<exp>/seg_summary.json`** with `wheat_heads_found` (the
  predicted plot total) + mask-match stats — previously the count only went to stdout/wandb.
- **`src/segmentation_3d/headcount_gt.py`** — `load_headcount_gt(xlsx)` → `{plot: {total, rows, comments}}`
  (+ CLI to print the table). Reads the xlsx via `openpyxl` (now in `pyproject.toml`).

**Still TODO (write at run time):** a small compare script that joins `seg_summary.json` (predicted) with
`load_headcount_gt()` (GT) per plot → error / ratio table.

## Planned experiment 2 — FIP metric benchmark via tape (SPEC, write at run time)
Run our COLMAP (ALIKED + exhaustive — cheap on 36 views, no drift concern) on the FIP `distorted_*` images
and benchmark vs Agisoft, the same as the phone sessions. **The marker-ID mismatch is NOT blocking** — skip
our CCT detector and use the **Agisoft-provided** `marker_projections.csv` + `distances.json`.

**Precise spec of the marker-scale adapter (`fip_marker_scale.py`, to write when we run):**
1. **Input:** a staged FIP plot (via `stage_fip_variant.py`) → has `sparse/0/` (poses) +
   `marker_projections.csv`; plus `distances.json` (root) for that `plot_id`.
2. **Parse markers:** read `marker_projections.csv` (`Marker,Camera,X,Y,Pinned`). Map `target N` → integer
   `N` (so `target 1,2,3 → 1,2,3`). Keep `Pinned=True` rows (all of them are anyway). Group into the
   per-marker observation list `{marker_id: [{cam, xy}]}` — **the exact format `triangulate_markers.py`
   already consumes** (so reuse its DLT+RANSAC triangulator rather than re-implementing).
3. **Coordinate frame:** the csv pixels are in **undistorted** space. So run this on an **`undistorted_*`
   staged variant** (pixels match `sparse/0/` directly). If a `distorted_*` variant is ever used, first
   `cv2.undistortPoints` the marker pixels with that variant's intrinsics.
4. **Camera name match:** the csv `Camera` must map to `sparse/0/images.txt` names — watch the jpg `_6_`
   vs png filename difference (key on the `_cam_NN` tail if needed).
5. **Triangulate + scale:** triangulate each of the 3 markers → 3D points; for pairs `1-2,2-3,1-3` compute
   reconstructed distance, and `scale = median(tape_cm / recon_dist)` from `distances.json` (cm → m). Write
   a `metric_frame.json` analog (scale, per-pair residual, CV).
6. **Caveats to expect:** only **3 markers / 3 pairs** ⇒ noisier CV than phone's 6; the 3 ground markers in
   a single-row plot may be near-**collinear** ⇒ weak triangulation geometry → sanity-check parallax. This
   makes FIP a phone-style SfM/metric benchmark instead of only a pre-calibrated input.

## Planned experiment 3 — jpg vs png (and distorted vs undistorted) (staging DONE, A/B TODO)
Cleanest controlled study: identical scene + identical Agisoft poses, only the **image encoding** differs.
Two axes: **jpg (lossy) vs png (lossless)**, and **distorted vs undistorted**. Each variant ships its own
`sparse/0/`, so **skip COLMAP and train 3DGS directly** on each variant's `images/` + `sparse/0/`, then
compare PSNR/SSIM/LPIPS + segmentation + head count. Finally *measures* whether JPEG compression hurts the
pipeline (what `preprocess_uniform_size.py`'s `quality="keep"` guards against blindly).

**Staging built this session:** `src/preprocessing/stage_fip_variant.py` copies (or `--link`) a chosen
`plot`+`variant` out of `demoanlage/` into `input_plots/fip/<plot>_<variant>/` (`images/` + `sparse/` +
`marker_projections.csv`), so the data is self-contained in `input_plots/` — e.g.
`--plot 461 --variant all`.

**Still TODO (write at run time):** the A/B driver that loops the 4 staged variants through the pipeline and
collects PSNR/SSIM/LPIPS + count. **Important — the full chain is NOT just "train 3DGS":** each variant
needs its own **YOLO+SAM masks** (`run_mask_generation.py` on that variant's images) → 3DGS → seg → count;
masks differ per variant. And for the PSNR numbers to be comparable across variants, all must use the
**same held-out test split** (the staged dirs have no `transforms.json`, so either stage the paper
`transforms.json` too or pin the split explicitly — otherwise the FIP `_cam_11/_cam_12` regex fallback must
be confirmed identical across variants). Mind the jpg `_6_` vs png filename difference when pairing cameras.

## Open questions / things to verify before running (answer to "am I missing something?")
1. **Same plots as the existing `input_plots/fip/plot_461…467`? — CONFIRMED YES.** 36/36 exact
   camera-name overlap (`FPWW036_SR0461_1_FIP2_cam_01…`) between `input_plots/fip/plot_461/images` and the
   reprocessed `plot_461`. So the existing `manual_label/` GT masks **and** the new head-count xlsx describe
   the **same heads** — per-image mask GT *and* plot-count GT for one scene, can be cross-used.
2. **Head-count population mismatch.** xlsx = every physical head in 6 rows of the *full* plot (with boundary
   heads explicitly excluded in the comments); our count = heads in the *reconstructed volume*. These are
   not the same set → expect a systematic offset; use it as a relative/correlation check, set expectations.
3. **Exp-2 vs exp-3 test different things — don't conflate.** Exp-2 benchmarks **our COLMAP** vs Agisoft
   (uses our SfM). Exp-3 uses **Agisoft's** shipped `sparse/0/` for *both* jpg and png (our COLMAP not
   involved) — it isolates the image encoding only.
4. **Marker geometry (exp-2).** 3 markers in a single-row plot may be near-collinear → weak scale geometry;
   check parallax / condition before trusting the scale.

## Files
- Data root: `demoanlage2025_v0/FIP_single_row_exp_2024_reprocessed/` (gitignored).
- Existing FIP pipeline input: `input_plots/fip/plot_461/` (has `manual_label/` GT masks).
- Related: [PHONE_SFM_CAMERA_MODEL.md](PHONE_SFM_CAMERA_MODEL.md) (distorted/undistorted + jpg quality),
  [src/segmentation_3d/eval_seg_2d.py](../src/segmentation_3d/eval_seg_2d.py) (head-count eval).
