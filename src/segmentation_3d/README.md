# 3D Segmentation — `src/segmentation_3d/`

## Overview

Assigns consistent 3D wheat head IDs to Gaussians across all views. Takes the trained 3DGS model (step 1) and the 2D masks from YOLO+SAM (step 2) as input.

Run via `src/run_reconstruction.py` with `run_seg=true` — not directly. The segmentation script (`run_3d_seg.py`) is orchestrated by the pipeline.

---

> **Want to understand the algorithm, not just run it?** See the deep-dive explainer [`docs/segmentation_3d/SEGMENTATION_3D_EXPLAINED.md`](../../docs/segmentation_3d/SEGMENTATION_3D_EXPLAINED.md) — the FlashSplat lift mechanics, full pseudo-code, the `processed`/`buffered` trust gate, the overlap-merge rule, and where the runtime goes.

## How It Works — Iterative Match-and-Fine-Tune

For each 2D mask (one wheat head instance in one camera view):

1. **Lift to 3D** — run FlashSplat ILP solver on one view to assign Gaussians to this mask
2. **Project to all views** — render the lifted Gaussians into every other camera
3. **Find best match** — compare projected mask against all unassigned 2D masks across views; accept if precision > 0.8
4. **Fine-tune** — re-optimize with all matched views combined for a more accurate 3D assignment
5. **Save** — write the Gaussians for this head to `ply/wh_{id}.ply` and record 2D label maps in `2DSeg/`

Repeat until all masks are processed. Each accepted head gets a unique integer ID.

**Overlap handling:** if a new mask overlaps significantly with an already-assigned head, it updates that head's Gaussians instead of creating a new one. The PLY is saved with a letter suffix (e.g. `wh_0042_b.ply`).

---

## Scripts

| Script | What it does |
|--------|--------------|
| `run_3d_seg.py` | Main segmentation loop — iterates over all masks, runs match-and-fine-tune |
| `export_colored_ply.py` | Bakes per-head HSV colors into `gaussians_colored.ply` — auto-run after step 4 |
| `eval_wheatgs.py` | Step 6: renders overlay PNGs and binary segmentation masks per camera into `test/segmentation/` |
| `eval_seg_2d.py` | Step 6b: pixel-level 2D metrics (IoU/P/R/F1/MSE/SSIM/MCC/BalAcc/FPR + head counts) vs manual GT masks |

---

## Configuration

Controlled via `configs/reconstruction_seg3d/segmentation_3d/default.yaml`:

```yaml
exp_name: "run_1"           # subfolder name inside segmentation_3d/ — change to re-run without overwriting
mask_gen_experiment: "initial"   # which mask-generation run to read masks from
save_vis_overlay: true      # save colored overlay images per head (good for debugging)
vis_max_heads: 10           # only save overlays for first N heads (0 = all — can be slow for 300+ heads)
use_mask_cache: true        # crop-cache speedup (see below); true = on (recommended), lossless
seg_seed: 0                 # fixed seed for the mask-processing order → reproducible seg + valid A/B
wandb_enabled: false        # log per-head progress to wandb.ai
```

**`exp_name`** is the key parameter — it controls the output subfolder so you can re-run segmentation with different settings on the same trained model without retraining:

```bash
python src/run_reconstruction.py run_seg=true exp_name=run_2
python src/run_reconstruction.py run_seg=true exp_name=iou05 --iou_threshold 0.5
```

### ⚠️ `use_principal_point` MUST match how the model was trained

The seg **renders** the trained Gaussians (via flashsplat) to match them across views. That render must use the **same `use_principal_point`** the model was *trained* with, or every projected blob shifts a few px, cross-view IoU falls under the 0.5 match threshold, and the result collapses (we saw **IoU 0.565 → 0.117** from exactly this). The flag reaches seg through `run_reconstruction.py` (`+ pp_flag`), driven by `reconstruction.use_principal_point`.

- **FIP:** models are trained `use_principal_point=true` (the Round-4 pixel-shift fix), so **seg must also pass `reconstruction.use_principal_point=true`** — e.g. `python src/run_reconstruction.py run_seg=true reconstruction.use_principal_point=true`. A full-pipeline run (train+seg in one call) is auto-consistent; only a **separate** seg run on a pre-trained model needs the flag set manually.
- **Phone:** trained (and segged) `false` — principal point ≈ image center, so it's a near-no-op; keep both at the default `false`.

Rule of thumb: **seg's `use_principal_point` == training's.** See [`docs/segmentation_3d/CROP_CACHE_OOM_AND_IOU_DEBUG.md`](../../docs/segmentation_3d/CROP_CACHE_OOM_AND_IOU_DEBUG.md).

### Crop-cache speedup (`use_mask_cache`, default on)

`find_match` used to re-decode each full-res mask PNG per candidate per head — the dominant cost on dense (phone/SAHI) data (a run was ~6% done after 48 h). The crop cache pre-decodes every mask **once** at startup into a compact tight-bbox crop kept in CPU RAM (~200 MB for 22 k phone masks), and IoU is computed on the crop — **provably byte-identical** to the old full-frame path (validated on the FIP GT A/B: same `all_obj_labels.pth` md5; offline tests `src/analysis/verify_crop_iou.py` + `verify_numpy_build.py`). Result: the phone run that was stuck at 6%/48 h now finishes in **~18.5 h** (~40×). Turn off with `segmentation_3d.use_mask_cache=false` (or env `WHEAT_SEG_NO_CACHE=1`) — then no cache is built and `find_match` falls back to the original per-candidate decode (used by the A/B baseline). `WHEAT_SEG_TIMING=1` prints the render-vs-match split. Design + the render-side Phase-2 follow-ups: [`docs/segmentation_3d/SEGMENTATION_3D_RUNTIME.md`](../../docs/segmentation_3d/SEGMENTATION_3D_RUNTIME.md).

---

## Output Structure

```
results/reconstruction/fip/{plot}/vanilla_3dgs/{experiment}/segmentation_3d/{exp_name}/
├── gaussians.ply            ← all segmented Gaussians (fine-tuned, with obj labels)
├── gaussians_colored.ply    ← same but with per-head HSV colors baked in (for viewer)
├── all_obj_labels.pth       ← per-Gaussian wheat head ID tensor
├── all_counts.pth           ← FlashSplat contribution counts
├── results.csv              ← per-head: ID, source mask, view count, Gaussian count
├── seg_summary.json         ← run totals: wheat_heads_found (= predicted PLOT head count), masks_matched/unmatched, total_masks
├── experiment.txt           ← run metadata
├── 2DSeg/                   ← per-camera 2D label maps (.pt, one per image) — used by eval
├── ply/                     ← per-head PLY files: wh_0001.ply, wh_0042_b.ply (overlap suffix)
├── img/                     ← overlay visualizations per head (only if save_vis_overlay=true)
└── eval_2d/                 ← output of eval_seg_2d.py (step 6b)
    ├── metrics_2d.json      ← IoU/Precision/Recall/F1 per annotated camera
    └── {stem}_eval2d.png    ← color-coded visualization: red=TP, gray=FN, yellow=FP
```

The `2DSeg/` folder is the key output used by `eval_wheatgs.py` and the viewer — it contains the final 2D projection of all 3D assignments per camera.

---

## Eval — Step 6 (`eval_wheatgs.py`)

Renders the 3D segmentation back to 2D — produces overlay PNGs per camera and binary segmentation masks into `test/segmentation/`. Required before step 6b.

```bash
python src/run_reconstruction.py run_eval=true
```

---

## Eval 2D — Step 6b (`eval_seg_2d.py`)

Computes pixel-level segmentation metrics by comparing the pipeline's 3D segmentation (projected back to 2D by step 6) against manually annotated ground truth masks. This is the same evaluation methodology as reported in Table 2 of the Wheat3DGS paper.

**Dependency:** requires `scikit-image` (`pip install scikit-image`) — listed in `pyproject.toml`, installed automatically with `pip install -e .`.

### How to run

```bash
# if step 6 was already run (test/segmentation/ already exists):
python src/run_reconstruction.py run_eval_2d=true

# if step 6 hasn't been run yet, run both together:
python src/run_reconstruction.py run_eval=true run_eval_2d=true
```

Step 6b depends on step 6 — it reads the binary segmentation PNGs that `eval_wheatgs.py` writes to `test/segmentation/`. If step 6 was already run for the current experiment, you can skip `run_eval=true`.

### What it measures

**Important: all metrics operate on collapsed binary masks — this is not per-instance evaluation.**

Both the GT mask and the prediction are single binary images where every wheat head pixel is 1 and background is 0, regardless of how many individual heads are present. The metrics answer "did the pipeline find wheat head area in the right places?" — not "did it correctly identify each individual head as a separate instance?".

This means two runs can score identically even if one correctly separates 80 individual heads and another produces one big merged blob covering the same area. Per-instance evaluation would require GT masks that label each head separately, which the current `*_gt_mask.png` files do not provide.

This is the same methodology the paper uses for Table 2, so the numbers are directly comparable.

**Pixel metrics** (match paper Table 2 directly):

| Metric | Formula | Meaning |
|--------|---------|---------|
| IoU | TP / (TP + FP + FN) | overlap quality — main headline metric |
| Precision | TP / (TP + FP) | of all predicted wheat pixels, how many are correct |
| Recall | TP / (TP + FN) | of all true wheat pixels, how many were found |
| F1 | 2·P·R / (P+R) | harmonic mean of precision and recall |
| MSE | mean((gt−pred)²) | fraction of pixels where GT and pred disagree — can look good despite poor segmentation due to class imbalance (far more background than foreground) |
| SSIM | structural similarity | captures spatial structure quality — more sensitive to shape and layout than MSE |

Paper results for comparison (Table 2): IoU=0.50, Precision=0.81, Recall=0.57, F1=0.67, MSE=0.06, SSIM=0.90

**Imbalance-robust metrics** (beyond the paper — more meaningful when background >> foreground):

| Metric | Formula | Meaning |
|--------|---------|---------|
| MCC | (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Matthews Correlation Coefficient — uses all four quadrants including TN; range −1 to +1, higher is better; more robust than F1 for imbalanced data |
| Balanced Accuracy | (Recall + Specificity) / 2 | Average of how well wheat is detected AND how well background is rejected — not skewed by class imbalance |
| FPR | FP / (FP + TN) | False Positive Rate — fraction of background pixels wrongly predicted as wheat; important for phenotyping where false alarms matter |

**Head count metrics** (instance-level — invisible to pixel metrics):

| Metric | How computed | Meaning |
|--------|-------------|---------|
| GT head count | lines in `manual_label/{stem}.txt` | wheat heads manually annotated in this camera view |
| Pred head count | distinct non-zero IDs in `2DSeg/{stem}.pt` | separate wheat head instances the pipeline found in this view |
| Count error ratio | (pred − GT) / GT | normalized count difference — negative = under-count, positive = over-count; e.g. −0.19 = found 19% fewer heads than GT |

> **Per-view vs plot-total — pick the right number.** These `eval_seg_2d` counts are **per-camera** (heads in *one* labeled view vs the YOLO `.txt` for that view) and only exist for images with `manual_label/` GT. For a **plot-level total** head count (e.g. vs the FIP `wheat_head_counts.xlsx` GT), use `run_3d_seg`'s **`seg_summary.json → wheat_heads_found`** (= number of unique 3D head instances = `num_wheat_head`), NOT the per-view `eval_seg_2d` counts. The xlsx comparison is relative-only — see the boundary/reconstructed-volume caveat in [docs/data/FIP_REPROCESSED_DATA.md](../../docs/data/FIP_REPROCESSED_DATA.md).

### Inputs

**GT masks** — `input_plots/{dataset}/{plot}/manual_label/`

Each plot's `manual_label/` folder contains 4 files for one annotated camera view:
```
{stem}.txt               ← YOLO-format bounding box annotations (used by YOLO metrics)
{stem}_gt_mask.png       ← binary mask: white=wheat head area, black=background  ← used here
{stem}_gt_bbox.png       ← visualization of bounding boxes on the image
{stem}_overlay.png       ← visualization of masks overlaid on the image
```

All 7 FIP plots (461–467) have GT masks pre-provided by the paper authors. One camera per plot is annotated. Phone data has no GT masks yet.

**Prediction** — `results/.../test/segmentation/{stem}.png`

Written by step 6 (`eval_wheatgs.py`). The stem is derived from the camera name in the GT mask filename so they match automatically. The script also checks `train/segmentation/` as fallback in case the annotated camera ended up in the training split.

### Outputs

All outputs go to `segmentation_3d/{exp_name}/eval_2d/`:

**`metrics_2d.json`** — one entry per annotated camera:
```json
[
  {
    "iou": 0.4821,
    "precision": 0.7934,
    "recall": 0.5612,
    "f1": 0.6571,
    "mse": 0.0612,
    "ssim": 0.8934,
    "mcc": 0.6103,
    "balanced_acc": 0.7756,
    "fpr": 0.0248,
    "gt_head_count": 47,
    "pred_head_count": 38,
    "count_error_ratio": -0.191,
    "camera": "FPWW036_SR0461_FIP2_cam_12"
  }
]
```

**`{stem}_eval2d.png`** — color-coded pixel mask blended over the raw RGB image (alpha=0.6):

| Color | Pixels | Metric category |
|-------|--------|-----------------|
| Red | GT=wheat AND pred=wheat | True Positive |
| Gray | GT=wheat AND pred=background | False Negative (missed) |
| Light yellow | GT=background AND pred=wheat | False Positive (wrong) |
| No overlay | GT=background AND pred=background | True Negative |

### FruitNeRF comparison

The FruitNeRF baseline comparison branch is preserved as commented-out code in `eval_seg_2d.py`. To enable it later: produce `*_fruitnerf.png` prediction masks (same dimensions as the GT mask) and place them in `manual_label/`, then call `eval_seg_2d(..., pred_type="fruit")`.

---

## Logging

Segmentation output is logged to `seg_logs/{exp_name}.txt` inside the reconstruction experiment folder. Controlled by `log_seg_only: true` in `configs/reconstruction_seg3d/segmentation_3d/default.yaml` — when true, only the segmentation step is logged to file (training stays terminal-only).
