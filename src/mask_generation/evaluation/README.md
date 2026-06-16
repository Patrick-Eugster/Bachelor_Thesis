# Mask-generation Evaluation — quick reference

| file | what it does |
|---|---|
| [`eval_yolo_boxes.py`](eval_yolo_boxes.py) | score **one** method (YOLO *or* SAHI) vs GT — P/R/F1/AP, match_viz, IoU histograms, FP/FN heatmaps, PR curves |
| [`eval_compare_3way.py`](eval_compare_3way.py) | **SAHI vs YOLO vs GT**, head-by-head (7-region Venn) — coverage tables (tertiles+COCO), split/merge, FP breakdown, coverage/FP overlays |
| [`eval_compare_nogt.py`](eval_compare_nogt.py) | **SAHI vs YOLO agreement, no GT** (FIP + phone) — agree / YOLO-only / SAHI-only + overlay |
| [`sahi_merge_debug.py`](../sahi_yolo_sam/sahi_merge_debug.py) | inspect SAHI's tile **merge** — tiles / before / after / clusters + N_raw→N_final counts (re-runs SAHI) |
| `compare_common.py` | shared helpers (not run directly) |

```bash
# 0. PREREQUISITE — produce boxes for BOTH methods once (eval_run sets only_labeled_images etc.)
python src/mask_generation/run_mask_generation.py --config-name eval_run method=yolo_sam_v1   experiment_name=metrics_v2
python src/mask_generation/run_mask_generation.py --config-name eval_run method=sahi_yolo_sam experiment_name=metrics_v2

# 1. one method vs GT  (swap method=yolo_sam_v1 for the YOLO arm)
python src/mask_generation/evaluation/eval_yolo_boxes.py method=sahi_yolo_sam mask_gen_experiment=metrics_v2 eval_experiment=sahi_v2

# 2. SAHI vs YOLO vs GT — the tuning instrument (FIP only)
python src/mask_generation/evaluation/eval_compare_3way.py yolo_experiment=metrics_v2 sahi_experiment=metrics_v2 overlay_mode=both fp_singles=true eval_experiment=cmp_v2

# 3. SAHI vs YOLO agreement, no GT (FIP cross-check, or dataset=phone)
python src/mask_generation/evaluation/eval_compare_nogt.py yolo_experiment=metrics_v2 sahi_experiment=metrics_v2 overlay_mode=both eval_experiment=cmp_v2

# 4. SAHI merge inspector (re-runs SAHI; one image per plot)
python src/mask_generation/sahi_yolo_sam/sahi_merge_debug.py plot_glob=plot_461 limit_images=1 eval_experiment=cmp_v2
```

Full details for each below (`eval_yolo_boxes.py` first, then the comparison tools).

---

# YOLO Bounding Box Evaluation — `eval_yolo_boxes.py`

## Overview

Each FIP plot has **one manually labeled image** in `input_plots/fip/{plot}/manual_label/` — a single representative image with ground-truth wheat head bounding boxes in YOLO format (`class_id cx cy w h`, normalized 0–1). The goal is to compare the YOLO boxes against these labels to measure box quality.

## How to Run

**Step 1 — Run the YOLO mask generation with the eval-run config:**
```bash
python src/mask_generation/run_mask_generation.py --config-name eval_run
```
This loads all settings from `config.yaml` + `method/yolo_sam_v1.yaml` as usual, but automatically overrides the values that must be set for evaluation:
- `only_labeled_images: true` — only processes the manually labeled image per plot, and saves `bboxes_with_conf/` needed for the AP curve
- `limit_plots: 0` / `limit_images: 0` — no limit (overrides any debug value in config.yaml)
- `method.conf_threshold_nms_floor: 0.01` — low floor so the full confidence range is captured for AP
- `method.only_yolo: true` — SAM is not needed for evaluation, skips it entirely

These overrides live in `configs/mask_generation/eval_run.yaml` — no manual edits to `config.yaml` needed.

**Step 2 — Run the evaluation script:**

```bash
python src/mask_generation/evaluation/eval_yolo_boxes.py
```

Override experiment names or any param on the CLI without editing files:
```bash
python src/mask_generation/evaluation/eval_yolo_boxes.py mask_gen_experiment=metrics_v1
python src/mask_generation/evaluation/eval_yolo_boxes.py dataset=phone mask_gen_experiment=my_run
```

## Experiment Names

Configured in `configs/mask_generation/eval_yolo_boxes.yaml`:

```yaml
mask_gen_experiment: "metrics_v1"   # exact folder name of the mask-generation run to read from
eval_experiment: "initial"          # name for this evaluation output — can be different
prepend_date: false                  # prepends today's date to eval_experiment
```

- **`mask_gen_experiment`** — points to the mask-generation run you want to evaluate (that run's folder holds the whole thing — `bboxes/` + `masks/` — the eval reads `bboxes/`). Must exactly match the folder name created in Step 1 (e.g. `metrics_v1` if you ran mask gen with `experiment_name=metrics_v1`). Reads from `results/mask_generation/fip/{plot}/{method.name}/{mask_gen_experiment}/`.
- **`eval_experiment`** — controls where the evaluation results are saved, independently of the mask-gen run. This lets you re-run the evaluation with different settings (e.g. a different `matching_iou_threshold`) on the same boxes and save each run separately without overwriting.

Example: run the evaluation twice with different IoU thresholds on the same boxes:
```bash
python src/mask_generation/evaluation/eval_yolo_boxes.py eval_experiment=iou35 matching_iou_threshold=0.35
python src/mask_generation/evaluation/eval_yolo_boxes.py eval_experiment=iou50 matching_iou_threshold=0.50
```

## Key Design Decisions

- `bboxes/*.pt` — 4-column `[x1, y1, x2, y2]` in absolute pixels, high-confidence only (SAM input, unchanged)
- `bboxes_with_conf/*.pt` — 5-column `[x1, y1, x2, y2, conf]`, all NMS-passing boxes — only created when `only_labeled_images: true`
- GT labels in `manual_label/*.txt` are YOLO format (normalized), converted to absolute pixels at load time
- Matching uses greedy IoU ≥ `matching_iou_threshold` (0.35 default, set in `eval_yolo_boxes.yaml`) — separate from YOLO's NMS `iou_threshold_nms`

## Output

Results are saved to `results/mask_generation/fip/evaluation/{method}/yolo_boxes/{eval_experiment}/`:

| Folder / File | Description |
|---|---|
| `config.yaml` | Parameters used for this run |
| `eval_yolo_boxes.json` | Aggregated and per-plot precision, recall, F1, AP, TP/FP/FN counts, count error ratio |
| `match_viz/` | Image with colored boxes: **blue = TP**, **orange = FP** (false alarm), **red = FN** (missed head) — good for visually diagnosing what YOLO gets wrong |
| `TP_IoU_histograms/` | Histogram of IoU values for TP matches — a peak near 1.0 means tight boxes, spread toward 0.35 means loose |
| `heatmaps_FP/` | Spatial heatmap of FP box centers — shows where false positives cluster (e.g. near poles or field edges) |
| `heatmaps_FN/` | Spatial heatmap of FN box centers — shows where missed heads cluster (e.g. dense or occluded regions) |
| `pr_curves/` | Precision-recall curve across all confidence thresholds — curve pushed to top-right is better, area under it is AP |

## Metrics

| Metric | Meaning |
|--------|---------|
| Precision | of all predicted boxes, fraction that match a GT box |
| Recall | of all GT boxes, fraction that were matched by a prediction |
| F1 | harmonic mean of precision and recall |
| AP | area under the PR curve across all confidence thresholds (COCO 101-point) |
| TP IoU | IoU of matched pred/GT pairs — peak near 1.0 means tight boxes, spread toward threshold means loose |
| Count error ratio | `(pred − GT) / GT` — negative = under-count, positive = over-count; e.g. −0.12 = found 12% fewer heads than annotated. Stored as raw ratio in JSON, displayed as % in terminal |

All metrics are **box-level**, not pixel-level. Note: TN is undefined for object boxes (no concept of "correctly not predicting a box"), so metrics requiring TN (MCC, Balanced Accuracy, FPR) cannot be computed here.

## AP Implementation

COCO-style 101-point interpolated AP. All predictions across all plots pooled globally, sorted by confidence descending. AP reported as `AP@IoU{MATCHING_IOU_THRESHOLD}`.

## Heatmap Notes

Box centers binned into 50×50 grid, Gaussian blur sigma=1.5, cells below 10% of max masked out, `YlOrRd` colormap from 0.15, alpha=0.80, `matplotlib Agg` backend (headless, avoids Qt warnings in WSL).

## Known YOLO Issues (not yet fixed)

1. **Missed heads** — some obvious wheat heads (clear to the human eye) are not found at all
2. **False positives — poles** — parts of the field support poles are classified as wheat heads
3. **False positives — other plants** — simple green leaves / non-wheat plants are flagged as wheat heads with high confidence
4. **Occlusion / splitting / merging issues:**
   - **(a) Box merging** — one predicted box contains multiple true wheat heads
   - **(b) Box splitting** — multiple predicted boxes belong to one true wheat head
   - **(c) Nested boxes** — one box is fully inside another, but IoU doesn't catch it. Idea: post-processing filter using IoS (Intersection over Smaller) — if a smaller box is fully inside a bigger one, remove the smaller box

---

# SAHI vs YOLO Comparison Tools

`eval_yolo_boxes.py` (above) scores **one** method against GT. These three tools compare **SAHI** (`sahi_yolo_sam`) against **plain YOLO** (`yolo_sam_v1`) — head by head — to tune SAHI's tiling knobs (slice size, overlap, merge) by *seeing* what SAHI adds vs breaks, not just aggregate P/R. They import the matching primitives from `eval_yolo_boxes.py` (no changes to it) plus a shared `compare_common.py`.

**Full design + the 7-region model + the metric→knob cheat-sheet:** `docs/archive/SAHI_YOLO_EVAL_PLAN.md` (archived, local-only — plan complete; the tools below are what it produced).

**Findings from running these tools** → [`docs/SAHI_EVAL_RESULTS.md`](../../../docs/SAHI_EVAL_RESULTS.md): the confidence-floor bug that made SAHI first look worse, the single-threshold fix (SAHI now beats YOLO on recall), and the IOS/IOU/CONF merge study (IOS wins on FIP; nested-head case needs mask-based dedup, parked for phone). **SAHI eval boxes are now produced by a normal run with `only_labeled_images=true` — NOT `--config-name eval_run` (SAHI no longer has the `conf_threshold_nms_floor` key).**

## Prerequisite — produce boxes for BOTH methods

The compare tools read each method's saved `bboxes/*.pt`, so run mask generation once per method first (the `eval_run` config sets `only_labeled_images` etc. — see top of this README):
```bash
python src/mask_generation/run_mask_generation.py --config-name eval_run method=yolo_sam_v1   experiment_name=metrics_v2
python src/mask_generation/run_mask_generation.py --config-name eval_run method=sahi_yolo_sam experiment_name=metrics_v2
```
Then point the compare tools at those run folders via `yolo_experiment=` / `sahi_experiment=`.

## 1. `eval_compare_3way.py` — SAHI vs YOLO vs GT (FIP only, needs `manual_label/`)

The main tuning instrument. Sorts every head/box into the 7 regions of the `{GT, YOLO, SAHI}` Venn (recall side: both found / SAHI-rescued / YOLO-only-regression / hard-miss; precision side: shared-FP / YOLO-unique-FP / SAHI-unique-FP).

```bash
python src/mask_generation/evaluation/eval_compare_3way.py \
  yolo_experiment=metrics_v2 sahi_experiment=metrics_v2 \
  overlay_mode=both fp_singles=true eval_experiment=cmp_v2
```

Produces (in `results/mask_generation/{dataset}/evaluation/compare/{eval_experiment}/`):

| Folder / File | Description |
|---|---|
| `compare.json` | per-GT-head **coverage 2×2 per size bucket** (BOTH dataset-relative **tertiles** *and* fixed **COCO** tables), **split/merge** counts vs GT, **FP breakdown** (shared / YOLO-unique / SAHI-unique), per-method **count-error ratio** |
| `overlay_coverage/` | mixed overlay of the 4 recall regions — green=both, orange=YOLO-only (SAHI regression), blue=SAHI-rescued, red=neither |
| `overlay_fp/` | mixed overlay of the 3 FP regions — magenta=shared, cyan=YOLO-unique, yellow=SAHI-unique (usually seam dupes) |
| `regions/<name>/` | single-region images (only if `overlay_mode=singles`/`both`); default set = regions 2,3,4,5, plus 6,7 when `fp_singles=true` |

Read the **tertiles** table for tuning (COCO's "small" bucket is often empty — FIP heads are big); SAHI's win should land in the small bucket, and `count_error_ratio` is the fastest dial for over-counting from un-merged seam duplicates.

## 2. `eval_compare_nogt.py` — YOLO vs SAHI agreement (FIP **and** phone, no GT)

When there's no GT (phone has none; on FIP it's a GT-free cross-check), the 7 regions collapse to 3: **agree** / **YOLO-only** / **SAHI-only**. Can't say who's right — only quantify and visualize divergence.

```bash
python src/mask_generation/evaluation/eval_compare_nogt.py dataset=phone \
  yolo_experiment=metrics_v2 sahi_experiment=metrics_v2 overlay_mode=both
```
(`plot_glob` auto-resolves to the dataset's — `*` FIP, `*/*` phone.)

Produces (in `.../evaluation/compare_nogt/{eval_experiment}/`): `agreement.json` (per-image + total agree / yolo-only / sahi-only + agreement rate), `overlay_agreement/` (green=agree, blue=YOLO-only, magenta=SAHI-only), and `regions/{yolo_only,sahi_only}/` singles. Signal: SAHI-only boxes clustering in dense/small regions = slicing doing its job.

## 3. `sahi_merge_debug.py` — inspect SAHI's merge (lives in `../sahi_yolo_sam/`)

Standalone tool that **re-runs** SAHI's tile inference (YOLO-on-tiles only, no SAM → seconds/image) to expose the merge step (the production pipeline only keeps the final merged boxes). It imports `compute_tile_boxes`/`load_and_slice`/`infer_tiles`/`merge_preds` from `sahi_yolo_pipelined.py` — the pre-merge boxes are simply `infer_tiles`' output before `merge_preds`.

```bash
python src/mask_generation/sahi_yolo_sam/sahi_merge_debug.py plot_glob=plot_461 limit_images=1
```
Reads the SAHI knobs from `method/sahi_yolo_sam.yaml`, so change `sahi_overlap_ratio` / `sahi_match_threshold` (file or CLI), re-run, and compare. Produces (in `.../evaluation/sahi_merge_debug/{eval_experiment}/{plot}/`): `tiles/` (tile grid + raw per-tile boxes), `before_merge/` (duplicates visible), `after_merge/` (final boxes), `clusters/` (each final box's contributing raw boxes share its color — judge over/under-merge by eye), and `merge_counts.json` (**N_raw → N_final**, `collapsed = N_raw − N_final` + the knob values).

## `overlay_mode` (tools 1 & 2)

- `themed` (default) — the mixed overlays (≤4 colors, readable like `match_viz`).
- `singles` — one single-color image per region (zero clutter); `agree`/`both` (the bulk) never get a single.
- `both` — themed + singles.

**Box labels:** every box is labeled with its **confidence** in white text + colored stroke (same style as `eval_yolo_boxes` `match_viz`). Confidences come from each method's `bboxes_with_conf/*.pt` (matched to the good boxes by coordinate). On the **coverage** overlay the boxes are GT boxes (no confidence of their own), so they show the **detecting method's** confidence — YOLO's for `both`/`yolo_only`, SAHI's for `sahi_rescued`, none for `neither`. If a run has no `bboxes_with_conf/` (i.e. it wasn't an `only_labeled_images` run), labels are simply omitted.

Legends on all overlays are semi-transparent so boxes underneath stay visible.
