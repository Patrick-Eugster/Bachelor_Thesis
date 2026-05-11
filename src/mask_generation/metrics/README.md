# YOLO Bounding Box Evaluation — `metrics_yolo_v1.py`

## Overview

Each FIP plot has **one manually labeled image** in `input_plots/fip/{plot}/manual_label/` — a single representative image with ground-truth wheat head bounding boxes in YOLO format (`class_id cx cy w h`, normalized 0–1). The goal is to compare YOLO predictions against these labels to measure detection quality.

## How to Run

**Step 1 — Run detection with the metrics config:**
```bash
python src/mask_generation/yolo_sam_v1/main_v1.py --config-name mask_generation/metrics
```
This loads all settings from `config.yaml` as usual, but automatically overrides the values that must be set for metrics:
- `only_labeled_images: true` — only processes the manually labeled image per plot, and saves `bboxes_with_conf/` needed for the AP curve
- `conf_threshold_nms_floor: 0.01` — low floor so the full confidence range is captured for AP
- `only_yolo: true` — SAM is not needed for metrics, skips it entirely
- `limit_plots: 0` — no limit (overrides any debug value in config.yaml)
- `limit_images: 0` — no limit (overrides any debug value in config.yaml)

These overrides live in `configs/mask_generation/metrics.yaml` — no manual edits to `config.yaml` needed.

**Step 2 — Set experiment names and run the metrics script:**

Open `metrics_yolo_v1.py` and set the constants at the top:
```python
DETECTION_EXPERIMENT = "metrics_v1"  # must match the experiment_name used in Step 1
METRICS_EXPERIMENT   = "initial"     # name for this metrics output — can be different
```
Then run:
```bash
python src/mask_generation/metrics/metrics_yolo_v1.py
```

## Experiment Names

There are **two separate experiment names** at the top of `metrics_yolo_v1.py`:

```python
DATASET_NAME         = "fip"         # "fip" or "phone"
DETECTION_EXPERIMENT = "metrics_v1"  # exact folder name of the detection run to read from
METRICS_EXPERIMENT   = "initial"     # name for this metrics output — can be different
PREPEND_DATE         = False         # prepends today's date to METRICS_EXPERIMENT
```

- **`DETECTION_EXPERIMENT`** — points to the yolo_sam detection run you want to evaluate. Must exactly match the folder name created in Step 1 (e.g. `metrics_v1` if you ran detection with `experiment_name=metrics_v1`). Reads from `results/mask_generation/fip/{plot}/yolo_sam_v1/{DETECTION_EXPERIMENT}/`.
- **`METRICS_EXPERIMENT`** — controls where the metrics results are saved, independently of the detection run. This lets you re-run metrics with different settings (e.g. a different `MATCHING_IOU_THRESHOLD`) on the same detection output and save each run separately without overwriting.

Example: run metrics twice with different IoU thresholds on the same detection output:
```python
DETECTION_EXPERIMENT = "metrics_v1"
METRICS_EXPERIMENT   = "iou35"       # MATCHING_IOU_THRESHOLD = 0.35
# run → results saved to evaluation/.../iou35/

METRICS_EXPERIMENT   = "iou50"       # MATCHING_IOU_THRESHOLD = 0.50
# run → results saved to evaluation/.../iou50/
```

## Key Design Decisions

- `bboxes/*.pt` — 4-column `[x1, y1, x2, y2]` in absolute pixels, high-confidence only (SAM input, unchanged)
- `bboxes_with_conf/*.pt` — 5-column `[x1, y1, x2, y2, conf]`, all NMS-passing boxes — only created when `only_labeled_images: true`
- GT labels in `manual_label/*.txt` are YOLO format (normalized), converted to absolute pixels at load time
- Matching uses greedy IoU ≥ `MATCHING_IOU_THRESHOLD` (0.35 default) — separate from YOLO's NMS `IOU_THRESHOLD_NMS`

## Output

Results are saved to `results/mask_generation/fip/evaluation/yolo_sam_v1/metrics_yolo_v1/{METRICS_EXPERIMENT}/`:

| Folder / File | Description |
|---|---|
| `config.yaml` | Parameters used for this run |
| `metrics_yolo_v1.json` | Aggregated and per-plot precision, recall, F1, AP, TP/FP/FN counts |
| `match_viz/` | Image with colored boxes: **blue = TP**, **orange = FP** (false alarm), **red = FN** (missed head) — good for visually diagnosing what YOLO gets wrong |
| `TP_IoU_histograms/` | Histogram of IoU values for TP matches — a peak near 1.0 means tight boxes, spread toward 0.35 means loose |
| `heatmaps_FP/` | Spatial heatmap of FP box centers — shows where false positives cluster (e.g. near poles or field edges) |
| `heatmaps_FN/` | Spatial heatmap of FN box centers — shows where missed detections cluster (e.g. dense or occluded regions) |
| `pr_curves/` | Precision-recall curve across all confidence thresholds — curve pushed to top-right is better, area under it is AP |

## AP Implementation

COCO-style 101-point interpolated AP. All predictions across all plots pooled globally, sorted by confidence descending. AP reported as `AP@IoU{MATCHING_IOU_THRESHOLD}`.

## Heatmap Notes

Box centers binned into 50×50 grid, Gaussian blur sigma=1.5, cells below 10% of max masked out, `YlOrRd` colormap from 0.15, alpha=0.80, `matplotlib Agg` backend (headless, avoids Qt warnings in WSL).

## Known YOLO Detection Issues (not yet fixed)

1. **Missed detections** — some obvious wheat heads (clear to the human eye) are not detected at all
2. **False positives — poles** — parts of the field support poles are classified as wheat heads
3. **False positives — other plants** — simple green leaves / non-wheat plants are detected as wheat heads with high confidence
4. **Occlusion / splitting / merging issues:**
   - **(a) Box merging** — one predicted box contains multiple true wheat heads
   - **(b) Box splitting** — multiple predicted boxes belong to one true wheat head
   - **(c) Nested boxes** — one box is fully inside another, but IoU doesn't catch it. Idea: post-processing filter using IoS (Intersection over Smaller) — if a smaller box is fully inside a bigger one, remove the smaller box
