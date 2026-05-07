# YOLO Bounding Box Evaluation — `metrics_yolo_v1.py`

## Overview

Each FIP plot has **one manually labeled image** in `input_plots/fip/{plot}/manual_label/` — a single representative image with ground-truth wheat head bounding boxes in YOLO format (`class_id cx cy w h`, normalized 0–1). The goal is to compare YOLO predictions against these labels to measure detection quality.

## How to Run

**Step 1 — Run detection with the metrics config:**
```bash
python src/mask_generation/yolo_sam_v1/main_v1.py --config-name mask_generation/metrics
```
This loads all settings from `config.yaml` as usual, but automatically overrides the four values that must be set for metrics:
- `only_labeled_images: true` — only processes the manually labeled image per plot, and saves `bboxes_with_conf/` needed for the AP curve
- `conf_threshold_nms_floor: 0.01` — low floor so the full confidence range is captured for AP
- `limit_plots: 0` — no limit (overrides any debug value in config.yaml)
- `limit_images: 0` — no limit (overrides any debug value in config.yaml)

These overrides live in `configs/mask_generation/metrics.yaml` — no manual edits to `config.yaml` needed.

**Step 2 — Run the metrics script:**
```bash
python src/mask_generation/metrics/metrics_yolo_v1.py
```

To change dataset or experiment, edit the constants at the top of the script:
```python
DATASET_NAME    = "fip"      # switch to "phone" for phone data
EXPERIMENT_NAME = "initial"  # or a custom name / "" for timestamp
APPEND_DATE     = False
```

## Key Design Decisions

- `bboxes/*.pt` — 4-column `[x1, y1, x2, y2]` in absolute pixels, high-confidence only (SAM input, unchanged)
- `bboxes_with_conf/*.pt` — 5-column `[x1, y1, x2, y2, conf]`, all NMS-passing boxes — only created when `only_labeled_images: true`
- GT labels in `manual_label/*.txt` are YOLO format (normalized), converted to absolute pixels at load time
- Matching uses greedy IoU ≥ `MATCHING_IOU_THRESHOLD` (0.35 default) — separate from YOLO's NMS `IOU_THRESHOLD_NMS`

## Output

Results are saved to `results/mask_generation/fip/evaluation/yolo_sam_v1/metrics_yolo_v1/{experiment}/`:

| Folder | Content |
|--------|---------|
| `config.yaml` | Copy of the config used for this run |
| `metrics_yolo_v1.json` | AP, precision, recall, TP/FP/FN counts |
| `match_viz/` | Original image with colored boxes: blue=TP, yellow-orange=FP, red=FN |
| `TP_IoU_histograms/` | Histogram of TP/FP/FN IoU values per image + aggregated |
| `heatmaps_FP/` | FP density heatmap overlaid on image |
| `heatmaps_FN/` | FN density heatmap overlaid on image |
| `pr_curves/` | PR curve: precision vs recall across all confidence thresholds |

## AP Implementation

COCO-style 101-point interpolated AP. All predictions pooled globally, sorted by confidence descending. AP reported as `AP@IoU{MATCHING_IOU_THRESHOLD}`.

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
