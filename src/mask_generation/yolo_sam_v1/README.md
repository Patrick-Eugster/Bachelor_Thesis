# YOLO + SAM Pipeline — `yolo_sam_v1`

## Overview

Two-stage pipeline: YOLO detects wheat head bounding boxes per image, SAM generates binary instance masks from those boxes.

Entry point: `main_v1.py` — configured via `configs/mask_generation/config.yaml`.

---

## How to Run

Run from the workspace root:

```bash
python src/mask_generation/yolo_sam_v1/main_v1.py
```

Configure via `configs/mask_generation/config.yaml` — set `plot`, `experiment_name`, thresholds, etc. Any parameter can also be overridden on the CLI without editing the file:

```bash
python src/mask_generation/yolo_sam_v1/main_v1.py dataset=phone plot=phone01
python src/mask_generation/yolo_sam_v1/main_v1.py experiment_name=my_run conf_threshold_detection=0.4
```

For metrics evaluation, use the metrics config (automatically sets `only_labeled_images=true` and `conf_threshold_nms_floor=0.01`):

```bash
python src/mask_generation/yolo_sam_v1/main_v1.py --config-name mask_generation/metrics
```

---

## Pipelined YOLO Inference (`yolo_v1_pipelined.py`)

Pipelines 3 phases (CPU resize / GPU infer / CPU save) so the GPU never idles:

```
              t=1      t=2      t=3      t=4      t=5      t=6      t=7
SEQUENTIAL
  CPU resize  [batch1]                   [batch2]
  GPU infer            [batch1]                   [batch2]
  CPU save                      [batch1]                   [batch2]

PIPELINED
  CPU resize  [batch1] [batch2] [batch3]
  GPU infer            [batch1] [batch2] [batch3]
  CPU save                      [batch1] [batch2] [batch3]
```
Sequential: each phase waits for the previous to finish — 2 batches take 6 time units, 1 worker active at a time.
Pipelined: all 3 phases overlap — 3 batches take 5 time units, GPU never idles.

**Concurrency structure:**
- Outer `ThreadPoolExecutor(max_workers=2)` — one slot for the active resize future, one for the active save future
- Each of those tasks spawns its own inner pool with `MAX_THREADS // 2` threads for per-image parallelism
- `MAX_THREADS // 2` (not full `MAX_THREADS`) because resize and save run simultaneously — using full `MAX_THREADS` each over-subscribes the CPU and starves YOLO's NMS step on the main thread
- `torch.cuda.synchronize()` blocks the main thread during GPU inference but background threads are unaffected — they keep running independently

**NMS time limit:** YOLOv5's NMS has a hardcoded time limit (`yolov5/utils/general.py`). In the pipelined version, background CPU threads compete with NMS, making it slower. The limit was raised from `0.5 + 0.05 * bs` to `1.5 + 0.15 * bs` (3×) to prevent false timeout warnings. This does not affect detection quality — NMS still exits immediately when done.

---

## Pipelined SAM Inference (`sam_v1_pipelined.py`)

Pipelines load / GPU encode / save so the GPU never idles:

```
              t=1      t=2      t=3      t=4      t=5      t=6      t=7
SEQUENTIAL
  CPU load    [img 1]                    [img 2]
  GPU encode           [img 1]                    [img 2]
  CPU save                      [img 1]                    [img 2]

PIPELINED
  CPU load    [img 1]  [img 2]  [img 3]
  GPU encode           [img 1]  [img 2]  [img 3]
  CPU save                      [img 1]  [img 2]  [img 3]
```
Sequential: each phase waits for the previous to finish — 2 images take 6 time units, 1 worker active at a time.
Pipelined: GPU encodes image N while CPU loads N+1 and saves N-1 — 3 images take 5 time units.

**Concurrency structure:**
- Outer `ThreadPoolExecutor(max_workers=2)` — one slot for the load future, one for the save future
- The save task spawns its own inner pool (`max_threads` threads) for parallel mask PNG writing
- Main thread runs GPU inference (`set_image` + `predict_torch`) — blocks on `torch.cuda.synchronize()` but background threads keep running independently

**`t_save` in pipelined version:** saving runs async so it is not printed per-image. It is still collected from futures at the end of each plot and added to `total_sam_pure_time`, so all totals/averages in the final report remain correct.

---

## wandb Integration

Both YOLO and SAM phases support wandb logging, controlled by `wandb_enabled` in `configs/mask_generation/config.yaml`.

- SAM logs per-image: `t_embed_s`, `t_pred_s`, `n_heads`, `plot`
- Automatically logs GPU%, VRAM, RAM, CPU% every few seconds (built into wandb, no extra code)
- Project name: `wheat3dgs-sam-v1` — created automatically on first run at wandb.ai
- One-time setup: run `wandb login` in terminal and paste API key from wandb.ai
