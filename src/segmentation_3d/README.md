# 3D Segmentation — `src/segmentation_3d/`

## Overview

Assigns consistent 3D wheat head IDs to Gaussians across all views. Takes the trained 3DGS model (step 1) and the 2D masks from YOLO+SAM (step 2) as input.

Run via `src/run_reconstruction.py` with `run_seg=true` — not directly. The segmentation script (`run_3d_seg.py`) is orchestrated by the pipeline.

---

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
| `eval_wheatgs.py` | Evaluates 3D segmentation quality by comparing projected 3D labels against SAM 2D masks |

---

## Configuration

Controlled via `configs/reconstruction_seg3d/segmentation_3d/default.yaml`:

```yaml
exp_name: "run_1"           # subfolder name inside segmentation_3d/ — change to re-run without overwriting
detection_experiment: "initial"  # which yolo_sam run to read masks from
save_vis_overlay: true      # save colored overlay images per head (good for debugging)
vis_max_heads: 10           # only save overlays for first N heads (0 = all — can be slow for 300+ heads)
wandb_enabled: false        # log per-head progress to wandb.ai
```

**`exp_name`** is the key parameter — it controls the output subfolder so you can re-run segmentation with different settings on the same trained model without retraining:

```bash
python src/run_reconstruction.py run_seg=true exp_name=run_2
python src/run_reconstruction.py run_seg=true exp_name=iou05 --iou_threshold 0.5
```

---

## Output Structure

```
results/reconstruction/fip/{plot}/vanilla_3dgs/{experiment}/segmentation_3d/{exp_name}/
├── gaussians.ply            ← all segmented Gaussians (fine-tuned, with obj labels)
├── gaussians_colored.ply    ← same but with per-head HSV colors baked in (for viewer)
├── all_obj_labels.pth       ← per-Gaussian wheat head ID tensor
├── all_counts.pth           ← FlashSplat contribution counts
├── results.csv              ← per-head: ID, source mask, view count, Gaussian count
├── experiment.txt           ← run metadata
├── 2DSeg/                   ← per-camera 2D label maps (.pt, one per image) — used by eval
├── ply/                     ← per-head PLY files: wh_0001.ply, wh_0042_b.ply (overlap suffix)
└── img/                     ← overlay visualizations per head (only if save_vis_overlay=true)
```

The `2DSeg/` folder is the key output used by `eval_wheatgs.py` and the viewer — it contains the final 2D projection of all 3D assignments per camera.

---

## Eval (`eval_wheatgs.py`)

Compares the 3D segmentation back against the original SAM 2D masks to measure how well the 3D assignment reproduced the 2D input:

```bash
python src/run_reconstruction.py run_eval=true exp_name=run_1
```

Outputs overlay PNGs per camera into `test/segmentation/` showing predicted vs GT mask alignment.

---

## Logging

Segmentation output is logged to `seg_logs/{exp_name}.txt` inside the reconstruction experiment folder. Controlled by `log_seg_only: true` in `configs/reconstruction_seg3d/segmentation_3d/default.yaml` — when true, only the segmentation step is logged to file (training stays terminal-only).
