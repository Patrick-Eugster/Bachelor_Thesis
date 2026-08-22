#!/bin/bash
# ============================================================================
# E2 (SAM1 parity) — full-res YOLOv5 into the phone mask-gen grid, SAM1 (LOCAL, GPU).
#
# The SAM2 full-res block (e2_*_sam2) is already on disk; this fills the SAM1 column so the full-res
# block matches the existing grid's SAM1/SAM2 structure. SAM1 (ViT-H) is also the mask backend the phone
# 3D-seg runs actually use (full-res YOLOv5 + per_tile + SAM1), so the SAM1 numbers are the directly
# pipeline-relevant ones. Same detector/res/conf as the SAM2 run; only sam_backend changes.
#
# Cells: YOLOv5 @4032 x {full_frame, per_tile, per_head} x SAM1. Fresh names e2_*_sam1; ABORTS if any
# already exists (no overwrite/mix). Masks saved under results/ (recreatable). Scored on the 6 pinhole GT
# images with the same grid scorer (eval_masks_instance, Hungarian @0.5, conf 0.35, micro).
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"

# batch_size_yolo=2: at 4032 px the config default (25) batches many full-res frames at once and blew up
# system RAM; 2 is the proven-safe cap the phone full-res scripts use. sam1_decode_batch stays 1 (VRAM-safe).
MG="dataset=phone method=yolo_sam_v1 method.target_image_size=4032 method.sam_backend=sam1 \
method.batch_size_yolo=2 method.conf_threshold_good_box=0.35 only_labeled_images=true"
declare -A GRAN=( [ff]=full_frame [pt]=per_tile [ph]=per_head )

echo "=== E2-SAM1 PREFLIGHT ==="
FAIL=0
[ -f "src/mask_generation/weights/sam_vit_h.pth" ] || [ -f "src/mask_generation/weights/sam_vit_h_4b8939.pth" ] || echo "  (note: verify SAM1 ViT-H weight path if mask-gen fails)"
for g in ff pt ph; do
  for s in "field_A 20250627" "field_A 20250706" "field_A 20250715" "field_D 20250627" "field_D 20250706" "field_D 20250715"; do
    set -- $s
    [ -e "results/mask_generation/phone/$1/$2/yolo_sam_v1/e2_${g}_sam1" ] && { echo "  X exists: $1/$2 e2_${g}_sam1"; FAIL=1; }
  done
done
[ "$FAIL" -ne 0 ] && { echo "ABORTING."; exit 1; }
echo "  ok: e2_*_sam1 names fresh"

for g in ff pt ph; do
  echo ""; echo "=== mask-gen YOLOv5 @4032, ${GRAN[$g]}, SAM1 -> e2_${g}_sam1 ==="
  python src/mask_generation/run_mask_generation.py $MG \
    method.sam_crop_mode=${GRAN[$g]} experiment_name=e2_${g}_sam1
  echo "=== score e2_${g}_sam1 ==="
  python src/mask_generation/evaluation/eval_masks_instance.py dataset=phone method=yolo_sam_v1 \
    mask_gen_experiment=e2_${g}_sam1 eval_experiment=e2_${g}_sam1
done

echo ""; echo "=== assemble: full-res YOLOv5 SAM1 block next to the existing grid ==="
python src/analysis/assemble_e2_grid.py --sam sam1 \
  --out docs/analysis_results/e2_fullres_grid/E2_fullres_yolov5_sam1.md

echo ""; echo "=== E2-SAM1 DONE -> docs/analysis_results/e2_fullres_grid/E2_fullres_yolov5_sam1.md ==="
