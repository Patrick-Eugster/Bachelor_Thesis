#!/bin/bash
# ============================================================================
# E2 — full-res YOLOv5 into the phone mask-gen grid (LOCAL, GPU).
#
# Question: the grid ran plain YOLOv5 at 1280px while YOLO11 got 3008 and SAHI tiles at ~native res, so
# the detector comparison is confounded by resolution. Here we run YOLOv5 at FULL res (4032) through the
# SAME grid scorer (eval_masks_instance, Hungarian, conf 0.35) at all 3 granularities, SAM2, so it slots
# into the existing table as a new "YOLOv5 @4032" block next to "YOLOv5 @1280". If it reaches SAHI/YOLO11,
# SAHI's tiling is redundant.
#
# Cells: YOLOv5 @4032 x {full_frame, per_tile, per_head} x SAM2  (SAM3 skipped; SAM1 later if wanted).
# All local (SAM2 fits 16 GB). Fresh names e2_*_sam2; ABORTS if any already exists (no overwrite/mix).
# Masks saved under results/ (recreatable). Scored on the 6 pinhole GT images.
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"

MG="dataset=phone method=yolo_sam_v1 method.target_image_size=4032 method.sam_backend=sam2 \
method.conf_threshold_good_box=0.35 only_labeled_images=true"
declare -A GRAN=( [ff]=full_frame [pt]=per_tile [ph]=per_head )

echo "=== E2 PREFLIGHT ==="
FAIL=0
[ -f "src/mask_generation/weights/sam2.1_l.pt" ] || { echo "  X missing sam2.1_l.pt"; FAIL=1; }
for g in ff pt ph; do
  for s in "field_A 20250627" "field_A 20250706" "field_A 20250715" "field_D 20250627" "field_D 20250706" "field_D 20250715"; do
    set -- $s
    [ -e "results/mask_generation/phone/$1/$2/yolo_sam_v1/e2_${g}_sam2" ] && { echo "  X exists: $1/$2 e2_${g}_sam2"; FAIL=1; }
  done
done
[ "$FAIL" -ne 0 ] && { echo "ABORTING."; exit 1; }
echo "  ok: weights present, e2_* names fresh"

for g in ff pt ph; do
  echo ""; echo "=== mask-gen YOLOv5 @4032, ${GRAN[$g]}, SAM2 -> e2_${g}_sam2 ==="
  python src/mask_generation/run_mask_generation.py $MG \
    method.sam_crop_mode=${GRAN[$g]} experiment_name=e2_${g}_sam2
  echo "=== score e2_${g}_sam2 ==="
  python src/mask_generation/evaluation/eval_masks_instance.py dataset=phone method=yolo_sam_v1 \
    mask_gen_experiment=e2_${g}_sam2 eval_experiment=e2_${g}_sam2
done

echo ""; echo "=== assemble: full-res YOLOv5 next to the existing grid (SAM2) ==="
python src/analysis/assemble_e2_grid.py \
  --out docs/analysis_results/e2_fullres_grid/E2_fullres_yolov5.md

echo ""; echo "=== E2 DONE -> docs/analysis_results/e2_fullres_grid/E2_fullres_yolov5.md ==="
