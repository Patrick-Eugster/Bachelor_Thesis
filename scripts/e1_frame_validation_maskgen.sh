#!/bin/bash
# ============================================================================
# E1 — mask-gen FRAME validation: pinhole vs opencv (LOCAL, GPU).
#
# Question it answers: the phone mask-gen grid was evaluated on the PINHOLE GT frame, but recon/seg run
# on the OPENCV frame. Does the mask-gen result actually depend on the frame? We run the SAME config on
# both and compare — small delta => the pinhole eval is valid for the opencv pipeline (answers the
# examiner's "why pinhole not opencv" with a number). Big delta => the chapter must move to opencv.
#
# Config (fixed, matches the conf-sweep): YOLOv5 detector @ full-res 4032, SAM per_tile, SAM2, conf 0.35,
# only the 6 GT images. per_tile is light, so this runs locally.
#
# SAFETY: fresh experiment names (e1_frame_*) that must NOT already exist; the warp writes only to
# opencv/manual_label/<stem>_sets/ (never the source) and never overwrites (overwrite=false). All outputs
# are saved locally: full-res per-head mask PNGs under results/ (recreatable), warped instance GT under
# opencv/manual_label/, eval JSONs + the compare table. The script ABORTS if any target already exists.
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

PIN_EXP=e1_frame_pinhole
OCV_EXP=e1_frame_opencv
SESSIONS=("field_A 20250627" "field_A 20250706" "field_A 20250715"
          "field_D 20250627" "field_D 20250706" "field_D 20250715")

MG_ARGS="dataset=phone method=yolo_sam_v1 method.target_image_size=4032 method.sam_crop_mode=per_tile \
method.sam_backend=sam2 method.conf_threshold_good_box=0.35 only_labeled_images=true"

echo "=== E1 PREFLIGHT (nothing is overwritten) ==="
FAIL=0
# SAM2 + YOLO weights present
[ -f "src/mask_generation/weights/sam2.1_l.pt" ] || { echo "  X missing SAM2 weight sam2.1_l.pt"; FAIL=1; }
# the 6 pinhole instance-GT sets exist
for s in "${SESSIONS[@]}"; do set -- $s; F=$1; D=$2
  ls -d input_plots/phone/$F/$D/manual_label/*_sets >/dev/null 2>&1 \
    || { echo "  X no pinhole GT _sets in $F/$D"; FAIL=1; }
done
# fresh mask-gen exp dirs must NOT exist (no overwrite / no mix)
for s in "${SESSIONS[@]}"; do set -- $s; F=$1; D=$2
  [ -e "results/mask_generation/phone/$F/$D/yolo_sam_v1/$PIN_EXP" ] && { echo "  X exists: $F/$D .../$PIN_EXP"; FAIL=1; }
  [ -e "results/mask_generation/phone/$F/$D/opencv/yolo_sam_v1/$OCV_EXP" ] && { echo "  X exists: $F/$D opencv/.../$OCV_EXP"; FAIL=1; }
done
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: weights + 6 pinhole GT present, target exp names are fresh"

echo ""; echo "=== STEP 1: warp the INSTANCE GT into the opencv frame (6 sessions) ==="
# writes opencv/manual_label/<stem>_sets/ + manifest; overwrite=false so existing files are never clobbered
for s in "${SESSIONS[@]}"; do set -- $s; F=$1; D=$2
  echo "  -- $F/$D --"
  python src/preprocessing/warp_gt_to_variant.py field=$F date=$D variants='[opencv]' warp_instances=true
done

echo ""; echo "=== STEP 2: mask-gen on PINHOLE (6 GT imgs) -> $PIN_EXP ==="
python src/mask_generation/run_mask_generation.py $MG_ARGS experiment_name=$PIN_EXP

echo ""; echo "=== STEP 3: mask-gen on OPENCV (6 GT imgs) -> $OCV_EXP ==="
python src/mask_generation/run_mask_generation.py $MG_ARGS dataset.plot_glob='*/*/opencv' experiment_name=$OCV_EXP

echo ""; echo "=== STEP 4: score PINHOLE (Hungarian, IoU 0.5) ==="
python src/mask_generation/evaluation/eval_masks_instance.py dataset=phone method=yolo_sam_v1 \
  mask_gen_experiment=$PIN_EXP eval_experiment=$PIN_EXP

echo ""; echo "=== STEP 5: score OPENCV (Hungarian, IoU 0.5, warped GT) ==="
python src/mask_generation/evaluation/eval_masks_instance.py dataset=phone method=yolo_sam_v1 \
  dataset.plot_glob='*/*/opencv' mask_gen_experiment=$OCV_EXP eval_experiment=$OCV_EXP

echo ""; echo "=== STEP 6: compare pinhole vs opencv ==="
PIN_JSON="results/mask_generation/phone/evaluation/yolo_sam_v1/masks_instance/$PIN_EXP/eval_masks_instance.json"
OCV_JSON="results/mask_generation/phone/evaluation/yolo_sam_v1/masks_instance/$OCV_EXP/eval_masks_instance.json"
python src/analysis/compare_frame_e1.py --pinhole "$PIN_JSON" --opencv "$OCV_JSON" \
  --out docs/analysis_results/e1_frame/E1_frame_validation.md

echo ""; echo "=== E1 DONE — table at docs/analysis_results/e1_frame/E1_frame_validation.md ==="
