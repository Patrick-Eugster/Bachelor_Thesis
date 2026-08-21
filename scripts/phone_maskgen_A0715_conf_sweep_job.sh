#!/bin/bash -l
#SBATCH -J maskgen_A0715_confsweep
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G       # 64 GB total — per_tile SAM emits many full-res masks (RAM-heavy; save-queue backpressure caps peak ~10-12 GB)
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00         # SAM2 per_tile ~5-7 min/session x 3 confs + YOLO 35s each; generous 1.5h ceiling
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/maskgen_A0715_confsweep_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/maskgen_A0715_confsweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# STEP 1 of the 3D-seg CONF TEST — generate the 3 mask sets (one per YOLO confidence) for A/0715 opencv,
# so the 3 seg jobs each read a clean, conf-specific mask set. CLEAN RERUN (not a filter) so we're on the
# full-res per_tile SAM2 mask set we actually ship, not a possibly-crippled old set.
#
#   detector : full-res YOLOv5 (target 3904 = no downscale on the ~3890px opencv frame) + per_tile + SAM2
#              (sam2.1_l.pt — best-scored granularity/backend, faster than SAM1) + marker ROI.
#   confs    : 0.40 / 0.55 / 0.70  (the precision-first triplet; picked from downstream 3D-seg, not the 2D curve)
#   output   : results/mask_generation/phone/field_A/20250715/opencv/yolo_sam_v1/pertile_sam2_conf0{40,55,70}
#
# Cheap (~7 min/conf) so we run all 3 here, then EYEBALL the box/head counts (should DROP as conf rises)
# BEFORE spending seg hours. Fresh experiment names -> ABORTS if any already exists (no overwrite/mix).
# ============================================================================
set -euo pipefail
FIELD=field_A
DATE=20250715
VARIANT=opencv
TARGET=3904                          # full-res on the ~3890px opencv frame (stride-32 safe)
CONFS=(0.40 0.55 0.70)
declare -A EXP=( [0.40]=pertile_sam2_conf040 [0.55]=pertile_sam2_conf055 [0.70]=pertile_sam2_conf070 )

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"
IN_IMG="$REPO/input_plots/phone/$FIELD/$DATE/$VARIANT/images"
IN_MARK="$REPO/input_plots/phone/$FIELD/$DATE/$VARIANT/logs/marker_points3d.json"
MASK_BASE="$REPO/results/mask_generation/phone/$FIELD/$DATE/$VARIANT/yolo_sam_v1"
SAM_W="$REPO/src/mask_generation/weights/sam2.1_l.pt"

echo "=== PREFLIGHT ==="
FAIL=0
[ -d "$IN_IMG" ]  || { echo "  X no opencv images: $IN_IMG"; FAIL=1; }
[ -f "$IN_MARK" ] || { echo "  X no marker_points3d.json for ROI: $IN_MARK"; FAIL=1; }
[ -f "$SAM_W" ]   || { echo "  X missing SAM2 large weights: $SAM_W"; FAIL=1; }
for c in "${CONFS[@]}"; do
  [ -e "$MASK_BASE/${EXP[$c]}" ] && { echo "  X exists: ${EXP[$c]}"; FAIL=1; }
done
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: images + markers + SAM2 weights present, all 3 conf exp names fresh"

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat-maskgen
module load eth_proxy 2>/dev/null || true
nvidia-smi

for c in "${CONFS[@]}"; do
  echo ""; echo "================ mask-gen conf=$c  ->  ${EXP[$c]} ================"
  python src/mask_generation/run_mask_generation.py \
    dataset=phone method=yolo_sam_v1 \
    dataset.plot_glob=$FIELD/$DATE/$VARIANT \
    method.target_image_size=$TARGET method.batch_size_yolo=2 \
    method.sam_crop_mode=per_tile method.sam_backend=sam2 \
    method.conf_threshold_good_box=$c \
    roi.enabled=true \
    experiment_name=${EXP[$c]}
  OUT="$MASK_BASE/${EXP[$c]}"
  [ -n "$(ls -A "$OUT/masks" 2>/dev/null)" ] || { echo "ABORTING — conf=$c produced no masks"; exit 1; }
  echo "  -> $OUT"
done

echo ""; echo "================ SUMMARY — mask/instance count per conf (should DROP as conf rises) ================"
# mask files = one per detected head INSTANCE across the session -> this is what shrinks with conf.
# bbox files = one .pt per IMAGE (stays ~constant = #images), so report it only as a sanity count.
# { ls || true; } | wc -l  keeps a missing subdir from tripping pipefail.
for c in "${CONFS[@]}"; do
  OUT="$MASK_BASE/${EXP[$c]}"
  NM=$( { ls "$OUT/masks"  2>/dev/null || true; } | wc -l )
  NB=$( { ls "$OUT/bboxes" 2>/dev/null || true; } | wc -l )
  echo "  conf=$c  ${EXP[$c]}:  masks(instances)=$NM   bbox files(images)=$NB"
done
echo "DONE. Check masks(instances) drops as conf rises, then submit the 3 per-conf seg jobs."
