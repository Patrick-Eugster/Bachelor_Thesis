#!/bin/bash -l
#SBATCH -J seg_A0715_pin15k_fullcull
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G       # 64 GB total — phone 3D seg decodes many full-res masks (RAM-heavy)
#SBATCH --cpus-per-task=8
#SBATCH --time=09:59:00        # masks reused (skip ~30min) + seg ~4-5h (fast-paint) + eval; 10h ceiling for safety
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_pin15k_fullcull_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_pin15k_fullcull_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# A/0715 seg — PINHOLE 15k with the FULL best-cull config (tilt fix), to make the
# pinhole-vs-opencv comparison clean. The existing pinhole seg (pin15k_yolov5_pertile, IoU 0.265) used
# frustum-only cull and PREDATES the tilt fix; the best opencv run (ocv15k_frust_paint, IoU 0.369) uses
# roi+marker+frustum+ground-cull-fix. This run puts pinhole on the SAME config with the SAME mask family,
# so the only difference vs ocv15k_frust_paint is the SfM/undistortion.
#
# Compare pin15k_fullcull against:
#   (1) ocv15k_frust_paint (IoU 0.369)      -> clean pinhole-vs-opencv at best config (SfM effect)
#   (2) pin15k_yolov5_pertile (IoU 0.265)   -> tilt-fix effect on pinhole (same masks, config-only)
#
# Reuses the existing pinhole 3DGS model (NO retrain) and the existing yolov5_pertile_pin15k masks
# (SKIP mask-gen if present, else regenerate identically). NO render_360. eval_2d ON (pinhole GT native).
# ============================================================================

FIELD=field_A
DATE=20250715
MODEL_EXP=baseline                       # existing pinhole 3DGS model (results/.../vanilla_3dgs/<MODEL_EXP>)
ITER=15000
TARGET=4032                              # pinhole is native 4032 -> letterbox r=1.0 (no scaling)
MASK_EXP=yolov5_pertile_pin15k           # REUSE the same SAM1 per_tile masks the pin15k run used
SEG_EXP=pin15k_fullcull                  # NEW seg output name (never collides with pin15k_yolov5_pertile)

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"

MODEL="$REPO/results/reconstruction/phone/$FIELD/$DATE/vanilla_3dgs/$MODEL_EXP"
MASK_OUT="$REPO/results/mask_generation/phone/$FIELD/$DATE/yolo_sam_v1/$MASK_EXP"
SEG_OUT="$MODEL/segmentation_3d/$SEG_EXP"
IN_IMG="$REPO/input_plots/phone/$FIELD/$DATE/images"
IN_MARK="$REPO/input_plots/phone/$FIELD/$DATE/logs/marker_points3d.json"

echo "=== PREFLIGHT ==="
FAIL=0
[ -f "$MODEL/point_cloud/iteration_$ITER/point_cloud.ply" ] || { echo "  X no model at iter $ITER: $MODEL"; FAIL=1; }
[ -d "$IN_IMG" ]  || { echo "  X no images: $IN_IMG"; FAIL=1; }
[ -f "$IN_MARK" ] || { echo "  X no marker_points3d.json (roi_cull + marker_exclude NEED it): $IN_MARK"; FAIL=1; }
[ -d "$SEG_OUT" ] && [ -n "$(ls -A "$SEG_OUT" 2>/dev/null)" ] && { echo "  X seg target exists (no overwrite): $SEG_OUT"; FAIL=1; }
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: model(iter $ITER) + images + markers present, seg target fresh"

module purge
source ~/miniconda3/etc/profile.d/conda.sh

# ---------------- STAGE 1: MASK GENERATION — SKIP if the masks already exist ----------------
if [ -n "$(ls -A "$MASK_OUT/masks" 2>/dev/null)" ]; then
  echo ""; echo "======== STAGE 1 — SKIPPED (masks already exist: $MASK_OUT/masks) ========"
  MASK_RC=0
else
  echo ""; echo "======== STAGE 1 — full-res YOLOv5 + per_tile SAM1 + ROI (target $TARGET) ========"
  conda activate wheat-maskgen
  module load eth_proxy 2>/dev/null || true
  python src/mask_generation/run_mask_generation.py \
    dataset=phone method=yolo_sam_v1 \
    dataset.plot_glob=$FIELD/$DATE \
    method.target_image_size=$TARGET method.batch_size_yolo=2 \
    method.sam_crop_mode=per_tile method.sam_backend=sam1 \
    roi.enabled=true \
    experiment_name=$MASK_EXP
  MASK_RC=$?
  conda deactivate
  echo "  mask-gen rc=$MASK_RC -> $MASK_OUT"
  if [ $MASK_RC -ne 0 ] || [ -z "$(ls -A "$MASK_OUT/masks" 2>/dev/null)" ]; then
    echo "ABORTING — mask generation failed or produced no masks."; exit 1
  fi
fi

# ---------------- STAGE 2: 3D SEG (full cull) + eval + eval_2d (wheat3dgs) ----------------
echo ""; echo "======== STAGE 2 — 3D seg ONLY (pinhole $MODEL_EXP, iter $ITER) FULL cull ========"
module purge
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!; trap "kill $VRAM_PID 2>/dev/null" EXIT
export WHEAT_RUN_REPORT="$REPO/slurm_logs/run_report_${SLURM_JOB_ID}.txt"

# pinhole -> no sfm_variant. SEG ONLY (no run_eval/run_eval_2d): step 6 eval writes into the SHARED
# model-level test/overlay + test/segmentation folders and would OVERWRITE the existing pin15k_yolov5_pertile
# eval renders there. So we produce only the per-exp 2DSeg here and CPU-score locally afterwards (same scorer
# as the rest of the phone table, against the native pinhole GT). NO retrain, NO render_360.
# FULL cull config mirrors ocv15k_frust_paint: frustum + roi + marker (defaults marker_radius_m=0.075,
# roi_buffer_m=0.25), ground-cull tilt-fix by default (markers present), fast-paint default.
python src/run_reconstruction.py \
  dataset=phone plot=$FIELD date=$DATE \
  experiment_name=$MODEL_EXP \
  reconstruction.iterations=$ITER reconstruction.resolution=1 \
  reconstruction.use_principal_point=false \
  run_seg=true \
  segmentation_3d.frustum_cull=true \
  segmentation_3d.roi_cull=true \
  segmentation_3d.marker_exclude=true \
  segmentation_3d.detection_method=yolo_sam_v1 \
  segmentation_3d.mask_gen_experiment=$MASK_EXP \
  segmentation_3d.exp_name=$SEG_EXP
SEG_RC=$?

echo ""; echo "======== STATUS ========"
echo "  mask-gen: rc=$MASK_RC -> $MASK_OUT"
echo "  seg-only: rc=$SEG_RC -> $SEG_OUT  (CPU-score locally against pinhole GT after pulling 2DSeg)"
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
