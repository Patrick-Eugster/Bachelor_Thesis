#!/bin/bash -l
#SBATCH -J seg_A0715_ocv15k_conf035_sam2
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G       # 64 GB total — per_tile SAM emits many full-res masks (RAM-heavy)
#SBATCH --cpus-per-task=8
#SBATCH --time=07:59:00        # mask-gen SAM2 ~7 min + opencv baseline (1.23 M) seg ~4-5h; 8h ceiling
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv15k_conf035_sam2_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv15k_conf035_sam2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# A/0715 seg — the clean SAM1-vs-SAM2 per_tile A/B at conf 0.35 (opencv, full cull). The SAM1 arm already
# exists = ocv15k_frust_paint (yolov5_pertile_ocv15k SAM1, conf 0.35, full cull, IoU 0.369). This run makes
# the matching SAM2 arm at the SAME conf 0.35 so the ONLY difference is SAM1 -> SAM2. No conf-0.35 full-image
# SAM2 mask set existed (the conf sweep started at 0.40; the *_sam2 sets are GT-images-only), so we generate
# it CLEAN here (stage 1) with the EXACT conf-sweep config (opencv, target 3904, per_tile, SAM2, roi), only
# the confidence set to 0.35. Named pertile_sam2_conf035 / ocv15k_conf035 so it ALSO fills the missing
# conf-0.35 point of the SAM2 conf sweep (conf040/055/070 -> now +035).
#
# Compare ocv15k_conf035 against:
#   (1) ocv15k_frust_paint (0.369, SAM1 conf 0.35, same full cull) -> the clean SAM1-vs-SAM2 effect
#   (2) ocv15k_conf040/055/070 (SAM2) -> extends the conf sweep down to 0.35
#
# Reuses the existing opencv baseline 3DGS model (NO retrain). SEG ONLY (CPU-score locally vs warped opencv
# GT after; avoids overwriting the shared model-level test/ renders). NO render_360.
# ============================================================================

FIELD=field_A
DATE=20250715
VARIANT=opencv
MODEL_EXP=baseline                       # existing opencv baseline model (results/.../opencv/vanilla_3dgs/baseline)
ITER=15000
TARGET=3904                              # full-res on the ~3890px opencv frame (stride-32 safe) — matches the conf sweep
CONF=0.35                                # THE point of this run: SAM2 at the SAM1 control's conf
MASK_EXP=pertile_sam2_conf035            # NEW SAM2 conf-0.35 mask set (also the missing conf-sweep point)
SEG_EXP=ocv15k_conf035                   # NEW seg output name (extends ocv15k_conf040/055/070)

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"

MODEL="$REPO/results/reconstruction/phone/$FIELD/$DATE/$VARIANT/vanilla_3dgs/$MODEL_EXP"
MASK_OUT="$REPO/results/mask_generation/phone/$FIELD/$DATE/$VARIANT/yolo_sam_v1/$MASK_EXP"
SEG_OUT="$MODEL/segmentation_3d/$SEG_EXP"
IN_IMG="$REPO/input_plots/phone/$FIELD/$DATE/$VARIANT/images"
IN_MARK="$REPO/input_plots/phone/$FIELD/$DATE/$VARIANT/logs/marker_points3d.json"
SAM_W="$REPO/src/mask_generation/weights/sam2.1_l.pt"

echo "=== PREFLIGHT ==="
FAIL=0
[ -f "$MODEL/point_cloud/iteration_$ITER/point_cloud.ply" ] || { echo "  X no opencv baseline model at iter $ITER: $MODEL"; FAIL=1; }
[ -d "$IN_IMG" ]  || { echo "  X no opencv images: $IN_IMG"; FAIL=1; }
[ -f "$IN_MARK" ] || { echo "  X no opencv marker_points3d.json (roi + marker_exclude NEED it): $IN_MARK"; FAIL=1; }
[ -f "$SAM_W" ]   || { echo "  X missing SAM2 large weights: $SAM_W"; FAIL=1; }
[ -e "$MASK_OUT" ] && { echo "  X mask target exists (clean rerun wants a fresh name): $MASK_OUT"; FAIL=1; }
[ -d "$SEG_OUT" ] && [ -n "$(ls -A "$SEG_OUT" 2>/dev/null)" ] && { echo "  X seg target exists (no overwrite): $SEG_OUT"; FAIL=1; }
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: opencv baseline model + images + markers + SAM2 weights present, mask & seg targets fresh"

module purge
source ~/miniconda3/etc/profile.d/conda.sh

# ---------------- STAGE 1: MASK GENERATION — SAM2 per_tile @ conf 0.35 (clean) ----------------
echo ""; echo "======== STAGE 1 — full-res YOLOv5 (target $TARGET) + per_tile SAM2 + ROI @ conf $CONF ========"
conda activate wheat-maskgen
module load eth_proxy 2>/dev/null || true
python src/mask_generation/run_mask_generation.py \
  dataset=phone method=yolo_sam_v1 \
  dataset.plot_glob=$FIELD/$DATE/$VARIANT \
  method.target_image_size=$TARGET method.batch_size_yolo=2 \
  method.sam_crop_mode=per_tile method.sam_backend=sam2 \
  method.conf_threshold_good_box=$CONF \
  roi.enabled=true \
  experiment_name=$MASK_EXP
MASK_RC=$?
conda deactivate
echo "  mask-gen rc=$MASK_RC -> $MASK_OUT"
if [ $MASK_RC -ne 0 ] || [ -z "$(ls -A "$MASK_OUT/masks" 2>/dev/null)" ]; then
  echo "ABORTING — mask generation failed or produced no masks."; exit 1
fi

# ---------------- STAGE 2: 3D SEG (full cull, identical to ocv15k_frust_paint) ----------------
echo ""; echo "======== STAGE 2 — 3D seg ONLY (opencv $MODEL_EXP, iter $ITER) FULL cull ========"
module purge
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!; trap "kill $VRAM_PID 2>/dev/null" EXIT
export WHEAT_RUN_REPORT="$REPO/slurm_logs/run_report_${SLURM_JOB_ID}.txt"

# opencv variant, baseline model, SEG ONLY. FULL cull IDENTICAL to ocv15k_frust_paint: frustum + roi + marker
# (defaults marker_radius_m=0.075, roi_buffer_m=0.25), ground-cull tilt-fix by default (markers present),
# fast-paint + crop cache default. use_principal_point=false. The ONLY difference vs ocv15k_frust_paint is
# the masks (SAM2 here vs SAM1 there).
python src/run_reconstruction.py \
  dataset=phone plot=$FIELD date=$DATE sfm_variant=$VARIANT \
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
echo "  seg-only: rc=$SEG_RC -> $SEG_OUT  (CPU-score locally vs warped opencv GT after pulling 2DSeg)"
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
