#!/bin/bash -l
#SBATCH -J seg_A0715_ocv15k_sam1_conf070
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=5G       # 40 GB total — measured seg-only peak ~34 GB (seg ~24 GB, then auto post-steps 4b/4c spike to ~34 GB); 5G leaves ~6 GB margin. Lower than the old 64 GB so more jobs fit under QOSMaxMemoryPerUser.
#SBATCH --cpus-per-task=8
#SBATCH --time=07:59:00        # per_tile SAM1 mask-gen ~15 min + opencv baseline (1.55 M) seg ~4-5h; conf 0.70 keeps fewer heads so seg is if anything faster than the 0.35 baseline; 8h ceiling
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv15k_sam1_conf070_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv15k_sam1_conf070_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# A/0715 seg — the conf lever WITHIN SAM1 (per_tile, opencv, full cull). The SAM1 conf-0.35 arm already
# exists = ocv15k_frust_paint (yolov5_pertile_ocv15k SAM1, conf 0.35, full cull, IoU 0.369). This run makes
# the SAME config at conf 0.70 so the ONLY difference is 0.35 -> 0.70. It answers two things at once:
#   (1) does higher conf also lift SAM1 (as it did SAM2: 0.360 -> 0.394)? -> confirms conf is a separable
#       main effect, not a SAM2-only quirk.
#   (2) SAM1@0.70 vs SAM2@0.70 (ocv15k_conf070, 0.394): the clean SAM1-vs-SAM2 A/B at the high operating point.
#
# No per_tile SAM1 conf-0.70 mask set exists (the SAM1 masks are conf 0.35 only), so it is generated CLEAN
# here (stage 1) with the EXACT control config (opencv, target 3904, per_tile, SAM1, roi), only conf -> 0.70.
# Named pertile_sam1_conf070 / ocv15k_sam1_conf070.
#
# Reuses the existing opencv baseline 3DGS model (NO retrain). SEG ONLY (CPU-score locally vs warped opencv
# GT after; avoids overwriting the shared model-level test/ renders). NO render_360.
# ============================================================================

FIELD=field_A
DATE=20250715
VARIANT=opencv
MODEL_EXP=baseline                       # existing opencv baseline model (results/.../opencv/vanilla_3dgs/baseline)
ITER=15000
TARGET=3904                              # full-res on the ~3890px opencv frame (stride-32 safe) — matches the control
CONF=0.70                                # THE point of this run: SAM1 per_tile at the high operating point
MASK_EXP=pertile_sam1_conf070            # NEW SAM1 conf-0.70 mask set
SEG_EXP=ocv15k_sam1_conf070              # NEW seg output name

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"

MODEL="$REPO/results/reconstruction/phone/$FIELD/$DATE/$VARIANT/vanilla_3dgs/$MODEL_EXP"
MASK_OUT="$REPO/results/mask_generation/phone/$FIELD/$DATE/$VARIANT/yolo_sam_v1/$MASK_EXP"
SEG_OUT="$MODEL/segmentation_3d/$SEG_EXP"
IN_IMG="$REPO/input_plots/phone/$FIELD/$DATE/$VARIANT/images"
IN_MARK="$REPO/input_plots/phone/$FIELD/$DATE/$VARIANT/logs/marker_points3d.json"
SAM_W="$REPO/src/mask_generation/weights/sam_vit_h_4b8939.pth"   # SAM1 ViT-H checkpoint

echo "=== PREFLIGHT ==="
FAIL=0
[ -f "$MODEL/point_cloud/iteration_$ITER/point_cloud.ply" ] || { echo "  X no opencv baseline model at iter $ITER: $MODEL"; FAIL=1; }
[ -d "$IN_IMG" ]  || { echo "  X no opencv images: $IN_IMG"; FAIL=1; }
[ -f "$IN_MARK" ] || { echo "  X no opencv marker_points3d.json (roi + marker_exclude NEED it): $IN_MARK"; FAIL=1; }
[ -f "$SAM_W" ]   || { echo "  X missing SAM1 ViT-H weights: $SAM_W"; FAIL=1; }
[ -e "$MASK_OUT" ] && { echo "  X mask target exists (clean rerun wants a fresh name): $MASK_OUT"; FAIL=1; }
[ -d "$SEG_OUT" ] && [ -n "$(ls -A "$SEG_OUT" 2>/dev/null)" ] && { echo "  X seg target exists (no overwrite): $SEG_OUT"; FAIL=1; }
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: opencv baseline model + images + markers + SAM1 weights present, mask & seg targets fresh"

module purge
source ~/miniconda3/etc/profile.d/conda.sh

# ---------------- STAGE 1: MASK GENERATION — SAM1 per_tile @ conf 0.70 (clean) ----------------
echo ""; echo "======== STAGE 1 — full-res YOLOv5 (target $TARGET) + per_tile SAM1 + ROI @ conf $CONF ========"
conda activate wheat-maskgen
module load eth_proxy 2>/dev/null || true
python src/mask_generation/run_mask_generation.py \
  dataset=phone method=yolo_sam_v1 \
  dataset.plot_glob=$FIELD/$DATE/$VARIANT \
  method.target_image_size=$TARGET method.batch_size_yolo=2 \
  method.sam_crop_mode=per_tile method.sam_backend=sam1 \
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
# the mask confidence (0.70 here vs 0.35 there); both are SAM1 per_tile.
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
