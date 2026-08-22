#!/bin/bash -l
#SBATCH -J seg_A0715_ocv15k_perhead_sam2
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G       # 48 GB total — seg-only peak ~34 GB (seg + post-steps 4b/4c); per_head can hold more masks than per_tile, so a bit more margin than the per_tile jobs
#SBATCH --cpus-per-task=8
#SBATCH --time=11:59:00        # per_head 15k seg ~5-8h (mask count similar to per_tile's ~24.6k, but per-head lift/match can run longer); 12h ceiling
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv15k_perhead_sam2_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv15k_perhead_sam2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# A/0715 seg — PER_HEAD SAM2 @ conf 0.35 (opencv, full cull). SEG ONLY: reuses the per_head SAM2 masks
# pre-baked by scripts/phone_A0715_perhead_maskgen_job.sh (MASK_EXP perhead_sam2_conf035) so no mask-gen
# runs inside this 24h slot. Everything else is IDENTICAL to the per_tile control ocv15k_frust_paint
# (opencv baseline model, full cull, conf 0.35) EXCEPT sam_crop_mode=per_head AND SAM2. So it gives:
#   ocv15k_perhead_sam2 vs ocv15k_perhead_sam1 -> SAM1-vs-SAM2 at per_head, conf 0.35.
#   ocv15k_perhead_sam2 vs ocv15k_conf035 (per_tile SAM2, 0.335) -> granularity effect at SAM2, conf 0.35.
#
# Reuses the existing opencv baseline 3DGS model (NO retrain). CPU-score locally vs warped opencv GT after
# pulling 2DSeg. NO render_360.
# ============================================================================

FIELD=field_A
DATE=20250715
VARIANT=opencv
MODEL_EXP=baseline                       # opencv baseline model (results/.../opencv/vanilla_3dgs/baseline, 1.55 M)
ITER=15000
MASK_EXP=perhead_sam2_conf035            # per_head SAM2 masks pre-baked by the Queue-B mask-gen job
SEG_EXP=ocv15k_perhead_sam2              # NEW seg output name

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"

MODEL="$REPO/results/reconstruction/phone/$FIELD/$DATE/$VARIANT/vanilla_3dgs/$MODEL_EXP"
MASK_OUT="$REPO/results/mask_generation/phone/$FIELD/$DATE/$VARIANT/yolo_sam_v1/$MASK_EXP"
SEG_OUT="$MODEL/segmentation_3d/$SEG_EXP"
IN_IMG="$REPO/input_plots/phone/$FIELD/$DATE/$VARIANT/images"
IN_MARK="$REPO/input_plots/phone/$FIELD/$DATE/$VARIANT/logs/marker_points3d.json"

echo "=== PREFLIGHT ==="
FAIL=0
[ -f "$MODEL/point_cloud/iteration_$ITER/point_cloud.ply" ] || { echo "  X no opencv baseline model at iter $ITER: $MODEL"; FAIL=1; }
[ -d "$IN_IMG" ]  || { echo "  X no opencv images: $IN_IMG"; FAIL=1; }
[ -f "$IN_MARK" ] || { echo "  X no opencv marker_points3d.json (roi + marker_exclude NEED it): $IN_MARK"; FAIL=1; }
[ -n "$(ls -A "$MASK_OUT/masks" 2>/dev/null)" ] || { echo "  X per_head SAM2 masks missing (run the Queue-B mask-gen job first): $MASK_OUT/masks"; FAIL=1; }
[ -d "$SEG_OUT" ] && [ -n "$(ls -A "$SEG_OUT" 2>/dev/null)" ] && { echo "  X seg target exists (no overwrite): $SEG_OUT"; FAIL=1; }
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: opencv baseline model + images + markers + per_head SAM2 masks present, seg target fresh"

module purge
source ~/miniconda3/etc/profile.d/conda.sh
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
# the masks (per_head SAM2 here vs per_tile SAM1 there).
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
echo "  seg-only: rc=$SEG_RC -> $SEG_OUT  (CPU-score locally vs warped opencv GT after pulling 2DSeg)"
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
