#!/bin/bash -l
#SBATCH -J seg_A0715_ocv15k_absgrad
#SBATCH --gpus=a100-pcie-40gb:1   # A100 40 GB — opencv absgrad model is ~3 M gaussians (2.5x baseline);
                                  # rtx_4090 (24 GB) CUDA-OOM'd in flashsplat seg on the 30k opencv model
#SBATCH --mem-per-cpu=8G          # 64 GB total — phone 3D seg decodes many full-res masks (RAM-heavy)
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00           # masks reused (no mask-gen) + seg ~5-7h on the denser absgrad model; 24h ceiling
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv15k_absgrad_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv15k_absgrad_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# A/0715 seg — OPENCV 15k ABSGRAD, full best-cull config. Does AbsGS densification help the 3D seg on the
# HEADLINE phone config (opencv)? Clean A/B: SAME opencv SfM + SAME yolov5_pertile_ocv15k SAM1 masks +
# SAME full cull as the absgrad-OFF control ocv15k_frust_paint (IoU 0.369). The ONLY difference is the model:
# opencv/vanilla_3dgs/absgrad (absgrad=true, densify_grad_threshold=0.0008, 15k) vs the absgrad-off baseline.
# Chose OPENCV over pinhole because opencv is our reported phone config (opencv > pinhole for recon, and the
# warped-GT eval_2d is valid on opencv). Mirrors the FIP absgrad A/B, now on the phone headline config.
#
# Compare ocv15k_absgrad_fullcull against ocv15k_frust_paint (IoU 0.369, absgrad off) -> the AbsGS effect.
#
# Reuses the existing opencv ABSGRAD 3DGS model (NO retrain; trained on Euler, phone_absgrad_opencv_Y) and the
# existing yolov5_pertile_ocv15k masks (NO mask-gen). SEG ONLY (CPU-score locally vs warped opencv GT after,
# to avoid overwriting the shared model-level test/ renders). NO render_360.
# ============================================================================

FIELD=field_A
DATE=20250715
VARIANT=opencv
MODEL_EXP=absgrad                        # existing opencv ABSGRAD model (results/.../opencv/vanilla_3dgs/absgrad)
ITER=15000
MASK_EXP=yolov5_pertile_ocv15k           # REUSE the SAME SAM1 per_tile masks ocv15k_frust_paint used
SEG_EXP=ocv15k_absgrad_fullcull          # NEW seg output name (never collides)

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"

MODEL="$REPO/results/reconstruction/phone/$FIELD/$DATE/$VARIANT/vanilla_3dgs/$MODEL_EXP"
MASK_OUT="$REPO/results/mask_generation/phone/$FIELD/$DATE/$VARIANT/yolo_sam_v1/$MASK_EXP"
SEG_OUT="$MODEL/segmentation_3d/$SEG_EXP"
IN_MARK="$REPO/input_plots/phone/$FIELD/$DATE/$VARIANT/logs/marker_points3d.json"

echo "=== PREFLIGHT ==="
FAIL=0
[ -f "$MODEL/point_cloud/iteration_$ITER/point_cloud.ply" ] || { echo "  X no opencv ABSGRAD model at iter $ITER: $MODEL (was it trained/kept on Euler?)"; FAIL=1; }
grep -q 'absgrad: true' "$MODEL/config.yaml" 2>/dev/null || echo "  ! WARN: could not confirm absgrad:true in $MODEL/config.yaml"
[ -n "$(ls -A "$MASK_OUT/masks" 2>/dev/null)" ] || { echo "  X no reused masks: $MASK_OUT/masks (expected on Euler from earlier opencv runs)"; FAIL=1; }
[ -f "$IN_MARK" ] || { echo "  X no opencv marker_points3d.json (roi_cull + marker_exclude NEED it): $IN_MARK"; FAIL=1; }
[ -d "$SEG_OUT" ] && [ -n "$(ls -A "$SEG_OUT" 2>/dev/null)" ] && { echo "  X seg target exists (no overwrite): $SEG_OUT"; FAIL=1; }
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: opencv absgrad model(iter $ITER) + reused masks + opencv markers present, seg target fresh"

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!; trap "kill $VRAM_PID 2>/dev/null" EXIT
export WHEAT_RUN_REPORT="$REPO/slurm_logs/run_report_${SLURM_JOB_ID}.txt"

# opencv variant, ABSGRAD model. SEG ONLY. FULL cull identical to ocv15k_frust_paint: frustum + roi + marker
# (defaults marker_radius_m=0.075, roi_buffer_m=0.25), ground-cull tilt-fix by default (markers present),
# fast-paint + crop cache default. use_principal_point=false (matches how the model was trained). NO absgrad
# flag needed here — absgrad only affects TRAINING (already baked into the model); seg is flashsplat.
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
