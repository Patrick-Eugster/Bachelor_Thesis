#!/bin/bash -l
#SBATCH -J seg_A0715_ocv15k_mR065
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G       # 64 GB total — phone 3D seg decodes many full-res masks (RAM-heavy)
#SBATCH --cpus-per-task=8
#SBATCH --time=09:59:00        # masks reused (no mask-gen) + seg ~4-5h (fast-paint); 10h ceiling for safety
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv15k_mR065_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv15k_mR065_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# A/0715 seg — OPENCV 15k marker-radius A/B. The best-config run ocv15k_frust_paint (IoU 0.369) used the
# DEFAULT marker_radius_m=0.075, which covers only the physical plate (~13 cm) — but the RECONSTRUCTED
# marker blob is ~3x that (~0.20 world units, plate+shadow+halo baked by 3DGS), so markers stay partly
# colored (visible in the seg videos). This run is the SAME model + SAME SAM1 masks + SAME full cull, with
# ONLY the marker exclusion radius changed to marker_radius_rel=0.065 (~0.20 u, the scale-free radius the
# conf sweep used). It isolates how much better marker exclusion lifts precision.
#
# Compare ocv15k_markerR065 against ocv15k_frust_paint (IoU 0.369, radius 0.075) -> the marker-removal effect.
#
# Reuses the existing opencv 3DGS model (NO retrain) and the existing yolov5_pertile_ocv15k masks (NO
# mask-gen — they're on Euler from the earlier opencv runs). SEG ONLY (CPU-score locally after, to avoid
# overwriting the shared model-level test/ renders). NO render_360.
# ============================================================================

FIELD=field_A
DATE=20250715
VARIANT=opencv
MODEL_EXP=baseline                       # existing opencv 3DGS model (results/.../opencv/vanilla_3dgs/<MODEL_EXP>)
ITER=15000
MASK_EXP=yolov5_pertile_ocv15k           # REUSE the SAME SAM1 per_tile masks ocv15k_frust_paint used
SEG_EXP=ocv15k_markerR065                # NEW seg output name (never collides)

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"

MODEL="$REPO/results/reconstruction/phone/$FIELD/$DATE/$VARIANT/vanilla_3dgs/$MODEL_EXP"
MASK_OUT="$REPO/results/mask_generation/phone/$FIELD/$DATE/$VARIANT/yolo_sam_v1/$MASK_EXP"
SEG_OUT="$MODEL/segmentation_3d/$SEG_EXP"
IN_MARK="$REPO/input_plots/phone/$FIELD/$DATE/$VARIANT/logs/marker_points3d.json"

echo "=== PREFLIGHT ==="
FAIL=0
[ -f "$MODEL/point_cloud/iteration_$ITER/point_cloud.ply" ] || { echo "  X no opencv model at iter $ITER: $MODEL"; FAIL=1; }
[ -n "$(ls -A "$MASK_OUT/masks" 2>/dev/null)" ] || { echo "  X no reused masks: $MASK_OUT/masks (expected on Euler from earlier opencv runs)"; FAIL=1; }
[ -f "$IN_MARK" ] || { echo "  X no opencv marker_points3d.json (roi_cull + marker_exclude NEED it): $IN_MARK"; FAIL=1; }
[ -d "$SEG_OUT" ] && [ -n "$(ls -A "$SEG_OUT" 2>/dev/null)" ] && { echo "  X seg target exists (no overwrite): $SEG_OUT"; FAIL=1; }
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: opencv model(iter $ITER) + reused masks + opencv markers present, seg target fresh"

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!; trap "kill $VRAM_PID 2>/dev/null" EXIT
export WHEAT_RUN_REPORT="$REPO/slurm_logs/run_report_${SLURM_JOB_ID}.txt"

# opencv variant. SEG ONLY (no run_eval/run_eval_2d -> avoids overwriting the shared model-level test/
# renders; CPU-score locally against the warped opencv GT afterwards). FULL cull identical to
# ocv15k_frust_paint EXCEPT marker_radius_rel=0.065 (the only changed knob). fast-paint + crop cache default.
python src/run_reconstruction.py \
  dataset=phone plot=$FIELD date=$DATE sfm_variant=$VARIANT \
  experiment_name=$MODEL_EXP \
  reconstruction.iterations=$ITER reconstruction.resolution=1 \
  reconstruction.use_principal_point=false \
  run_seg=true \
  segmentation_3d.frustum_cull=true \
  segmentation_3d.roi_cull=true \
  segmentation_3d.marker_exclude=true \
  segmentation_3d.marker_radius_rel=0.065 \
  segmentation_3d.detection_method=yolo_sam_v1 \
  segmentation_3d.mask_gen_experiment=$MASK_EXP \
  segmentation_3d.exp_name=$SEG_EXP
SEG_RC=$?

echo ""; echo "======== STATUS ========"
echo "  seg-only: rc=$SEG_RC -> $SEG_OUT  (CPU-score locally vs warped opencv GT after pulling 2DSeg)"
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
