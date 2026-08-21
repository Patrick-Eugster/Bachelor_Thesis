#!/bin/bash -l
#SBATCH -J seg_A0715_conftest
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=5G       # 40 GB total — SEG-ONLY peaks ~25 GB CPU RAM (measured); render_360/eval (the ~42 GB steps) are NOT run here
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00        # seg-only; A/0715 opencv full cull ran ~4-7 h; higher conf = fewer masks = faster. 12 h ceiling
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_conftest_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_conftest_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# 3D-SEG CONF TEST — one arm per job. Segments the A/0715 opencv model on the mask set generated at a
# given YOLO confidence, so we can pick the operating conf from DOWNSTREAM 3D output (not the 2D curve).
# Submit THREE times, once per conf (they queue/run independently, possibly in parallel):
#     sbatch .../phone_seg_A0715_conf_test_job.sh 0.40
#     sbatch .../phone_seg_A0715_conf_test_job.sh 0.55
#     sbatch .../phone_seg_A0715_conf_test_job.sh 0.70
#
# Baseline config (same as the analyzed cull runs, PLUS the two new fixes):
#   roi_cull + marker_exclude + frustum_cull + fast-paint (default) + ground-fix (default)
#   + marker_radius_rel=0.065  (scale-free ~0.20u marker sphere — fixes the marker rainbow)
# SEG-ONLY: no eval / no render_360 (decision metrics = seg_summary head counts + head-size stats, computed
# locally from all_obj_labels.pth). Reads masks from pertile_sam2_conf0XX (made by the mask-gen conf-sweep
# job). Fresh seg output name per conf -> ABORTS if it already exists (no overwrite/mix).
# ============================================================================
set -euo pipefail
CONF="${1:-}"
case "$CONF" in
  0.40) CC=040 ;;
  0.55) CC=055 ;;
  0.70) CC=070 ;;
  *) echo "usage: sbatch $0 <0.40|0.55|0.70>"; exit 1 ;;
esac

FIELD=field_A
DATE=20250715
VARIANT=opencv
MODEL_EXP=baseline
ITER=15000
MASK_EXP=pertile_sam2_conf${CC}          # per-conf masks from phone_maskgen_A0715_conf_sweep_job.sh
SEG_EXP=ocv15k_conf${CC}                  # per-conf seg output -> never collides across confs

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"
MODEL="$REPO/results/reconstruction/phone/$FIELD/$DATE/$VARIANT/vanilla_3dgs/$MODEL_EXP"
MASK_OUT="$REPO/results/mask_generation/phone/$FIELD/$DATE/$VARIANT/yolo_sam_v1/$MASK_EXP"
SEG_OUT="$MODEL/segmentation_3d/$SEG_EXP"
IN_MARK="$REPO/input_plots/phone/$FIELD/$DATE/$VARIANT/logs/marker_points3d.json"

echo "=== PREFLIGHT (conf=$CONF -> masks=$MASK_EXP, seg=$SEG_EXP) ==="
FAIL=0
[ -f "$MODEL/point_cloud/iteration_$ITER/point_cloud.ply" ] || { echo "  X no model at iter $ITER: $MODEL"; FAIL=1; }
[ -n "$(ls -A "$MASK_OUT/masks" 2>/dev/null)" ] || { echo "  X no masks at $MASK_OUT/masks — run the mask-gen conf-sweep job first"; FAIL=1; }
[ -f "$IN_MARK" ] || { echo "  X no marker_points3d.json (roi_cull + marker_exclude NEED it): $IN_MARK"; FAIL=1; }
{ [ -d "$SEG_OUT" ] && [ -n "$(ls -A "$SEG_OUT" 2>/dev/null)" ]; } && { echo "  X seg target exists: $SEG_OUT"; FAIL=1; }
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: model(iter $ITER) + conf-$CONF masks + markers present, seg target fresh"

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!; trap "kill $VRAM_PID 2>/dev/null" EXIT
export WHEAT_RUN_REPORT="$REPO/slurm_logs/run_report_${SLURM_JOB_ID}.txt"
export WHEAT_SEG_TIMING=1

echo ""; echo "======== 3D seg conf=$CONF (roi+marker+frustum, marker_radius_rel=0.065, fast-paint) ========"
# capture rc instead of letting set -e abort here, so the STATUS + run_report tail below always print
SEG_RC=0
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
  segmentation_3d.exp_name=$SEG_EXP || SEG_RC=$?

echo ""; echo "======== STATUS (conf=$CONF) ========"
echo "  seg: rc=$SEG_RC -> $SEG_OUT"
echo "  heads: $(grep -o '\"wheat_heads_found\": [0-9]*' "$SEG_OUT/seg_summary.json" 2>/dev/null || echo '?')"
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT" || true
# exit with the actual seg result so SLURM marks the job correctly (not the run_report cat's status)
exit $SEG_RC
