#!/bin/bash -l
#SBATCH -J fip_absgrad_AB
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=4G       # 32 GB total — FIP seg RAM peak ~17 GB (crop cache); ample. No render_360.
#SBATCH --cpus-per-task=8
#SBATCH --time=03:59:00        # <4h short-job tier. 2 plots; per plot ~ train 26m + seg ~30m + render/metrics/eval; ~2.5h expected.
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_absgrad_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_absgrad_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# FIP 3D-SEG AbsGrad A/B — the AbsGS-ON arm on the CURRENT masks (the OFF control already exists as
# fipseg15k_pp/seg_yv5_1280, absgrad=false). The old ABSGRAD_RESULTS seg comparison used the OLD `initial`
# mask set, so this re-does AbsGS-ON with today's fipseg_yv5_1280_c35 masks for an apples-to-apples A/B.
# For each plot: TRAIN (absgrad=true) + SEG with YOLOv5@1280. Recon: gsplat + use_principal_point=TRUE +
# 15k + resolution 1 + absgrad ON (densify_grad_threshold=0.0008 REQUIRED with absgrad, else over-densify/OOM).
# Only variable vs the control = absgrad. Seg opts: crop cache + frustum_cull (lossless); NO roi/marker/ground
# cull (FIP has no markers). No render_360. Compare eval_2d IoU vs fipseg15k_pp/seg_yv5_1280.
#
# This script does plots 461 + 462. Submit:
#     sbatch /cluster/project/cropsci/peugste/wheat3dgs/scripts/fip_absgrad_seg_461_462_job.sh
# ============================================================================
set -euo pipefail
PLOTS=(plot_461 plot_462)

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"
RECON_EXP=fipseg15k_absgrad
ITER=15000
METHOD=yolo_sam_v1
MASKEXP=fipseg_yv5_1280_c35
SEGEXP=seg_yv5_1280

echo "=== PREFLIGHT ==="
FAIL=0
for PLOT in "${PLOTS[@]}"; do
  M="$REPO/results/mask_generation/fip/$PLOT/$METHOD/$MASKEXP/masks"
  [ -n "$(ls -A "$M" 2>/dev/null)" ] || { echo "  X no masks: $M"; FAIL=1; }
  [ -d "$REPO/input_plots/fip/$PLOT/sparse" ] || { echo "  X no sparse/ for $PLOT"; FAIL=1; }
done
[ "$FAIL" -ne 0 ] && { echo "ABORTING."; exit 1; }
echo "  ok: masks + sparse present for ${PLOTS[*]}"

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

export WHEAT_RENDERER=gsplat
export WHEAT_RUN_REPORT="$REPO/slurm_logs/run_report_${SLURM_JOB_ID}.txt"
export WHEAT_SEG_TIMING=1
VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!; trap "kill $VRAM_PID 2>/dev/null" EXIT

RC=0
for PLOT in "${PLOTS[@]}"; do
  RECON_COMMON=(dataset=fip plot=$PLOT experiment_name=$RECON_EXP
    reconstruction.iterations=$ITER reconstruction.resolution=1
    reconstruction.absgrad=true reconstruction.densify_grad_threshold=0.0008
    reconstruction.use_principal_point=true)

  echo ""; echo "======== [$PLOT] TRAIN + RENDER + METRICS (15k, gsplat, pp=true, absgrad=TRUE) ========"
  TRC=0
  python src/run_reconstruction.py "${RECON_COMMON[@]}" \
    run_train=true run_render=true run_metrics=true || TRC=$?
  [ "$TRC" -ne 0 ] && { echo "TRAIN failed ($PLOT, rc=$TRC) — skipping its seg."; RC=$TRC; continue; }

  echo ""; echo "======== [$PLOT] SEG $SEGEXP (absgrad model) ========"
  python src/run_reconstruction.py "${RECON_COMMON[@]}" \
    run_seg=true run_eval=true run_eval_2d=true \
    segmentation_3d.exp_name=$SEGEXP \
    segmentation_3d.detection_method=$METHOD \
    segmentation_3d.mask_gen_experiment=$MASKEXP \
    segmentation_3d.frustum_cull=true \
    segmentation_3d.use_mask_cache=true || RC=$?
done

echo ""; echo "======== STATUS ========"
for PLOT in "${PLOTS[@]}"; do
  ss="$REPO/results/reconstruction/fip/$PLOT/vanilla_3dgs/$RECON_EXP/segmentation_3d/$SEGEXP"
  iou=$(grep -o '"iou": [0-9.]*' "$ss/eval_2d/metrics_2d.json" 2>/dev/null | head -1 || echo '?')
  echo "  $PLOT $SEGEXP: $iou"
done
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT" || true
exit $RC
