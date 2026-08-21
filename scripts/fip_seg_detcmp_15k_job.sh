#!/bin/bash -l
#SBATCH -J fipseg_detcmp
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=4G       # 32 GB total — measured FIP seg RAM peak ~17 GB (crop cache); ~1.6x margin. No render_360 (the ~42 GB step).
#SBATCH --cpus-per-task=8
#SBATCH --time=03:59:00        # ceiling, kept strictly under 4h (shorter-job scheduling tier). Expected ~2 h/plot (train ~23m + 3 segs ~15/25/30m + metrics + 3 eval2d)
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fipseg_detcmp_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fipseg_detcmp_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# FIP 3D-SEG DETECTOR COMPARISON — one plot per job. Trains ONE 15k model (detector-independent) then
# segments it with the THREE detector mask sets, so the only variable is the 2D mask quality:
#     bad      = YOLOv5 @ 640   (yolo_sam_v1/fipseg_yv5_640_c35)
#     baseline = YOLOv5 @ 1280  (yolo_sam_v1/fipseg_yv5_1280_c35)
#     best     = YOLO11 + SAM   (yolo11_sam/fipseg_yolo11_c35)
# Recon (locked): gsplat + use_principal_point=true + 15k + resolution 1 + absgrad OFF.
# Steps: Train + Render + Metrics ONCE (recon quality) -> then per detector Seg + Eval(6) + Eval2D(6b).
# NO render_360 (do it later as a render-only job for the chosen best config).
# Optimizations: crop cache + fast-paint (default ON, bit-identical) + frustum_cull (lossless). NO
# roi/marker/ground cull (FIP has no markers; ground cull is a result-changer).
#
# Submit once per plot (they run independently / in parallel):
#     for P in 461 462 463 464 465 466 467; do
#       sbatch /cluster/project/cropsci/peugste/wheat3dgs/scripts/fip_seg_detcmp_15k_job.sh plot_$P
#     done
# ============================================================================
set -euo pipefail
PLOT="${1:-}"
case "$PLOT" in plot_46[1-7]) ;; *) echo "usage: sbatch $0 <plot_461..plot_467>"; exit 1 ;; esac

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"
RECON_EXP=fipseg15k_pp
ITER=15000
MODEL="$REPO/results/reconstruction/fip/$PLOT/vanilla_3dgs/$RECON_EXP"

# detector -> "method mask_gen_experiment seg_exp_name"
DETS=(
  "yolo_sam_v1 fipseg_yv5_640_c35  seg_yv5_640"
  "yolo_sam_v1 fipseg_yv5_1280_c35 seg_yv5_1280"
  "yolo11_sam  fipseg_yolo11_c35   seg_yolo11"
)

echo "=== PREFLIGHT ($PLOT) ==="
FAIL=0
for d in "${DETS[@]}"; do
  set -- $d; METHOD=$1; MASKEXP=$2
  M="$REPO/results/mask_generation/fip/$PLOT/$METHOD/$MASKEXP/masks"
  [ -n "$(ls -A "$M" 2>/dev/null)" ] || { echo "  X no masks: $M — run mask-gen trio first"; FAIL=1; }
done
[ -d "$REPO/input_plots/fip/$PLOT/sparse" ] || { echo "  X no sparse/ for $PLOT"; FAIL=1; }
[ "$FAIL" -ne 0 ] && { echo "ABORTING."; exit 1; }
echo "  ok: 3 mask sets + sparse present"

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

RECON_COMMON=(dataset=fip plot=$PLOT experiment_name=$RECON_EXP
  reconstruction.iterations=$ITER reconstruction.resolution=1
  reconstruction.use_principal_point=true)

# ---- 1. Train + Render + Metrics ONCE (recon quality; detector-independent) ----
echo ""; echo "======== [$PLOT] TRAIN + RENDER + METRICS (15k, gsplat, pp=true) ========"
TRAIN_RC=0
python src/run_reconstruction.py "${RECON_COMMON[@]}" \
  run_train=true run_render=true run_metrics=true || TRAIN_RC=$?
[ "$TRAIN_RC" -ne 0 ] && { echo "TRAIN failed (rc=$TRAIN_RC) — cannot seg. Aborting."; exit $TRAIN_RC; }

# ---- 2. Seg + Eval(6) + Eval2D(6b) per detector (reuse the trained model) ----
SEG_RC=0
for d in "${DETS[@]}"; do
  set -- $d; METHOD=$1; MASKEXP=$2; SEGEXP=$3
  echo ""; echo "======== [$PLOT] SEG $SEGEXP  (method=$METHOD masks=$MASKEXP) ========"
  python src/run_reconstruction.py "${RECON_COMMON[@]}" \
    run_seg=true run_eval=true run_eval_2d=true \
    segmentation_3d.exp_name=$SEGEXP \
    segmentation_3d.detection_method=$METHOD \
    segmentation_3d.mask_gen_experiment=$MASKEXP \
    segmentation_3d.frustum_cull=true \
    segmentation_3d.use_mask_cache=true || SEG_RC=$?
done

echo ""; echo "======== STATUS ($PLOT) ========"
for d in "${DETS[@]}"; do
  set -- $d; SEGEXP=$3
  ss="$MODEL/segmentation_3d/$SEGEXP"
  heads=$(grep -o '"wheat_heads_found": [0-9]*' "$ss/seg_summary.json" 2>/dev/null || echo '?')
  iou=$(grep -o '"iou": [0-9.]*' "$ss/eval_2d/metrics_2d.json" 2>/dev/null | head -1 || echo '?')
  echo "  $SEGEXP: heads=$heads  $iou"
done
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT" || true
exit $SEG_RC
