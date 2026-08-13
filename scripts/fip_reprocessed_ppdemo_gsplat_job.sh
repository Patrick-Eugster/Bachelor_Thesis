#!/bin/bash -l
#SBATCH -J fip_reproc_ppdemo_gsplat
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G      # 36 GB total — FIP 36 imgs -> RAM ~10-15 GB (fewer imgs than phone), ample headroom
#SBATCH --cpus-per-task=6
#SBATCH --time=03:59:00       # 7 plots x ~29 min (15k train + render + metrics) ~= 3.4h -> under 4h
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_reproc_ppdemo_gsplat_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_reproc_ppdemo_gsplat_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# FIP RE-CENTERED PP DEMONSTRATION (gsplat). Trains 3DGS on the reprocessed, RE-CENTERED undistorted_png FIP
# (SIMPLE_PINHOLE, cx=W/2 exact) with pp OFF. Point: the pixel-shift bug has TWO independent cures —
#   (1) fix the renderer (use_principal_point, the +7.8 dB headline), OR
#   (2) re-center the pp at undistortion time (what this reprocessed data already did).
# On centered data, pp OFF should already give render-vs-GT shift ~= 0 (measured offline with
# src/analysis/make_pp_shift_figure.py / fip_principal_point_offset.py) and good PSNR — proving cure #2.
# recon-only: train + render + metrics (NO seg). gsplat, resolution 1, 15000 iters, DEFAULT densify, pp OFF.
# Reads   input_plots/fip/<plot>/reprocessed_png/{images,sparse/0}   (via sfm_variant=reprocessed_png).
# Writes  results/reconstruction/fip/<plot>/reprocessed_png/vanilla_3dgs/recentered_ppoff  (own subtree).
# NB PSNR here is NOT comparable to the original FIP (different SfM/undistortion) — it is a self-metric;
#    the render-vs-GT SHIFT is the demonstration signal.
# ============================================================================

VARIANT=reprocessed_png
EXP=recentered_ppoff
PLOTS=(plot_461 plot_462 plot_463 plot_464 plot_465 plot_466 plot_467)

REPO=/cluster/project/cropsci/peugste/wheat3dgs
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
cd "$REPO"
nvidia-smi

# ── PREFLIGHT: abort if any target exists or a reprocessed_png input is missing ──
echo "=== PREFLIGHT ==="
COLLIDE=0
for P in "${PLOTS[@]}"; do
  OUT="$REPO/results/reconstruction/fip/$P/$VARIANT/vanilla_3dgs/$EXP"
  IN="$REPO/input_plots/fip/$P/$VARIANT/sparse/0"
  if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "  X EXISTS: $OUT"; COLLIDE=1
  elif [ ! -d "$IN" ]; then echo "  X MISSING INPUT: $IN"; COLLIDE=1
  else echo "  ok: $OUT"; fi
done
[ "$COLLIDE" -ne 0 ] && { echo "ABORTING — target exists or reprocessed_png input missing."; exit 1; }

# ── peak VRAM/RAM loggers ──
VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!
trap "kill $VRAM_PID 2>/dev/null" EXIT
export WHEAT_RUN_REPORT="$REPO/slurm_logs/run_report_${SLURM_JOB_ID}.txt"

declare -A STATUS
for P in "${PLOTS[@]}"; do
  echo ""
  echo "================ gsplat  $P  (train+render+metrics, 15k, default densify, pp OFF) ================"
  python src/run_reconstruction.py \
    dataset=fip plot=$P \
    run_train=true run_render=true run_metrics=true \
    reconstruction.iterations=15000 reconstruction.resolution=1 reconstruction.absgrad=false \
    experiment_name=$EXP sfm_variant=$VARIANT
  rc=$?
  if [ $rc -eq 0 ]; then STATUS["$P"]=OK; else STATUS["$P"]="FAIL(rc=$rc)"; fi
done

echo ""
echo "================ gsplat reprocessed_png PER-PLOT STATUS ================"
for P in "${PLOTS[@]}"; do
  echo "  $P : ${STATUS[$P]}  -> results/reconstruction/fip/$P/$VARIANT/vanilla_3dgs/$EXP"
done
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
