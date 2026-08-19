#!/bin/bash -l
#SBATCH -J fip461_gsplat_ppoff_15k
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G      # 36 GB total — one FIP plot (36 imgs) -> RAM ~10-15 GB, ample headroom
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00       # one plot: 15k train + render + metrics ~= 25-30 min -> 1h is plenty
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip461_gsplat_ppoff_15k_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip461_gsplat_ppoff_15k_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# FIP plot_461 — gsplat renderer, principal-point flag OFF, 15k, NO absgrad.
# Purpose: fill the missing empirical cell — gsplat WITHOUT the pp correction on the
# ORIGINAL (off-center-pp) FIP SfM. The code gates cx/cy on use_principal_point for BOTH
# renderers (camera_utils.py:54), so gsplat+pp-off falls back to the image-center projection
# and should REPRODUCE the pixel shift (low PSNR), same as diff-gaussian-rasterization+pp-off.
# recon-only: train + render + metrics (NO seg). resolution 1, DEFAULT densify (absgrad=false).
# Reads   input_plots/fip/plot_461/{images,sparse/0}   (default sfm_variant — original data).
# Writes  results/reconstruction/fip/plot_461/vanilla_3dgs/gsplat_ppoff_15k  (own subtree).
# ============================================================================

PLOT=plot_461
EXP=gsplat_ppoff_15k

REPO=/cluster/project/cropsci/peugste/wheat3dgs
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
cd "$REPO"
export WHEAT_RENDERER=gsplat   # explicit (gsplat is already the default) — this run is the gsplat arm
nvidia-smi

# ── PREFLIGHT: abort if the target already exists ──
echo "=== PREFLIGHT ==="
OUT="$REPO/results/reconstruction/fip/$PLOT/vanilla_3dgs/$EXP"
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "ABORTING — target already exists: $OUT"; exit 1
fi
echo "  ok: $OUT"

# ── peak VRAM logger ──
VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!
trap "kill $VRAM_PID 2>/dev/null" EXIT
export WHEAT_RUN_REPORT="$REPO/slurm_logs/run_report_${SLURM_JOB_ID}.txt"

echo ""
echo "================ gsplat  $PLOT  (train+render+metrics, 15k, default densify, pp OFF) ================"
python src/run_reconstruction.py \
  dataset=fip plot=$PLOT \
  run_train=true run_render=true run_metrics=true \
  reconstruction.iterations=15000 reconstruction.resolution=1 \
  reconstruction.absgrad=false reconstruction.use_principal_point=false \
  experiment_name=$EXP
rc=$?

echo ""
echo "================ STATUS ================"
if [ $rc -eq 0 ]; then echo "  $PLOT : OK"; else echo "  $PLOT : FAIL(rc=$rc)"; fi
echo "  -> results/reconstruction/fip/$PLOT/vanilla_3dgs/$EXP"
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
