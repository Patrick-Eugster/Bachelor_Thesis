#!/bin/bash -l
#SBATCH -J fip461_res_ab
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=4G      # 24 GB total — one FIP plot (36 imgs) uses ~10-15 GB RAM; kept small to
                              # ease the QOSMaxMemoryPerUser cap while the 3D-seg jobs are also running
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00       # two runs (res1 + res2) x (15k train + render + metrics ~25-30 min) -> 2h ceiling
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip461_res_ab_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip461_res_ab_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# FIP plot_461 — resolution ablation off the DECIDED base config.
# Base config = gsplat + principal-point ON + AbsGS (densify_grad_threshold 0.0008) + 15k, sh_degree 3.
# We run it TWICE, flipping ONLY the resolution: res 1 (full) vs res 2 (half). The res-1 run is the fair
# reference (no clean res1/15k/AbsGS/pp-on run for 461 exists on disk yet), the res-2 run is the ablation.
# recon-only: train + render + metrics (NO seg). Own experiment folders so nothing collides.
# NOTE: metrics at res 2 are computed against the 2x-downscaled GT, so PSNR/SSIM/LPIPS are NOT directly
# comparable across the two runs in absolute terms — read this as a detail-vs-cost (time/VRAM) tradeoff.
# Reads   input_plots/fip/plot_461/{images,sparse/0}   (default sfm_variant — original data).
# Writes  results/reconstruction/fip/plot_461/vanilla_3dgs/{res1_absgrad_15k, res2_absgrad_15k}.
# ============================================================================

PLOT=plot_461
DGT=0.0008                       # AbsGS needs the raised densify threshold (default 0.0002 over-densifies)

REPO=/cluster/project/cropsci/peugste/wheat3dgs
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
cd "$REPO"
export WHEAT_RENDERER=gsplat     # explicit (gsplat is already the default)
nvidia-smi

# ── PREFLIGHT: abort if either target already exists ──
echo "=== PREFLIGHT ==="
FAIL=0
for EXP in res1_absgrad_15k res2_absgrad_15k; do
  OUT="$REPO/results/reconstruction/fip/$PLOT/vanilla_3dgs/$EXP"
  if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "  X target already exists: $OUT"; FAIL=1
  fi
done
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: both targets fresh"

# ── peak VRAM logger ──
VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!
trap "kill $VRAM_PID 2>/dev/null" EXIT
export WHEAT_RUN_REPORT="$REPO/slurm_logs/run_report_${SLURM_JOB_ID}.txt"

run_one () {
  local RES=$1 EXP=$2
  echo ""
  echo "================ gsplat  $PLOT  res=$RES  (train+render+metrics, 15k, AbsGS, pp ON) ================"
  python src/run_reconstruction.py \
    dataset=fip plot=$PLOT \
    run_train=true run_render=true run_metrics=true \
    reconstruction.iterations=15000 reconstruction.resolution=$RES \
    reconstruction.absgrad=true reconstruction.densify_grad_threshold=$DGT \
    reconstruction.use_principal_point=true \
    experiment_name=$EXP
  echo "  res=$RES rc=$? -> results/reconstruction/fip/$PLOT/vanilla_3dgs/$EXP"
}

run_one 1 res1_absgrad_15k
run_one 2 res2_absgrad_15k

echo ""
echo "================ STATUS ================"
echo "  res1 -> results/reconstruction/fip/$PLOT/vanilla_3dgs/res1_absgrad_15k"
echo "  res2 -> results/reconstruction/fip/$PLOT/vanilla_3dgs/res2_absgrad_15k"
echo "Peak VRAM (both runs): $(sort -n "$VRAM_LOG" | tail -1) MiB"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
