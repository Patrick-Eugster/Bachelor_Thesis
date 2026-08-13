#!/bin/bash -l
#SBATCH -J fip_reproc_ppdemo_diffgs
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G      # 36 GB total — FIP 36 imgs -> RAM ~10-15 GB (fewer imgs than phone), ample headroom
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00       # 7 plots x ~47 min (diffgs 1.77x slower: 15k train ~41 + render + metrics) ~= 5.5h; 8h margin
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_reproc_ppdemo_diffgs_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_reproc_ppdemo_diffgs_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# FIP RE-CENTERED PP DEMONSTRATION (diffgs). Same as the gsplat job but forces the OLD diff-gaussian engine
# (export WHEAT_RENDERER=diffgs). WHY diffgs: the original +7.8 dB pixel-shift result (baseline 20.37 ->
# pp-on 28.17) was measured on diffgs, so diffgs is the ENGINE-MATCHED comparison. If diffgs pp-OFF on the
# RE-CENTERED data lands near the original pp-ON number (~28) instead of the buggy pp-OFF baseline (~20), it
# CORROBORATES that the pixel shift was the cause (cure #2 = re-center the data == cure #1 = the flag).
# CAVEAT: reprocessed is a DIFFERENT SfM (poses/focal/undistortion) + this is 15k not the original 30k, so
# the PSNR jump is CORROBORATING, not a clean isolation — the confound-free proof is the render-vs-GT SHIFT
# (~0 px on centered data), measured offline (src/analysis/make_pp_shift_figure.py).
# recon-only: train + render + metrics (NO seg). diffgs, resolution 1, 15000 iters, DEFAULT densify, pp OFF.
# Writes results/reconstruction/fip/<plot>/reprocessed_png/vanilla_3dgs/recentered_ppoff_diffgs (own subtree).
# ============================================================================

export WHEAT_RENDERER=diffgs   # <-- forces the old diff-gaussian render path (render() routes to render_diffgs())

VARIANT=reprocessed_png
EXP=recentered_ppoff_diffgs
PLOTS=(plot_461 plot_462 plot_463 plot_464 plot_465 plot_466 plot_467)

REPO=/cluster/project/cropsci/peugste/wheat3dgs
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
cd "$REPO"
nvidia-smi

# ── PREFLIGHT ──
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

VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!
trap "kill $VRAM_PID 2>/dev/null" EXIT
export WHEAT_RUN_REPORT="$REPO/slurm_logs/run_report_${SLURM_JOB_ID}.txt"

declare -A STATUS
for P in "${PLOTS[@]}"; do
  echo ""
  echo "================ diffgs  $P  (train+render+metrics, 15k, default densify, pp OFF) ================"
  python src/run_reconstruction.py \
    dataset=fip plot=$P \
    run_train=true run_render=true run_metrics=true \
    reconstruction.iterations=15000 reconstruction.resolution=1 reconstruction.absgrad=false \
    experiment_name=$EXP sfm_variant=$VARIANT
  rc=$?
  if [ $rc -eq 0 ]; then STATUS["$P"]=OK; else STATUS["$P"]="FAIL(rc=$rc)"; fi
done

echo ""
echo "================ diffgs reprocessed_png PER-PLOT STATUS ================"
for P in "${PLOTS[@]}"; do
  echo "  $P : ${STATUS[$P]}  -> results/reconstruction/fip/$P/$VARIANT/vanilla_3dgs/$EXP"
done
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
