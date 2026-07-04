#!/bin/bash -l
#SBATCH -J seg_cropcache_ab
#SBATCH --gpus=rtx_4090:1     # pin a 4090 (24 GB, ~3× faster than a Titan RTX for 3DGS/seg). Trade: longer queue wait.
#SBATCH --mem-per-cpu=6G      # 6 × 6G = 36 GB (small bump from 30 for cache-build headroom; the
#SBATCH --cpus-per-task=6     #   Euler-specific build leak is still a real bug to fix — see CROP_CACHE_OOM_AND_IOU_DEBUG.md)
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_cropcache_ab_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_cropcache_ab_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ------------------------------------------------------------------------------
# FIP plot_461 A/B for the segmentation_3d crop-cache speedup.
# Reuses the already-trained model (no retraining) and runs 3D seg TWICE on it:
#   (1) BASELINE  — segmentation_3d.use_mask_cache=false → old decode-per-candidate path
#   (2) CROP CACHE — the new lossless mask crop cache (default use_mask_cache=true)
# Both with WHEAT_SEG_TIMING=1 so each prints its render-vs-match time split.
# Afterwards compare the two seg folders' seg_summary.json (wheat_heads_found) and
# eval_2d/metrics_2d.json — they MUST be identical (only the runtime should change).
# See docs/segmentation_3d/SEGMENTATION_3D_RUNTIME.md.
# ------------------------------------------------------------------------------

RECON_EXP=test_absgrad_v2   # existing trained model folder under vanilla_3dgs/ (reused, not retrained)

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi
cd /cluster/project/cropsci/peugste/wheat3dgs

# Problem A guard: the crop-cache BUILD decodes many 12MB full-frame masks across worker threads.
# glibc keeps freed chunks in per-thread malloc arenas, which can bloat RSS on Euler (flat locally).
# Cap arenas so freed mask memory is actually reused instead of accumulating. Harmless if not needed.
# (If the build STILL climbs, the real culprit is a stale __pycache__ shadowing the numpy-.copy() fix —
#  clear it: find src -name __pycache__ -type d -exec rm -rf {} +. See CROP_CACHE_OOM_AND_IOU_DEBUG.md.)
export MALLOC_ARENA_MAX=2

# CRITICAL: test_absgrad_v2 was TRAINED with use_principal_point=true (asymmetric frustum, the
# Round-4 pixel-shift fix). The seg MUST render with the same setting or every blob shifts ~2-12px
# and cross-view IoU collapses (matches 15->3, IoU 0.565->0.117). The config default is false, so we
# MUST pass reconstruction.use_principal_point=true here to match how the model was trained.
# This was the whole "IoU regression" — a train/seg projection mismatch, NOT the env or masks.
# See docs/segmentation_3d/CROP_CACHE_OOM_AND_IOU_DEBUG.md.
PP=reconstruction.use_principal_point=true

# Skip the baseline half by submitting with --export=ALL,RUN_BASELINE=0 — e.g. it already completed
# in a prior job (seg_baseline_v3 is saved) and you only need to (re)run the crop-cache half after a
# build fix. The compare block below still reads the saved seg_baseline_v3 for the A/B.
RUN_BASELINE=${RUN_BASELINE:-1}

if [ "$RUN_BASELINE" = "1" ]; then
echo "==================== (1) BASELINE (no crop cache) ===================="
date
# cache OFF via the config knob (run_reconstruction -> run_3d_seg --no_mask_cache); the
# WHEAT_SEG_NO_CACHE=1 env override does the same and is kept as a fallback, but the flag is cleaner.
WHEAT_SEG_TIMING=1 python src/run_reconstruction.py \
  plot=plot_461 experiment_name=$RECON_EXP $PP \
  run_seg=true run_eval=true run_eval_2d=true \
  segmentation_3d.use_mask_cache=false \
  segmentation_3d.exp_name=seg_baseline_v3
else
echo "==================== (1) BASELINE SKIPPED (RUN_BASELINE=0) — reusing saved seg_baseline_v3 ===================="
fi

echo "==================== (2) CROP CACHE (new) ===================="
date
WHEAT_SEG_TIMING=1 python src/run_reconstruction.py \
  plot=plot_461 experiment_name=$RECON_EXP $PP \
  run_seg=true run_eval=true run_eval_2d=true \
  segmentation_3d.exp_name=seg_cropcache_v3

echo "==================== DONE — compare the two seg runs ===================="
date
SEG=results/reconstruction/fip/plot_461/vanilla_3dgs/$RECON_EXP/segmentation_3d
echo "--- baseline  seg_summary.json ---";  cat $SEG/seg_baseline_v3/seg_summary.json  2>/dev/null
echo "--- cropcache seg_summary.json ---";  cat $SEG/seg_cropcache_v3/seg_summary.json 2>/dev/null
echo "--- baseline  metrics_2d.json ---";   cat $SEG/seg_baseline_v3/eval_2d/metrics_2d.json  2>/dev/null
echo "--- cropcache metrics_2d.json ---";   cat $SEG/seg_cropcache_v3/eval_2d/metrics_2d.json 2>/dev/null
echo "EXPECT: both back to IoU ~0.565 (pp fix), num_matches ~15, and IDENTICAL to each other (cache lossless)"
