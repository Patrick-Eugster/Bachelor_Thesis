#!/bin/bash -l
#SBATCH -J seg_cropcache_ab
#SBATCH --gpus=rtx_4090:1     # pin a 4090 (24 GB, ~3× faster than a Titan RTX for 3DGS/seg). Trade: longer queue wait.
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_cropcache_ab_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_cropcache_ab_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ------------------------------------------------------------------------------
# FIP plot_461 A/B for the segmentation_3d crop-cache speedup.
# Reuses the already-trained model (no retraining) and runs 3D seg TWICE on it:
#   (1) BASELINE  — WHEAT_SEG_NO_CACHE=1 forces the old decode-per-candidate path
#   (2) CROP CACHE — the new lossless mask crop cache (default)
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

echo "==================== (1) BASELINE (no crop cache) ===================="
date
WHEAT_SEG_NO_CACHE=1 WHEAT_SEG_TIMING=1 python src/run_reconstruction.py \
  plot=plot_461 experiment_name=$RECON_EXP \
  run_seg=true run_eval=true run_eval_2d=true \
  segmentation_3d.exp_name=seg_baseline

echo "==================== (2) CROP CACHE (new) ===================="
date
WHEAT_SEG_TIMING=1 python src/run_reconstruction.py \
  plot=plot_461 experiment_name=$RECON_EXP \
  run_seg=true run_eval=true run_eval_2d=true \
  segmentation_3d.exp_name=seg_cropcache

echo "==================== DONE — compare the two seg runs ===================="
date
SEG=results/reconstruction/fip/plot_461/vanilla_3dgs/$RECON_EXP/segmentation_3d
echo "--- baseline  seg_summary.json ---";  cat $SEG/seg_baseline/seg_summary.json  2>/dev/null
echo "--- cropcache seg_summary.json ---";  cat $SEG/seg_cropcache/seg_summary.json 2>/dev/null
echo "--- baseline  metrics_2d.json ---";   cat $SEG/seg_baseline/eval_2d/metrics_2d.json  2>/dev/null
echo "--- cropcache metrics_2d.json ---";   cat $SEG/seg_cropcache/eval_2d/metrics_2d.json 2>/dev/null
