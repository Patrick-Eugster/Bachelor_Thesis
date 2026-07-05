#!/bin/bash -l
#SBATCH -J seg_frustumcull_ab
#SBATCH --gpus=rtx_4090:1     # pin a 4090 (24 GB, ~3× faster than a Titan RTX for 3DGS/seg). Trade: longer queue wait.
#SBATCH --mem-per-cpu=6G      # 6 × 6G = 36 GB. Crop cache is ON here (default) so the build still runs; this
#SBATCH --cpus-per-task=6     #   is the same headroom that let seg_cropcache_v3 finish. Frustum cull adds no RAM.
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_frustumcull_ab_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_frustumcull_ab_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ------------------------------------------------------------------------------
# FIP plot_461 A/B for the segmentation_3d FRUSTUM CULL speedup (idea ① + the
# inference=True render flag ④). Reuses the already-trained model (no retraining).
#
# The frustum cull is designed to be BIT-IDENTICAL: it only skips rendering a head
# into cameras its bounding sphere misses, where the alpha is provably empty. So the
# gate is not "similar metrics" — it's the SAME all_obj_labels.pth md5 as the existing
# no-cull, cache-on reference seg_cropcache_v3 (md5 0bcd708…, IoU 0.564375, 514 heads).
# That single check validates BOTH the frustum cull AND the inference=True change at once.
#
# DEFAULT: run ONLY the treatment (cull ON) and diff it against the saved seg_cropcache_v3.
# Set RUN_BASELINE=1 (--export=ALL,RUN_BASELINE=1) to ALSO run a fresh no-cull baseline
# (seg_nocull_v1, --no_frustum_cull) in this same job for a fully self-contained A/B.
# Both runs use WHEAT_SEG_TIMING=1 so each prints its render-vs-match time split.
# See docs/segmentation_3d/SEGMENTATION_3D_RUNTIME.md §8.
# ------------------------------------------------------------------------------
# sbatch --export=ALL,RUN_BASELINE=1 /cluster/project/cropsci/peugste/wheat3dgs/scripts/fip_seg_frustumcull_ab_job.sh

RECON_EXP=test_absgrad_v2   # existing trained model folder under vanilla_3dgs/ (reused, not retrained)
REF=seg_cropcache_v3        # the known-good no-cull, cache-on reference already on disk
EXPECT_MD5=0bcd708dfe026d1a4ecd2f3f0d68c386   # md5 of REF's all_obj_labels.pth — the exactness gate
SEGEXP=${SEGEXP:-seg_cull_v3}   # treatment output folder; bump per code version (v2 = + lift inference + 6-bucket timers)

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi
cd /cluster/project/cropsci/peugste/wheat3dgs

# Crop-cache build RAM guard (cache is ON here): cap glibc malloc arenas so freed 12MB mask
# frames are reused instead of bloating RSS across worker threads. Harmless if not needed.
# (If the build climbs anyway, suspect a stale __pycache__ shadowing the numpy fix:
#  find src -name __pycache__ -type d -exec rm -rf {} +. See CROP_CACHE_OOM_AND_IOU_DEBUG.md.)
export MALLOC_ARENA_MAX=2

# CRITICAL (same footgun as the crop-cache A/B): test_absgrad_v2 was TRAINED with
# use_principal_point=true. The seg MUST render with the SAME setting or every blob shifts and
# cross-view IoU collapses (0.565 -> 0.117). The config default is false, so we MUST pass it here.
PP=reconstruction.use_principal_point=true

RUN_BASELINE=${RUN_BASELINE:-0}

if [ "$RUN_BASELINE" = "1" ]; then
echo "==================== (opt) FRESH BASELINE — cull OFF (--no_frustum_cull) ===================="
date
WHEAT_SEG_TIMING=1 python src/run_reconstruction.py \
  plot=plot_461 experiment_name=$RECON_EXP $PP \
  run_seg=true run_eval=true run_eval_2d=true \
  segmentation_3d.frustum_cull=false \
  segmentation_3d.exp_name=seg_nocull_v1
fi

echo "==================== TREATMENT — frustum cull ON ===================="
date
# frustum_cull=true is the config default; pass it explicitly so a config drift can't silently
# turn it off. Crop cache stays on (default). inference=True is internal + always on.
WHEAT_SEG_TIMING=1 python src/run_reconstruction.py \
  plot=plot_461 experiment_name=$RECON_EXP $PP \
  run_seg=true run_eval=true run_eval_2d=true \
  segmentation_3d.frustum_cull=true \
  segmentation_3d.exp_name=$SEGEXP

echo "==================== DONE — exactness gate ===================="
date
SEG=results/reconstruction/fip/plot_461/vanilla_3dgs/$RECON_EXP/segmentation_3d
CULL_PTH=$SEG/$SEGEXP/all_obj_labels.pth
REF_PTH=$SEG/$REF/all_obj_labels.pth

echo "--- md5 all_obj_labels.pth ---"
md5sum "$CULL_PTH" 2>/dev/null || echo "MISSING: $CULL_PTH (treatment seg did not finish)"
if [ -f "$REF_PTH" ]; then md5sum "$REF_PTH"; else echo "MISSING reference $REF_PTH — set RUN_BASELINE=1 for a self-contained A/B"; fi

CULL_MD5=$(md5sum "$CULL_PTH" 2>/dev/null | awk '{print $1}')
if [ "$CULL_MD5" = "$EXPECT_MD5" ]; then
  echo "GATE: PASS ✅  frustum cull is bit-identical to the reference (md5 $EXPECT_MD5)"
else
  echo "GATE: FAIL ❌  cull md5 $CULL_MD5 != expected $EXPECT_MD5"
  echo "  -> the 3-sigma sphere margin may be too tight, or inference=True changed pixels. Do NOT ship."
fi

echo "--- cull   seg_summary.json ---";  cat $SEG/$SEGEXP/seg_summary.json 2>/dev/null
echo "--- cull   metrics_2d.json  ---";  cat $SEG/$SEGEXP/eval_2d/metrics_2d.json 2>/dev/null
if [ "$RUN_BASELINE" = "1" ]; then
  echo "--- nocull seg_summary.json ---";  cat $SEG/seg_nocull_v1/seg_summary.json 2>/dev/null
  echo "--- nocull md5 (should also equal $EXPECT_MD5) ---"; md5sum $SEG/seg_nocull_v1/all_obj_labels.pth 2>/dev/null
fi
echo "EXPECT: cull md5 == $EXPECT_MD5, IoU 0.564375, 514 heads. Timing = render share should drop sharply."
