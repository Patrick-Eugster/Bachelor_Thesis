#!/bin/bash -l
#SBATCH -J phone_seg_timers
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G      # 6 × 8G = 48 GB (crop-cache headroom, same as the other phone seg jobs)
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00       # phone seg ≈ 5.7h; 12h is ample
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_seg_timers_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_seg_timers_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ------------------------------------------------------------------------------
# Re-run the phone seg WITH the new fine-grained timers to split the ~2h "untimed".
# WHEAT_SEG_TIMING=1 now prints 11 buckets, incl. the 5 new ones that break down what used
# to be untimed CPU/IO:
#   setup        = one-time model+scene+camera load + crop-cache build/load + initial 2DSeg write
#   commit_paint = CPU side of the commit loop (alpha->CPU->threshold->paint the H×W 2DSeg map)
#   ply_prep     = per-head full-model CPU copy (feeds the async per-head PLY save)
#   overlay_wait = blocking on the async overlay-JPG saves (first vis_max_heads heads)
#   seg2d_save   = final 2DSeg/*.pt write
# Read the "seg time breakdown" block at the end of the '4. Segmentation' output in the .out.
#
# Cull OFF (the new default) + all other optimizations ON. NO use_principal_point (phone=false).
# The disk crop-cache from the earlier v3 run is reused, so `setup` reflects a steady-state run
# (cache LOAD, not build) — which is the normal case we want to characterize.
#
# Lossless sanity gate: the timers must NOT change the result -> md5 MUST stay aabb9b… .
# ------------------------------------------------------------------------------
# sbatch /cluster/project/cropsci/peugste/wheat3dgs/scripts/phone_seg_timers_job.sh

FIELD=${FIELD:-field_A}
DATE=${DATE:-20250715}
EXP=${EXP:-phone_sahi}
SEGEXP=${SEGEXP:-seg_timers_v4}
EXPECT_MD5=aabb9b504a2c07c9712722274d46ad88   # cull-off≡cull-on v3 md5 — instrumentation must not change it

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi
cd /cluster/project/cropsci/peugste/wheat3dgs
export MALLOC_ARENA_MAX=2

echo "==================== phone seg WITH new timers (cull off, all opts on) ===================="
date
WHEAT_SEG_TIMING=1 python src/run_reconstruction.py \
  dataset=phone plot=$FIELD date=$DATE experiment_name=$EXP \
  run_seg=true \
  segmentation_3d.detection_method=sahi_yolo_sam \
  segmentation_3d.frustum_cull=false \
  segmentation_3d.exp_name=$SEGEXP
date

echo "==================== lossless sanity gate (timers must not change the result) ===================="
SEG=results/reconstruction/phone/$FIELD/$DATE/vanilla_3dgs/$EXP/segmentation_3d
MD5=$(md5sum "$SEG/$SEGEXP/all_obj_labels.pth" 2>/dev/null | awk '{print $1}')
echo "md5      = ${MD5:-MISSING}"
echo "expected = $EXPECT_MD5"
if [ "$MD5" = "$EXPECT_MD5" ]; then
  echo "GATE: PASS ✅  instrumentation is lossless (result unchanged)"
else
  echo "GATE: FAIL ❌  md5 changed — the timers altered behaviour, investigate before trusting the split"
fi

echo "--- the 11-bucket 'seg time breakdown' is in the '4. Segmentation' output above (WHEAT_SEG_TIMING) ---"
echo "--- seg_summary.json ---"; cat $SEG/$SEGEXP/seg_summary.json 2>/dev/null
echo "NOTE: untimed remainder = wall-clock (SLURM Elapsed / date stamps) minus TOTAL(timed) = pure Python loop glue."
