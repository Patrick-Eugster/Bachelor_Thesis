#!/bin/bash -l
#SBATCH -J phone_seg_frustumcull
#SBATCH --gpus=rtx_4090:1     # pin a 4090 (24 GB, ~3× faster than a Titan RTX for 3DGS/seg) — seg is the long pole. Trade: longer queue wait.
#SBATCH --mem-per-cpu=8G      # 6 × 8G = 48 GB — the finished no-cull run peaked 38.1 GB (post leak-fix),
#SBATCH --cpus-per-task=6     #   so ~10 GB headroom; crop-cache build fits, frustum cull adds no RAM.
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_seg_frustumcull_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_seg_frustumcull_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ------------------------------------------------------------------------------
# Phone field_A/20250715 — the HEADLINE frustum-cull measurement. This is the plot
# where the cull actually pays off: 96 cameras sweeping along rows, so each head is
# only in a LOCAL subset of views → the sphere skips ~80 of 96 cameras per head.
# Reuses the model already trained on Euler (no retraining), on the dense SAHI masks.
#
# The no-cull reference is the finished crop-cache run seg_cropcache (18h38m, cache on,
# cull off) — same model, same masks. So this run isolates the cull + inference=True win:
#   - GATE (exactness): seg_cull_v1 all_obj_labels.pth md5 MUST == seg_cropcache's
#     c515f1196206b6a090a944e1e01116dc  (bit-identical, same as the FIP A/B proved).
#   - SPEED: compare this run's wall-clock to the reference's 18h38m
#     (its find_match split was render 91% / match 9%).
# No pp flag: phone was TRAINED with use_principal_point=false (principal point ≈ image
# centre), so seg must also render pp=false — i.e. pass NOTHING (the config default).
# No eval_2d: phone has no ground-truth masks. WHEAT_SEG_TIMING=1 prints the split.
# ------------------------------------------------------------------------------
# sbatch /cluster/project/cropsci/peugste/wheat3dgs/scripts/phone_seg_frustumcull_job.sh

FIELD=${FIELD:-field_A}
DATE=${DATE:-20250715}
EXP=${EXP:-phone_sahi}
REF=${REF:-seg_cropcache}   # the finished no-cull, cache-on reference already on disk
EXPECT_MD5=c515f1196206b6a090a944e1e01116dc   # md5 of REF's all_obj_labels.pth — the exactness gate
SEGEXP=${SEGEXP:-seg_cull_v3}   # output folder; bump per code version (v2 = + lift inference + 6-bucket timers)

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi
cd /cluster/project/cropsci/peugste/wheat3dgs

# Crop-cache build RAM guard (cache is ON here): cap glibc malloc arenas so freed 12MB mask
# frames are reused instead of bloating RSS across worker threads. Harmless if not needed.
export MALLOC_ARENA_MAX=2

echo "==================== TREATMENT — frustum cull ON (phone) ===================="
date
# frustum_cull=true is the config default; pass explicitly so a config drift can't silently disable it.
# Crop cache stays on (default). inference=True is internal + always on. NO use_principal_point (phone=false).
WHEAT_SEG_TIMING=1 python src/run_reconstruction.py \
  dataset=phone plot=$FIELD date=$DATE experiment_name=$EXP \
  run_seg=true \
  segmentation_3d.detection_method=sahi_yolo_sam \
  segmentation_3d.frustum_cull=true \
  segmentation_3d.exp_name=$SEGEXP
date

echo "==================== DONE — exactness gate ===================="
SEG=results/reconstruction/phone/$FIELD/$DATE/vanilla_3dgs/$EXP/segmentation_3d
CULL_PTH=$SEG/$SEGEXP/all_obj_labels.pth
REF_PTH=$SEG/$REF/all_obj_labels.pth

echo "--- md5 all_obj_labels.pth ---"
md5sum "$CULL_PTH" 2>/dev/null || echo "MISSING: $CULL_PTH (seg did not finish)"
if [ -f "$REF_PTH" ]; then md5sum "$REF_PTH"; else echo "MISSING reference $REF_PTH"; fi

CULL_MD5=$(md5sum "$CULL_PTH" 2>/dev/null | awk '{print $1}')
if [ "$CULL_MD5" = "$EXPECT_MD5" ]; then
  echo "GATE: PASS ✅  frustum cull is bit-identical to the no-cull phone reference (md5 $EXPECT_MD5)"
else
  echo "GATE: FAIL ❌  cull md5 $CULL_MD5 != expected $EXPECT_MD5"
  echo "  -> the 3-sigma sphere margin may be too tight, or inference=True changed pixels. Do NOT ship."
fi

echo "--- cull seg_summary.json ---"; cat $SEG/$SEGEXP/seg_summary.json 2>/dev/null
echo "SPEED: compare this run's wall-clock (date stamps above / SLURM Elapsed) to the reference's 18h38m."
