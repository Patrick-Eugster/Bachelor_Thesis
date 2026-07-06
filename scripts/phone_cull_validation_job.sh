#!/bin/bash -l
#SBATCH -J phone_cull_val
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G      # 6 × 8G = 48 GB (crop-cache build headroom, same as the cull-on run)
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00       # cull-off ≈ cull-on speed (~6h) since the cull barely culls on phone
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_cull_val_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_cull_val_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ------------------------------------------------------------------------------
# CLEAN phone cull A/B — the decisive test the old seg_cropcache comparison couldn't be.
#
# This runs cull OFF but EVERYTHING ELSE ON (crop cache + disk cache + lift-from-crop +
# inference=True), on the SAME model + SAME seg_seed(0) as the existing seg_cull_v3 (cull ON).
# The ONLY difference vs seg_cull_v3 is --no_frustum_cull, so:
#
#   seg_nocull_v3 == seg_cull_v3 (aabb9b…)  => the cull is LOSSLESS on phone. And since BOTH
#     differ from the old no-cull reference seg_cropcache (c515f119…, the v0 code), that proves
#     the c515f119-vs-aabb9b gap was the CONFOUNDED v0 baseline (different code/order), NOT the cull.
#   seg_nocull_v3 != seg_cull_v3            => the cull DOES change the phone result (then bisect).
#
# WHY we expect identical: the render probe (src/analysis/probe_cull_conservativeness.py) showed
# the cull keeps 83/83 cameras for 119 of 120 phone heads — the close-orbit capture puts every head
# in every view, so there's almost nothing to cull. So cull-off renders the same cameras => same result.
#
# NO use_principal_point: phone was trained pp=false, so seg renders pp=false (config default = pass nothing).
# Disk crop-cache from seg_cull_v3 is reused (manifest-validated) so the build is skipped.
# ------------------------------------------------------------------------------
# sbatch /cluster/project/cropsci/peugste/wheat3dgs/scripts/phone_cull_validation_job.sh

FIELD=${FIELD:-field_A}
DATE=${DATE:-20250715}
EXP=${EXP:-phone_sahi}
CULLREF=${CULLREF:-seg_cull_v3}         # existing cull-ON v3 run (aabb9b) to compare against
V0REF=${V0REF:-seg_cropcache}           # old v0 no-cull run (c515f119) — the confounded baseline
SEGEXP=${SEGEXP:-seg_nocull_v3}         # this run: cull OFF, everything else ON
EXPECT_MD5=aabb9b504a2c07c9712722274d46ad88   # seg_cull_v3's md5 — target if the cull is lossless

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi
cd /cluster/project/cropsci/peugste/wheat3dgs
export MALLOC_ARENA_MAX=2

echo "==================== CULL OFF, everything else ON (phone) ===================="
date
# --no_frustum_cull is the ONLY change vs seg_cull_v3. Cache on (default) => lift-from-crop + disk cache.
# inference=True is internal + always on. NO pp flag (phone=false).
WHEAT_SEG_TIMING=1 python src/run_reconstruction.py \
  dataset=phone plot=$FIELD date=$DATE experiment_name=$EXP \
  run_seg=true \
  segmentation_3d.detection_method=sahi_yolo_sam \
  segmentation_3d.frustum_cull=false \
  segmentation_3d.use_mask_cache=true \
  segmentation_3d.exp_name=$SEGEXP
date

echo "==================== GATE — cull-off vs cull-on (must be IDENTICAL if cull is lossless) ===================="
SEG=results/reconstruction/phone/$FIELD/$DATE/vanilla_3dgs/$EXP/segmentation_3d

# primary: does cull-off == cull-on (both v3 code)?  -> the real cull losslessness test
python src/analysis/compare_seg_runs.py "$SEG/$SEGEXP" "$SEG/$CULLREF"
RC=$?

echo "--- md5 summary ---"
echo -n "nocull_v3 (this) : "; md5sum "$SEG/$SEGEXP/all_obj_labels.pth" 2>/dev/null | awk '{print $1}'
echo -n "cull_v3          : "; md5sum "$SEG/$CULLREF/all_obj_labels.pth" 2>/dev/null | awk '{print $1}'
echo -n "cropcache (v0)   : "; md5sum "$SEG/$V0REF/all_obj_labels.pth" 2>/dev/null | awk '{print $1}'

if [ "$RC" = "0" ]; then
  echo "GATE: PASS ✅  cull-off == cull-on (v3) => FRUSTUM CULL IS LOSSLESS ON PHONE."
  echo "  => the earlier aabb9b-vs-c515f119 gap was the confounded v0 baseline (code/order), not the cull."
else
  echo "GATE: FAIL ❌  cull-off != cull-on (v3) => the cull DOES change the phone result. Bisect next."
fi

echo "--- nocull seg_summary.json ---"; cat $SEG/$SEGEXP/seg_summary.json 2>/dev/null
echo "SPEED: cull-off wall-clock should ≈ seg_cull_v3's 5h56m (the cull skips ~nothing on phone)."
