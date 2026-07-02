#!/bin/bash -l
#SBATCH -J phone_seg_cropcache
#SBATCH --gpus=rtx_4090:1     # pin a 4090 (24 GB, ~3× faster than a Titan RTX for 3DGS/seg) — seg is the long pole. Trade: longer queue wait.
#SBATCH --mem-per-cpu=8G      # 8 × 8G = 64 GB — the ~46 GB seg baseline (model + 96 full-res phone
#SBATCH --cpus-per-task=8     #   images in CPU RAM) sat right at the old 48 GB limit; give headroom
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_seg_cropcache_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_seg_cropcache_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ------------------------------------------------------------------------------
# Phone field_A/20250715 — the actual payoff run: 3D seg on the dense SAHI masks
# WITH the crop cache. Reuses the model already trained on Euler (no retraining).
# This is the run that was stuck at "6% after 48h" before the crop cache.
# No baseline here — the baseline IS that too-slow run; correctness was already
# proven bit-identical offline (src/analysis/verify_crop_iou.py) and on the FIP A/B.
# No eval_2d: phone has no ground-truth masks. WHEAT_SEG_TIMING=1 prints the split.
# ------------------------------------------------------------------------------

# Reuses the model trained by phone_full_job.sh (its default experiment_name=phone_sahi).
# Override the session/model without editing this file, same convention as phone_full_job.sh:
#   sbatch --export=FIELD=field_D,DATE=20250603,EXP=phone_sahi /cluster/.../scripts/phone_seg_cropcache_job.sh
FIELD=${FIELD:-field_A}
DATE=${DATE:-20250715}
EXP=${EXP:-phone_sahi}

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi
cd /cluster/project/cropsci/peugste/wheat3dgs

date
WHEAT_SEG_TIMING=1 python src/run_reconstruction.py \
  dataset=phone plot=$FIELD date=$DATE experiment_name=$EXP \
  run_seg=true \
  segmentation_3d.detection_method=sahi_yolo_sam \
  segmentation_3d.exp_name=seg_cropcache
date

SEG=results/reconstruction/phone/$FIELD/$DATE/vanilla_3dgs/$EXP/segmentation_3d
echo "--- cropcache seg_summary.json ---"; cat $SEG/seg_cropcache/seg_summary.json 2>/dev/null
