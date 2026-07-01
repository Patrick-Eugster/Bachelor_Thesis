#!/bin/bash -l
#SBATCH -J phone_full
#SBATCH --gpus=rtx_4090:1     # pin a 4090 (24 GB, ~3× faster than a Titan RTX for 3DGS/seg) — seg is the long pole. Trade: longer queue wait.
#SBATCH --mem-per-cpu=8G      # 48 GB total (6 × 8G) — phone ~96 full-res images + SAM masks in CPU RAM
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00       # train 15k ~1-2h + seg over ~25k phone heads can be long → 48 h buffer
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_full_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_full_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# Full phone pipeline for ONE session: 3DGS train -> render -> metrics -> 3D segmentation.
# Reads the SAHI masks (detection_method=sahi_yolo_sam). Runs in the existing torch-2.1.2 env.
# Override the session without editing this file:
#   sbatch --export=FIELD=field_D,DATE=20250603,EXP=phone_sahi /cluster/.../scripts/phone_full_job.sh
# Plain `sbatch <path>` uses the defaults below (field_A/20250715).

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
which python
python --version
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi

FIELD=${FIELD:-field_A}
DATE=${DATE:-20250715}
EXP=${EXP:-phone_sahi}

cd /cluster/project/cropsci/peugste/wheat3dgs

echo ""
echo "========================================"
echo "  PHONE FULL PIPELINE: $FIELD / $DATE  (exp=$EXP, SAHI masks)"
echo "========================================"

python src/run_reconstruction.py \
  dataset=phone plot=$FIELD date=$DATE \
  run_train=true run_render=true run_metrics=true run_seg=true \
  segmentation_3d.detection_method=sahi_yolo_sam \
  experiment_name=$EXP
