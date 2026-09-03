#!/bin/bash -l
# ============================================================================
# TEMPLATE — stage 3, 3DGS reconstruction as a SLURM job (train, render, metrics).
#
# Runs the phone thesis config: 15000 iterations, resolution 1, default densification.
# Written for ETH Euler. Copy it, fill in the <PLACEHOLDERS>, and adjust the
# #SBATCH headers + module loads if you run somewhere else.
# ============================================================================
#SBATCH -J wheat_train
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:59:00
#SBATCH --output=<EULER_PROJECT_PATH>/slurm_logs/wheat_train_%j.out
#SBATCH --error=<EULER_PROJECT_PATH>/slurm_logs/wheat_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

# --- edit these ---
REPO="<EULER_PROJECT_PATH>"
FIELD=field_A                   # for FIP use profile=fip below, FIELD=plot_461 and an empty DATE
DATE=20250715
EXP=baseline                    # an existing name is refused

cd "$REPO"

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

python src/run_reconstruction.py \
  profile=phone plot=$FIELD date=$DATE \
  run_train=true run_render=true run_metrics=true \
  reconstruction.iterations=15000 reconstruction.resolution=1 \
  experiment_name=$EXP

echo "model -> $REPO/results/reconstruction/phone/$FIELD/$DATE/vanilla_3dgs/$EXP/"
