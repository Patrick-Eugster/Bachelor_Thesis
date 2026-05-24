#!/bin/bash -l
#SBATCH -J wheat_seg
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi

cd /cluster/project/cropsci/peugste/wheat3dgs
python src/run_reconstruction.py run_seg=true experiment_name=$RECON_EXP segmentation_3d.exp_name=$SEG_EXP