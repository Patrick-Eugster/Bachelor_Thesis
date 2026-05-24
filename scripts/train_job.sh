#!/bin/bash -l
#SBATCH -J wheat_train
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --mem-per-cpu=8G      # 32 GB total (4 × 8G) — enough for data_device_cpu with 36 images
#SBATCH --cpus-per-task=4     # CPU feeds data, GPU does the work — 4 is sufficient for training
#SBATCH --time=6:00:00        # training ~1.5h, 6h gives comfortable buffer
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/train_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/train_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# Module load order: purge → source conda → activate env → module load
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
which python
python --version
conda info --envs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi  # prints GPU model, VRAM, driver version to the log

cd /cluster/project/cropsci/peugste/wheat3dgs
python src/run_reconstruction.py run_train=true reconstruction.iterations=30000
