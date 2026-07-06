#!/bin/bash -l
#SBATCH -J fip_cullval_c
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_cullval_c_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_cullval_c_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# Split C of the FIP seg-optimization losslessness validation: plots 465 + 466.
PLOTS="plot_465 plot_466"
source /cluster/project/cropsci/peugste/wheat3dgs/scripts/_fip_cull_validation_body.sh
