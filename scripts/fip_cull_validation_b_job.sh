#!/bin/bash -l
#SBATCH -J fip_cullval_b
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_cullval_b_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_cullval_b_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# Split B of the FIP seg-optimization losslessness validation: plots 463 + 464.
PLOTS="plot_463 plot_464"
source /cluster/project/cropsci/peugste/wheat3dgs/scripts/_fip_cull_validation_body.sh
