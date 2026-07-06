#!/bin/bash -l
#SBATCH -J fip_cullval_a
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_cullval_a_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_cullval_a_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# Split A of the FIP seg-optimization losslessness validation: plots 461 + 462.
# Per plot: clean gsplat+AbsGrad train -> seg baseline (no cache,no cull) -> seg optimized
# -> compare_seg_runs.py gate. See scripts/_fip_cull_validation_body.sh for the full logic.
PLOTS="plot_461 plot_462"
source /cluster/project/cropsci/peugste/wheat3dgs/scripts/_fip_cull_validation_body.sh
