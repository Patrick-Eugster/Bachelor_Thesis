#!/bin/bash -l
#SBATCH -J ph_d17k_noabs_A
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=6
#SBATCH --time=03:59:00       # 2 sessions x ~1:00 (30k default densify, fewer Gaussians) ~2:00
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/ph_d17k_noabs_A_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/ph_d17k_noabs_A_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# dense17k opencv, ABSGRAD OFF (default signed-grad densify), sessions A (A/0627 + D/0627).
SESSIONS=("field_A 20250627" "field_D 20250627")
EXP=dense17k_noabsgrad
ABSGRAD=false
DGT=0.0002
source /cluster/project/cropsci/peugste/wheat3dgs/scripts/_phone_dense17k_body.sh
