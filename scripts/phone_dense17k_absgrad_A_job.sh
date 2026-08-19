#!/bin/bash -l
#SBATCH -J ph_d17k_abs_A
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=6
#SBATCH --time=03:59:00       # 2 sessions x ~1:21 (30k absgrad) ~2:42 -> fits the <4h short queue
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/ph_d17k_abs_A_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/ph_d17k_abs_A_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# dense17k opencv, ABSGRAD ON, sessions A (A/0627 + D/0627). See _phone_dense17k_body.sh.
SESSIONS=("field_A 20250627" "field_D 20250627")
EXP=dense17k
ABSGRAD=true
DGT=0.0008
source /cluster/project/cropsci/peugste/wheat3dgs/scripts/_phone_dense17k_body.sh
