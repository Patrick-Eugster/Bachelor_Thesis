#!/bin/bash -l
# ============================================================================
# TEMPLATE — 3DGS reconstruction (train + render + metrics) as a SLURM job.
#
# This is a generic example. Copy it, fill in the <PLACEHOLDERS>, and adjust the
# #SBATCH headers + module loads to your own cluster. The authors' real per-run
# scripts (Euler-specific) are kept local-only and are not shipped here.
#
# Pipeline stage 3 of 4 — see the repo README + configs/reconstruction_seg3d/.
# ============================================================================
#SBATCH -J wheat_train
#SBATCH --gpus=1                # a single GPU with >=16 GB VRAM (24 GB recommended)
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=6G        # ~36 GB total; phone recon RAM peak ~28 GB
#SBATCH --time=08:00:00
#SBATCH --output=<CLUSTER_PROJECT_PATH>/slurm_logs/wheat_train_%j.out
#SBATCH --error=<CLUSTER_PROJECT_PATH>/slurm_logs/wheat_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

# --- edit these ---
REPO="<CLUSTER_PROJECT_PATH>"     # absolute path to your clone of this repo
FIELD=field_A                   # phone field folder (FIP: use plot=plot_461 and drop date=)
DATE=20250715                   # phone session; leave empty for FIP
EXP=baseline                    # experiment_name — must be unique per run (see repo "Experiment Naming")

cd "$REPO"

# --- environment (ETH Euler example — replace with your cluster's setup) ---
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs                                   # torch 2.1.2 + CUDA submodules + gsplat
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

# --- train + render + metrics (no seg). resolution 1, 15k iters, default densification. ---
python src/run_reconstruction.py \
  dataset=phone plot=$FIELD date=$DATE \
  run_train=true run_render=true run_metrics=true \
  reconstruction.iterations=15000 reconstruction.resolution=1 \
  experiment_name=$EXP

# Output: results/reconstruction/phone/$FIELD/$DATE/vanilla_3dgs/$EXP/
