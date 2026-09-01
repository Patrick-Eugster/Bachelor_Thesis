#!/bin/bash -l
# ============================================================================
# TEMPLATE — 3D segmentation (FlashSplat match-and-fine-tune) as a SLURM job.
#
# Generic example. Copy it, fill in the <PLACEHOLDERS>, and adjust the #SBATCH
# headers + module loads to your own cluster. Assumes a trained 3DGS model
# (example_train_job.sh) and a mask set (example_maskgen_job.sh) already exist.
#
# Pipeline stage 4 of 4 — see the repo README + configs/reconstruction_seg3d/segmentation_3d/.
# ============================================================================
#SBATCH -J wheat_seg
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G        # ~40 GB total; seg peak ~34 GB
#SBATCH --time=03:59:00
#SBATCH --output=<CLUSTER_PROJECT_PATH>/slurm_logs/wheat_seg_%j.out
#SBATCH --error=<CLUSTER_PROJECT_PATH>/slurm_logs/wheat_seg_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

# --- edit these ---
REPO="<CLUSTER_PROJECT_PATH>"
FIELD=field_A
DATE=20250715
MODEL_EXP=baseline              # experiment_name of the trained model to segment
SEG_EXP=seg_example             # NEW seg output name — unique per run
IOU=0.6                         # cross-view fine-tune match threshold (thesis uses 0.6)

cd "$REPO"

# --- environment: seg uses the TRAIN env (torch 2.1.2 + CUDA submodules) ---
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

# --- 3D seg + eval. Reuses the trained model + a mask set; writes a fresh seg subtree. ---
python src/run_reconstruction.py \
  dataset=phone plot=$FIELD date=$DATE \
  run_seg=true run_eval=true run_eval_2d=true \
  experiment_name=$MODEL_EXP \
  segmentation_3d.exp_name=$SEG_EXP \
  segmentation_3d.iou_threshold=$IOU

# Output: results/reconstruction/phone/$FIELD/$DATE/vanilla_3dgs/$MODEL_EXP/segmentation_3d/$SEG_EXP/
