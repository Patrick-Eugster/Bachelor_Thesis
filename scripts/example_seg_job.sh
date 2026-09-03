#!/bin/bash -l
# ============================================================================
# TEMPLATE — stage 4, 3D segmentation (FlashSplat match-and-fine-tune) as a SLURM job.
#
# Needs a trained model (example_train_job.sh) and a mask set (example_maskgen_job.sh).
# Written for ETH Euler. Copy it, fill in the <PLACEHOLDERS>, and adjust the
# #SBATCH headers + module loads if you run somewhere else.
# ============================================================================
#SBATCH -J wheat_seg
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:59:00
#SBATCH --output=<EULER_PROJECT_PATH>/slurm_logs/wheat_seg_%j.out
#SBATCH --error=<EULER_PROJECT_PATH>/slurm_logs/wheat_seg_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

# --- edit these ---
REPO="<EULER_PROJECT_PATH>"
FIELD=field_A
DATE=20250715
MODEL_EXP=baseline              # the trained model to segment
MASK_EXP=perhead_sam2_conf070   # the mask set to read
SEG_EXP=seg_run1                # an existing name is overwritten
IOU=0.6                         # fine-tune matching threshold

cd "$REPO"

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

# roi_cull and marker_exclude need logs/marker_points3d.json from preprocessing and do nothing without it.
python src/run_reconstruction.py \
  profile=phone plot=$FIELD date=$DATE \
  run_seg=true run_eval=true run_eval_2d=true \
  experiment_name=$MODEL_EXP \
  segmentation_3d.mask_gen_experiment=$MASK_EXP \
  segmentation_3d.exp_name=$SEG_EXP \
  segmentation_3d.iou_threshold=$IOU \
  segmentation_3d.roi_cull=true \
  segmentation_3d.marker_exclude=true

echo "segmentation -> $REPO/results/reconstruction/phone/$FIELD/$DATE/vanilla_3dgs/$MODEL_EXP/segmentation_3d/$SEG_EXP/"
