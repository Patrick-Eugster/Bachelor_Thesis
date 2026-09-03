#!/bin/bash -l
# ============================================================================
# TEMPLATE — stage 2, mask generation (YOLOv5 boxes + SAM masks) as a SLURM job.
#
# Runs the phone thesis config: full-res YOLOv5, per_head SAM2, confidence 0.70.
# Written for ETH Euler. Copy it, fill in the <PLACEHOLDERS>, and adjust the
# #SBATCH headers + module loads if you run somewhere else.
# ============================================================================
#SBATCH -J wheat_maskgen
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:59:00
#SBATCH --output=<EULER_PROJECT_PATH>/slurm_logs/wheat_maskgen_%j.out
#SBATCH --error=<EULER_PROJECT_PATH>/slurm_logs/wheat_maskgen_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

# --- edit these ---
REPO="<EULER_PROJECT_PATH>"
FIELD=field_A
DATE=20250715
EXP=perhead_sam2_conf070        # an existing name is overwritten

cd "$REPO"

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat-maskgen
module load eth_proxy 2>/dev/null || true
nvidia-smi

# wheat_head_detection_model.pt has to be in src/mask_generation/weights/, see INSTALL.md.
# profile=phone sets full-res 4032, SAM2 and conf 0.70, so only the crop mode is overridden.
python src/mask_generation/run_mask_generation.py \
  profile=phone \
  dataset.plot_glob=$FIELD/$DATE \
  method.sam_crop_mode=per_head \
  experiment_name=$EXP

echo "masks -> $REPO/results/mask_generation/phone/$FIELD/$DATE/yolo_sam_v1/$EXP/"
