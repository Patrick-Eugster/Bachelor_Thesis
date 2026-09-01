#!/bin/bash -l
# ============================================================================
# TEMPLATE — mask generation (YOLO detection + SAM masks) as a SLURM job.
#
# Generic example. Copy it, fill in the <PLACEHOLDERS>, and adjust the #SBATCH
# headers + module loads to your own cluster.
#
# Pipeline stage 2 of 4 — see the repo README + configs/mask_generation/.
# ============================================================================
#SBATCH -J wheat_maskgen
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G        # ~48 GB total; per_head mask-gen is RAM-heavy (many full-res masks)
#SBATCH --time=03:59:00
#SBATCH --output=<CLUSTER_PROJECT_PATH>/slurm_logs/wheat_maskgen_%j.out
#SBATCH --error=<CLUSTER_PROJECT_PATH>/slurm_logs/wheat_maskgen_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

# --- edit these ---
REPO="<CLUSTER_PROJECT_PATH>"
FIELD=field_A
DATE=20250715
EXP=maskgen_example             # experiment_name — unique per run

cd "$REPO"

# --- environment: mask-gen needs the SEPARATE env (torch >=2.4 + sahi/SAM/yolov5) ---
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat-maskgen                               # NOT the train/seg env
module load eth_proxy 2>/dev/null || true

# SAM weights must be present under src/mask_generation/weights/ (gitignored — download separately).

# --- detect (YOLOv5) + segment (SAM) ---
# method=yolo_sam_v1 (default) or method=sahi_yolo_sam. sam_crop_mode=per_head crops each
# box before SAM (best small-head recall); drop it for a single full-frame SAM pass.
python src/mask_generation/run_mask_generation.py \
  dataset=phone method=yolo_sam_v1 \
  dataset.plot_glob=$FIELD/$DATE \
  method.sam_crop_mode=per_head \
  experiment_name=$EXP

# Output: results/mask_generation/phone/$FIELD/$DATE/<variant>/yolo_sam_v1/$EXP/{bboxes,masks}
