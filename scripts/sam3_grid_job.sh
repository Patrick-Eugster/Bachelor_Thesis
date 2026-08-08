#!/bin/bash -l
# SAM3 mask-generation GRID on Euler (batch job, 40GB A100).
# Runs the box-prompted SAM3 cells (3 granularities x 3 detectors) via only_sam,
# reading the gt_*_sam3/bboxes/*.pt already pushed to Euler. Scoring is done LOCALLY.
#
# WHY sbatch (not an interactive code-server): a short --time lands in a short-runtime
# partition tier and backfills quickly, so you get a 40GB GPU without babysitting a
# long interactive session at night. Keep --time UNDER 4h to stay in the 4h tier.
#
# PREREQUISITES on Euler (all done this session; re-check if unsure):
#   1. gt_ff_sam3 / gt_tile_sam3 / gt_head_sam3 bboxes pushed (18 files each).
#   2. manual_label GT present  ->  find input_plots/phone -name '*_gt_mask.png' | wc -l  == 6
#   3. wheat-maskgen env has SAM3 deps (clip ftfy wcwidth regex timm) + weights/sam3.pt on disk.
#   4. slurm_logs/ dir exists (other jobs use it; if not: mkdir -p .../wheat3dgs/slurm_logs)
#
# SUBMIT (full absolute path is REQUIRED on Euler):
#   sbatch /cluster/project/cropsci/peugste/wheat3dgs/scripts/sam3_grid_job.sh
#
#SBATCH --job-name=sam3_grid
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:38g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/sam3_grid_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/sam3_grid_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

set -u
REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO" || { echo "repo not found: $REPO"; exit 1; }
mkdir -p "$REPO/slurm_logs"

# --- environment (mirror the working euler jobs: purge -> conda -> proxy) ---
module purge
unset PYTHONHOME PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate wheat-maskgen
module load eth_proxy 2>/dev/null || true   # internet in case ultralytics does a one-off fetch
which python
python --version

# ultralytics writes a results dir on predictor init; point it somewhere writable
# (persisted already, but set again so a fresh job is self-contained)
python -c "from ultralytics import settings; settings.update({'runs_dir':'$REPO/runs'})" 2>/dev/null || true

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== GT images visible (expect 6): $(find input_plots/phone -name '*_gt_mask.png' 2>/dev/null | wc -l) ==="

# GRID definition. Edit these arrays to trim work (e.g. delete a granularity already done).
# "granularity:experiment_name"  — experiment_name MUST match the pushed gt_*_sam3 box folders.
# per_head already ran + was pulled locally, so it's dropped here. To include it, re-add
# "per_head:gt_head_sam3" to the front of the list.
GRANS=( "per_tile:gt_tile_sam3" "full_frame:gt_ff_sam3" )
DETECTORS=( yolo11_sam sahi_yolo_sam yolo_sam_v1 )   # yolo11 first = your best detector

RUN_START=$(date +%s)
declare -a RESULTS
for pair in "${GRANS[@]}"; do
  gran="${pair%%:*}"; exp="${pair##*:}"
  for det in "${DETECTORS[@]}"; do
    cell="$gran/$det"
    echo ""
    echo "############################################################"
    echo "# SAM3 CELL  $cell  ($exp)   $(date '+%H:%M:%S')"
    echo "############################################################"
    c0=$(date +%s)
    if python src/mask_generation/run_mask_generation.py dataset=phone method="$det" \
         only_labeled_images=true roi.enabled=true \
         method.sam_crop_mode="$gran" method.sam_backend=sam3 only_sam=true \
         experiment_name="$exp"
    then st="OK  "; else st="FAIL"; fi
    c1=$(date +%s)
    RESULTS+=( "$st  $cell  ($((c1-c0))s)" )
    echo "--- $st  $cell  in $((c1-c0))s  | elapsed $(( (c1-RUN_START)/60 ))m ---"
  done
done

echo ""
echo "============================================================"
echo " SAM3 GRID DONE in $(( ($(date +%s)-RUN_START)/60 )) min"
printf '   %s\n' "${RESULTS[@]}"
echo "============================================================"
echo "Next: rsync results/mask_generation/phone back locally, then score with"
echo "  eval_masks_instance.py + aggregate_maskgen_grid.py"
