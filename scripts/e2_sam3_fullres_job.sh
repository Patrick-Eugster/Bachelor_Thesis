#!/bin/bash -l
# ============================================================================
# E2 SAM3 @4032 grid block on Euler (batch job, 40 GB A100, FP32).
#
# Completes the full-res YOLOv5 detector block of the phone mask-gen grid for
# SAM3. Reads the full-res YOLOv5 boxes staged into e2_{ff,pt,ph}_sam3/bboxes by
# scripts/e2_sam3_fullres_prep_local.sh (and rsynced up), runs SAM3 on them via
# only_sam at each granularity, then scores each cell. FP32 (ultralytics default,
# no half) to match the existing SAM3 grid cells. Fresh e2_*_sam3 names => no
# overwrite of the SAM1/SAM2 cells; PREFLIGHT aborts if a name/box is missing.
#
# WHY A100 (gpumem:38g, not rtx_4090): SAM3 encoder alone is ~26 GB and does not
# fit the 24 GB rtx_4090 — it must land on the 40 GB A100.
# WHY sbatch + short --time: a <4h --time backfills quickly into the 4h tier.
#
# PREREQUISITES on Euler:
#   1. e2_{ff,pt,ph}_sam3/bboxes pushed for all 6 GT sessions (18 .pt each granularity).
#   2. 6 GT masks:  find input_plots/phone -name '*_gt_mask.png' | wc -l  == 6
#   3. wheat-maskgen env has SAM3 deps + weights/sam3.pt on disk.
#   4. slurm_logs/ exists.
#
# SUBMIT (full absolute path REQUIRED on Euler):
#   sbatch /cluster/project/cropsci/peugste/wheat3dgs/scripts/e2_sam3_fullres_job.sh
#
#SBATCH --job-name=e2_sam3_fullres
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:38g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/e2_sam3_fullres_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/e2_sam3_fullres_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

set -u
REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO" || { echo "repo not found: $REPO"; exit 1; }
mkdir -p "$REPO/slurm_logs"

# --- environment (mirror the working euler maskgen jobs: purge -> conda -> proxy) ---
module purge
unset PYTHONHOME PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate wheat-maskgen
module load eth_proxy 2>/dev/null || true
which python
python --version
# ultralytics writes a results dir on predictor init; point it somewhere writable
python -c "from ultralytics import settings; settings.update({'runs_dir':'$REPO/runs'})" 2>/dev/null || true

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# granularity -> experiment_name (must match the pushed e2_*_sam3 box folders)
declare -A EXP=( [full_frame]=e2_ff_sam3 [per_tile]=e2_pt_sam3 [per_head]=e2_ph_sam3 )
SESSIONS=( "field_A/20250627" "field_A/20250706" "field_A/20250715"
           "field_D/20250627" "field_D/20250706" "field_D/20250715" )

echo ""; echo "=== PREFLIGHT ==="
FAIL=0
[ -f "$REPO/src/mask_generation/weights/sam3.pt" ] || { echo "  X missing weights/sam3.pt"; FAIL=1; }
NGT=$(find input_plots/phone -name '*_gt_mask.png' 2>/dev/null | wc -l)
[ "$NGT" -eq 6 ] || { echo "  X expected 6 GT masks, found $NGT"; FAIL=1; }
for gran in "${!EXP[@]}"; do
  exp="${EXP[$gran]}"
  for sess in "${SESSIONS[@]}"; do
    box="results/mask_generation/phone/$sess/yolo_sam_v1/$exp/bboxes"
    [ -d "$box" ] || { echo "  X missing boxes: $box"; FAIL=1; }
  done
done
[ "$FAIL" -ne 0 ] && { echo "ABORTING (nothing run)."; exit 1; }
echo "  ok: sam3.pt present, 6 GT masks, all e2_*_sam3 boxes staged"

RUN_START=$(date +%s)
declare -a RESULTS
for gran in full_frame per_tile per_head; do
  exp="${EXP[$gran]}"
  echo ""; echo "############################################################"
  echo "# SAM3 @4032  $gran  ($exp)   $(date '+%H:%M:%S')"
  echo "############################################################"
  c0=$(date +%s)
  # only_sam: reads $exp/bboxes, runs SAM3 (FP32, no half) at this granularity, writes SAM3 masks.
  if python src/mask_generation/run_mask_generation.py dataset=phone method=yolo_sam_v1 \
       only_labeled_images=true \
       method.sam_crop_mode="$gran" method.sam_backend=sam3 only_sam=true \
       experiment_name="$exp"
  then mg="OK  "; else mg="FAIL"; fi
  # score the cell into eval_experiment=$exp (deterministic CPU IoU eval vs the 6 GT masks)
  if [ "$mg" = "OK  " ] && python src/mask_generation/evaluation/eval_masks_instance.py \
       dataset=phone method=yolo_sam_v1 mask_gen_experiment="$exp" eval_experiment="$exp"
  then ev="OK  "; else ev="FAIL"; fi
  c1=$(date +%s)
  RESULTS+=( "mg=$mg eval=$ev  $gran ($exp)  ($((c1-c0))s)" )
  echo "--- mg=$mg eval=$ev  $gran  in $((c1-c0))s | elapsed $(( (c1-RUN_START)/60 ))m ---"
done

echo ""; echo "============================================================"
echo " E2 SAM3 @4032 DONE in $(( ($(date +%s)-RUN_START)/60 )) min"
printf '   %s\n' "${RESULTS[@]}"
echo "============================================================"
echo "Eval JSONs at results/mask_generation/phone/evaluation/yolo_sam_v1/masks_instance/e2_{ff,pt,ph}_sam3/"
echo "Next: rsync those e2_*_sam3 eval folders back locally to fold into the grid."
