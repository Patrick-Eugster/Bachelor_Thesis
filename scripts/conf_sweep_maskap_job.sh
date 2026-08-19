#!/bin/bash -l
# YOLO conf sweep + mask-level AP on Euler (batch job, rtx_4090 24GB).
# Runs src/analysis/sweep_conf_mask_ap.py over the 6-cell grid: {sam1,sam2} x {full_frame,per_tile,per_head},
# each doing the DENSE conf sweep on the 6 GT images (configs/manifests/gt6.json). Writes one JSON per cell
# to results/mask_generation/phone/evaluation/conf_sweep/ (results/ IS pulled back locally; docs/ is not).
#
# WHY the two mode classes cost differently (see the script docstring):
#   full_frame / per_head : box-independent -> ONE low-floor SAM run per image covers the whole conf curve.
#   per_tile              : tile crop grows to its box group -> RE-RUN SAM at each conf (verified necessary).
# per_tile therefore dominates the runtime (~31 confs x 6 imgs); full_frame/per_head are ~one pass each.
#
# WHY rtx_4090 (not the SAM3 A100): SAM1/SAM2 fit in 24GB (SAM2 full_frame ~7GB via sam_ul_decode_batch;
# SAM1 ViT-H a few GB; per_tile/per_head are small crops). SAM3 is skipped here, so no 38GB node needed.
#
# PREREQUISITES on Euler (verify before submitting):
#   1. GT data pushed (18 files, ~30MB): the 6 images/, manual_label/*_sets/set0_instances.png, and
#      results/.../metrics_v1/bboxes_with_conf/*.pt  -> see scripts/push_conf_sweep_data_to_euler.sh
#   2. configs/manifests/gt6.json present (git-tracked, arrives with a normal code push)
#   3. weights/sam_vit_h_4b8939.pth (SAM1) + weights/sam2.1_l.pt (SAM2) on disk:
#        ls -la $REPO/src/mask_generation/weights/{sam_vit_h_4b8939.pth,sam2.1_l.pt}
#   4. slurm_logs/ exists
#
# SUBMIT (full absolute path is REQUIRED on Euler):
#   sbatch /cluster/project/cropsci/peugste/wheat3dgs/scripts/conf_sweep_maskap_job.sh
#
#SBATCH --job-name=conf_sweep_maskap
#SBATCH --time=03:00:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/conf_sweep_maskap_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/conf_sweep_maskap_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

set -u
REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO" || { echo "repo not found: $REPO"; exit 1; }
mkdir -p "$REPO/slurm_logs"
OUTDIR="$REPO/results/mask_generation/phone/evaluation/conf_sweep"
mkdir -p "$OUTDIR"

# --- environment (mirror the working euler maskgen jobs: purge -> conda -> proxy) ---
module purge
unset PYTHONHOME PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate wheat-maskgen
module load eth_proxy 2>/dev/null || true
which python; python --version

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== manifest images visible (expect 6): $(python -c "import json;print(len(json.load(open('configs/manifests/gt6.json'))))" 2>/dev/null) ==="

BACKENDS=( sam1 sam2 )
MODES=( per_tile full_frame per_head )   # per_tile first = the production baseline we most care about

RUN_START=$(date +%s)
declare -a RESULTS
for be in "${BACKENDS[@]}"; do
  for mode in "${MODES[@]}"; do
    cell="${be}_${mode}"
    out="$OUTDIR/${cell}.json"
    echo ""
    echo "############################################################"
    echo "# CONF SWEEP  $cell   $(date '+%H:%M:%S')"
    echo "############################################################"
    c0=$(date +%s)
    if python src/analysis/sweep_conf_mask_ap.py --manifest configs/manifests/gt6.json \
         --backend "$be" --mode "$mode" --sweep dense --iou 0.5 0.3 \
         --sam1_decode_batch 16 --out "$out"
    then st="OK  "; else st="FAIL"; fi
    c1=$(date +%s)
    RESULTS+=( "$st  $cell  ($(( (c1-c0)/60 ))m$(( (c1-c0)%60 ))s)" )
    echo "--- $st  $cell  in $((c1-c0))s | elapsed $(( (c1-RUN_START)/60 ))m ---"
  done
done

echo ""
echo "============================================================"
echo " CONF SWEEP GRID DONE in $(( ($(date +%s)-RUN_START)/60 )) min"
printf '   %s\n' "${RESULTS[@]}"
echo "============================================================"
echo "Next: rsync results/mask_generation/phone/evaluation/conf_sweep back locally, then plot/compare."
