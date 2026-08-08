#!/bin/bash -l
# SAM3 TEXT-PROMPT probe on Euler (batch job, 40GB A100).
# Runs the text/concept segmenter (SAM3SemanticPredictor) over the 13 GT images (phone 6 + fip 7) with a
# conf sweep 0.10..0.50, in TWO phases: (1) full-frame, then (2) tiled 2x2. Diagnostic only (no merge/GT
# scoring) — you eyeball the overlays to see which phrase + conf + resolution actually resolves heads.
#
# WHY sbatch (not interactive): a short --time lands in the short-runtime tier and backfills fast, so you get
# the 40GB GPU without babysitting a long code-server. Keep --time UNDER 4h to stay in the 4h tier.
#
# PREREQUISITES on Euler (mostly done in the box-prompt session; re-check if unsure):
#   1. weights/sam3.pt on disk (3.45GB) + wheat-maskgen has the text deps (clip ftfy wcwidth regex timm).
#   2. GT images present: input_plots/phone/*/*/manual_label/*_sets/  (6) and
#      input_plots/fip/*/manual_label/*.txt (7) with their images/ — pushed by the code rsync (input_plots
#      is NOT excluded). The script prints the discovered count before running.
#   3. slurm_logs/ dir exists.
#
# SUBMIT (full absolute path is REQUIRED on Euler):
#   sbatch /cluster/project/cropsci/peugste/wheat3dgs/scripts/sam3_text_probe_job.sh
#
#SBATCH --job-name=sam3_text
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:38g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/sam3_text_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/sam3_text_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

set -u
REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO" || { echo "repo not found: $REPO"; exit 1; }
mkdir -p "$REPO/slurm_logs"

# output under results/ (NOT docs/ — docs is excluded from the Euler rsync) so your normal pull grabs it
OUT="$REPO/results/sam3_text_probe"

# --- environment (mirror the working euler jobs: purge -> conda -> proxy) ---
module purge
unset PYTHONHOME PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate wheat-maskgen
module load eth_proxy 2>/dev/null || true   # internet in case ultralytics does a one-off fetch
which python
python --version

# ultralytics writes a results dir on predictor init; point it somewhere writable
python -c "from ultralytics import settings; settings.update({'runs_dir':'$REPO/runs'})" 2>/dev/null || true

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== GT images discovered (expect 13 = phone 6 + fip 7): ==="
python -c "import sys; sys.path.insert(0,'src/analysis'); from sam3_text_probe import discover_gt_images as d; print(len(d('all')))"

# MODE selects which phases run: both (default) | full | tiles. Each phase writes its OWN
# summary_<mode>.json (plus a back-compat summary.json), so a full-only rerun does NOT clobber the tiled data.
# Submit full-only with:  sbatch --export=ALL,MODE=full  <this script>
MODE="${MODE:-both}"
echo "=== MODE=$MODE ==="
RUN_START=$(date +%s)
p1=0; p2=0

if [ "$MODE" = "both" ] || [ "$MODE" = "full" ]; then
  echo ""
  echo "############################################################"
  echo "# PHASE 1 — FULL FRAME conf sweep   $(date '+%H:%M:%S')"
  echo "############################################################"
  python src/analysis/sam3_text_probe.py --auto all --mode full --fp32 --out "$OUT"
  p1=$?
fi

if [ "$MODE" = "both" ] || [ "$MODE" = "tiles" ]; then
  echo ""
  echo "############################################################"
  echo "# PHASE 2 — TILED 2x2 conf sweep    $(date '+%H:%M:%S')"
  echo "############################################################"
  python src/analysis/sam3_text_probe.py --auto all --mode tiles --tiles 2 --fp32 --out "$OUT"
  p2=$?
fi

echo ""
echo "============================================================"
echo " SAM3 TEXT PROBE DONE ($MODE) in $(( ($(date +%s)-RUN_START)/60 )) min  | phase1=$p1 phase2=$p2"
echo "============================================================"
echo "Output: $OUT/<dataset>/<plot>/<stem>/<phrase>/{full_frame,tiled}/  + summary.json"
echo "Pull it back with your results rsync (it's under results/, so the normal pull grabs it), then eyeball:"
echo "  - within a phrase's full_frame/ : compare c10..c50 (confidence)"
echo "  - full_frame/ vs tiled/         : does tiling resolve heads the downscaled full frame misses?"
echo "  - across phrases                : which vocabulary the model latches onto (summary.json has counts)"
