#!/bin/bash -l
#SBATCH -J phone_recon_radial_breadth
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G      # 36 GB total — fD/0618 (127 imgs) ~26-28 GB, safe headroom
#SBATCH --cpus-per-task=6
#SBATCH --time=03:59:00       # 4 sessions ~2.3-2.7h -> strictly under 4h (short-queue tier)
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_recon_radial_breadth_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_recon_radial_breadth_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# PHONE RECON — RADIAL BREADTH: the OTHER 4 canonical sessions (completes the 8-session radial arm).
# ----------------------------------------------------------------------------
# ⚠️ PREREQUISITE: each session must already have a RADIAL SfM variant on disk
#   (input_plots/phone/<field>/<date>/radial/{images,sparse/0}). Built LOCALLY 2026-08-13
#   (run_radial_sfm_missing4.sh, all 4 = 100% registered, one model) — rsync the radial/ folders up first.
#   The preflight below ABORTS in <1s if any radial/ input is missing, so this job is safe to submit early.
#
# Trains 3DGS on the SIMPLE_RADIAL SfM variant (one k1 term, undistorted to a cropped ~3950 PINHOLE frame).
# Same 4 sessions + pinned split as the pinhole/opencv breadth arms -> completes the pinhole/opencv/radial
# paired A/B on all 8 sessions. recon-only (train+render+metrics, NO seg — cropped frames don't match masks).
# Marker/ROI masked metrics auto-skip (no marker file under radial/); whole-image + inner region used.
# Baseline config: gsplat · resolution 1 · 15000 iters · default densify · pp off.
# Writes to results/reconstruction/phone/<field>/<date>/radial/vanilla_3dgs/baseline (own subtree).
# ============================================================================

VARIANT="radial"
VARIANT_TAG=radial
SUBTREE=radial/vanilla_3dgs
EXP=baseline
SESSIONS=("field_A 20250627" "field_A 20250706" "field_D 20250618" "field_D 20250715")

REPO=/cluster/project/cropsci/peugste/wheat3dgs
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
cd "$REPO"
nvidia-smi

# ── PREFLIGHT: fail LOUDLY if any target exists OR the radial/ SfM input is missing ──
echo "=== PREFLIGHT: checking target folders are fresh + radial/ inputs present ==="
COLLIDE=0
for s in "${SESSIONS[@]}"; do
  set -- $s; FIELD=$1; DATE=$2
  OUT="$REPO/results/reconstruction/phone/$FIELD/$DATE/$SUBTREE/$EXP"
  IN="$REPO/input_plots/phone/$FIELD/$DATE/radial/sparse/0"
  if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "  X EXISTS: $OUT"; COLLIDE=1
  elif [ ! -d "$IN" ]; then
    echo "  X MISSING INPUT (rsync the radial/ folder up first!): $IN"; COLLIDE=1
  else
    echo "  ok (fresh, radial input present): $OUT"
  fi
done
if [ "$COLLIDE" -ne 0 ]; then
  echo "ABORTING — a target exists or a radial/ SfM input is missing (rsync it up first)."
  exit 1
fi

# ── peak VRAM + RAM loggers ──
VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!
RAM_LOG="$REPO/slurm_logs/ram_${SLURM_JOB_ID}.log"
CGROUP_PATH=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)
CGROUP_MEM_FILE="/sys/fs/cgroup${CGROUP_PATH}/memory.current"
( while true; do
    if [ -r "$CGROUP_MEM_FILE" ]; then cat "$CGROUP_MEM_FILE"
    else ps -u "$USER" -o rss= | awk '{sum+=$1} END{print sum*1024}'; fi
    sleep 5
  done ) > "$RAM_LOG" &
RAM_PID=$!
trap "kill $VRAM_PID $RAM_PID 2>/dev/null" EXIT

export WHEAT_RUN_REPORT="$REPO/slurm_logs/run_report_${SLURM_JOB_ID}.txt"

# ── run the 4 sessions; DO NOT set -e ──
declare -A STATUS
for s in "${SESSIONS[@]}"; do
  set -- $s; FIELD=$1; DATE=$2
  echo ""
  echo "================================================================"
  echo "  ${VARIANT_TAG}  ${FIELD} / ${DATE}   (train+render+metrics, 15k, default densify)"
  echo "================================================================"
  python src/run_reconstruction.py \
    dataset=phone plot=$FIELD date=$DATE \
    run_train=true run_render=true run_metrics=true \
    reconstruction.iterations=15000 reconstruction.resolution=1 reconstruction.absgrad=false \
    experiment_name=$EXP sfm_variant=$VARIANT
  rc=$?
  if [ $rc -eq 0 ]; then STATUS["$FIELD/$DATE"]=OK; else STATUS["$FIELD/$DATE"]="FAIL(rc=$rc)"; fi
done

# ── summary ──
echo ""
echo "================ ${VARIANT_TAG} PER-SESSION STATUS ================"
for s in "${SESSIONS[@]}"; do
  set -- $s; FIELD=$1; DATE=$2
  echo "  $FIELD/$DATE : ${STATUS[$FIELD/$DATE]}  -> results/reconstruction/phone/$FIELD/$DATE/$SUBTREE/$EXP"
done
PEAK_VRAM_MIB=$(sort -n "$VRAM_LOG" | tail -1)
PEAK_RAM_GIB=$(awk "BEGIN{printf \"%.2f\", $(sort -n "$RAM_LOG" | tail -1)/1024/1024/1024}")
echo "----------------------------------------------------------------"
echo "Peak VRAM: ${PEAK_VRAM_MIB} MiB ($(awk "BEGIN{printf \"%.2f\", ${PEAK_VRAM_MIB}/1024}") GiB)"
echo "Peak RAM:  ${PEAK_RAM_GIB} GiB (job cgroup)"
echo "Combined report: $WHEAT_RUN_REPORT"
echo "================================================================"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
