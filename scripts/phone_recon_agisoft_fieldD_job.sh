#!/bin/bash -l
#SBATCH -J phone_recon_agisoft_fieldD
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G      # 36 GB total — fD/0618 is 127 imgs (largest) ~26-28 GB; safe headroom
#SBATCH --cpus-per-task=6
#SBATCH --time=03:59:00       # 5 runs ~2.9h -> strictly under 4h (short-queue tier)
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_recon_agisoft_fieldD_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_recon_agisoft_fieldD_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# PHONE RECON — AGISOFT arm (R1), field_D. Trains 3DGS on Agisoft's OWN undistorted reconstruction,
# to compare Agisoft-SfM vs our-COLMAP-SfM 3DGS quality. recon-only (train+render+metrics, NO seg —
# agisoft-undistorted pixels don't match the COLMAP-made masks). Pinned split is suffix-robust
# (split_utils._norm_stem) so each arm holds out the SAME physical test frames as the baseline.
#
# PER-SESSION VARIANT (3rd token). fD/{0618,0706,0715} use native 2-camera agisoft/. fD/20250627 is run
# BOTH ways so we can compare them:
#   - agisoft_2group_old = Agisoft's NATIVE 2-camera output (3948x2938 + 3942x2944) -> the CONSISTENT arm,
#     matching how the other 7 sessions are treated (all their agisoft/ are native 2-camera too).
#   - agisoft = the supervisor's MERGED single-camera set (3882x2878) -> a SPECIAL ADDITIONAL case, since the
#     merge is a manual one-off only fD/0627 received. (verified: images==cameras, split maps 12/12 for both.)
#   NOTE fD/0706 (blurry): Agisoft registered only 63 imgs -> 2 of 11 pinned test frames absent -> tests 9/11.
# Baseline config: gsplat · resolution 1 · 15000 iters · default densify · pp off.
# Writes to results/reconstruction/phone/field_D/<date>/<variant>/vanilla_3dgs/baseline (own subtree each).
# ============================================================================

EXP=baseline
# entries: "FIELD DATE VARIANT"
SESSIONS=(
  "field_D 20250618 agisoft"
  "field_D 20250627 agisoft_2group_old"   # native 2-camera (consistent main arm)
  "field_D 20250706 agisoft"
  "field_D 20250715 agisoft"
  "field_D 20250627 agisoft"              # merged 1-camera gold (special additional case)
)

REPO=/cluster/project/cropsci/peugste/wheat3dgs
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
cd "$REPO"
nvidia-smi

# ── PREFLIGHT: fail LOUDLY if any target exists OR the <variant>/ input is missing ──
echo "=== PREFLIGHT: checking target folders are fresh + inputs present ==="
COLLIDE=0
for s in "${SESSIONS[@]}"; do
  set -- $s; FIELD=$1; DATE=$2; VAR=$3
  OUT="$REPO/results/reconstruction/phone/$FIELD/$DATE/$VAR/vanilla_3dgs/$EXP"
  IN="$REPO/input_plots/phone/$FIELD/$DATE/$VAR/sparse/0"
  if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "  X EXISTS: $OUT"; COLLIDE=1
  elif [ ! -d "$IN" ]; then
    echo "  X MISSING INPUT: $IN"; COLLIDE=1
  else
    echo "  ok (fresh, input present): $OUT"
  fi
done
if [ "$COLLIDE" -ne 0 ]; then
  echo "ABORTING — a target exists or a variant input is missing."
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

# ── run each (session, variant); DO NOT set -e ──
declare -A STATUS
for s in "${SESSIONS[@]}"; do
  set -- $s; FIELD=$1; DATE=$2; VAR=$3
  KEY="$FIELD/$DATE/$VAR"
  echo ""
  echo "================================================================"
  echo "  agisoft[$VAR]  ${FIELD} / ${DATE}   (train+render+metrics, 15k, default densify)"
  echo "================================================================"
  python src/run_reconstruction.py \
    dataset=phone plot=$FIELD date=$DATE \
    run_train=true run_render=true run_metrics=true \
    reconstruction.iterations=15000 reconstruction.resolution=1 reconstruction.absgrad=false \
    experiment_name=$EXP sfm_variant=$VAR
  rc=$?
  if [ $rc -eq 0 ]; then STATUS["$KEY"]=OK; else STATUS["$KEY"]="FAIL(rc=$rc)"; fi
done

# ── summary ──
echo ""
echo "================ agisoft field_D PER-RUN STATUS ================"
for s in "${SESSIONS[@]}"; do
  set -- $s; FIELD=$1; DATE=$2; VAR=$3
  echo "  $FIELD/$DATE [$VAR] : ${STATUS[$FIELD/$DATE/$VAR]}  -> results/reconstruction/phone/$FIELD/$DATE/$VAR/vanilla_3dgs/$EXP"
done
PEAK_VRAM_MIB=$(sort -n "$VRAM_LOG" | tail -1)
PEAK_RAM_GIB=$(awk "BEGIN{printf \"%.2f\", $(sort -n "$RAM_LOG" | tail -1)/1024/1024/1024}")
echo "----------------------------------------------------------------"
echo "Peak VRAM: ${PEAK_VRAM_MIB} MiB ($(awk "BEGIN{printf \"%.2f\", ${PEAK_VRAM_MIB}/1024}") GiB)"
echo "Peak RAM:  ${PEAK_RAM_GIB} GiB (job cgroup)"
echo "Combined report: $WHEAT_RUN_REPORT"
echo "================================================================"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
