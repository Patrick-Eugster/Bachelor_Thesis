#!/bin/bash -l
#SBATCH -J phone_recon_pinhole
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G      # 36 GB total (6 x 6G) — phone recon RAM peak ~28 GiB measured (~8 GB headroom)
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00       # actual ~2.6h for 4 sessions (~38 min each) — generous ceiling, releases early
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_recon_pinhole_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_recon_pinhole_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# PHONE RECON — camera-model experiment, arm 1/3: PINHOLE (SIMPLE_PINHOLE baseline)
# ----------------------------------------------------------------------------
# Trains 3DGS on our COLMAP baseline SfM (session root: images/ + sparse/0/) for the 4 sessions that
# also have opencv/ + radial/ variants, so pinhole vs opencv vs radial is a clean paired A/B (only the
# camera model / lens-distortion handling differs). recon-only (train+render+metrics, NO seg).
#
# Baseline config (all explicit so a future default change can't move it):
#   gsplat renderer (default) · resolution 1 · 15000 iters · default densification (absgrad=false,
#   densify_grad_threshold=0.0002, densify_until 11000) · principal-point off (phone pp ~ center, no-op).
# Writes to results/reconstruction/phone/<field>/<date>/vanilla_3dgs/baseline  (own subtree).
# Companion arms: phone_recon_opencv_job.sh (sfm_variant=opencv), phone_recon_radial_job.sh.
# ============================================================================

VARIANT=""                 # pinhole = COLMAP baseline at the session root
VARIANT_TAG=pinhole
SUBTREE=vanilla_3dgs       # baseline result subtree
EXP=baseline
SESSIONS=("field_A 20250618" "field_A 20250715" "field_D 20250627" "field_D 20250706")

REPO=/cluster/project/cropsci/peugste/wheat3dgs
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
cd "$REPO"
nvidia-smi

# ── PREFLIGHT: fail LOUDLY in the first second if any target folder already exists, so a collision
#    never wastes hours (the overwrite guard would sys.exit anyway — this just surfaces it up front).
echo "=== PREFLIGHT: checking target folders are fresh ==="
COLLIDE=0
for s in "${SESSIONS[@]}"; do
  set -- $s; FIELD=$1; DATE=$2
  OUT="$REPO/results/reconstruction/phone/$FIELD/$DATE/$SUBTREE/$EXP"
  if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "  X EXISTS: $OUT"; COLLIDE=1
  else
    echo "  ok (fresh): $OUT"
  fi
done
if [ "$COLLIDE" -ne 0 ]; then
  echo "ABORTING before any training — a target folder already exists. Rename EXP or move the old folder."
  exit 1
fi

# ── peak VRAM + RAM loggers (background samplers) ──
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

# one combined per-plot report for the whole job
export WHEAT_RUN_REPORT="$REPO/slurm_logs/run_report_${SLURM_JOB_ID}.txt"

# ── run the 4 sessions; DO NOT set -e — one bad session must not kill the batch ──
declare -A STATUS
VARIANT_ARG=""
[ -n "$VARIANT" ] && VARIANT_ARG="sfm_variant=$VARIANT"
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
    experiment_name=$EXP $VARIANT_ARG
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
