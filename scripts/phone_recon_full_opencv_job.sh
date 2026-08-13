#!/bin/bash -l
#SBATCH -J phone_recon_full_opencv
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G      # 36 GB total (6 x 6G) — phone recon RAM peak ~28 GiB measured (~8 GB headroom)
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00       # actual ~2.6h for 4 sessions (~38 min each) — generous ceiling, releases early
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_recon_full_opencv_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_recon_full_opencv_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# PHONE RECON — camera-model experiment, arm 4/4: FULL_OPENCV (Agisoft's own model, distortion warped out)
# ----------------------------------------------------------------------------
# Trains 3DGS on the FULL_OPENCV SfM variant (full_opencv/images/ + full_opencv/sparse/0/ — COLMAP fit the
# 12-parameter FULL_OPENCV lens model (k1..k6 + p1,p2, the SAME family Agisoft uses), then image_undistorter
# warped it out to a clean PINHOLE frame, cropped ~3900x2930). This is OUR independent full-distortion arm,
# so the fair comparison against the Agisoft reference no longer relies on the weaker OPENCV/SIMPLE_PINHOLE
# distortion handling. Same 4 sessions + same pinned split as the pinhole arm → clean paired A/B.
# recon-only (train+render+metrics, NO seg — cropped frames don't match the masks). Marker/ROI masked
# metrics auto-skip here (no marker_points3d.json under full_opencv/) — whole-image PSNR + the common-region
# / edge-band metric (added later, re-run metrics only) are the signals for residual-distortion gain.
#
# NAMING NOTE: this writes to the full_opencv/ variant subtree for now, exactly like the opencv/radial arms,
# to avoid breaking the current pinhole="baseline" layout. The PLANNED move (later, deliberately) is to make
# FULL_OPENCV the pipeline baseline and re-label the current SIMPLE_PINHOLE baseline as a "simple_pinhole"
# arm — NOT done here to prevent confusion / breakage mid-experiment.
#
# Baseline config (explicit): gsplat · resolution 1 · 15000 iters · default densification · pp off.
# Writes to results/reconstruction/phone/<field>/<date>/full_opencv/vanilla_3dgs/baseline  (own subtree).
# ============================================================================

VARIANT="full_opencv"                  # reads full_opencv/images/ + full_opencv/sparse/0/
VARIANT_TAG=full_opencv
SUBTREE=full_opencv/vanilla_3dgs        # own result subtree — never touches the pinhole baseline
EXP=baseline
SESSIONS=("field_A 20250618" "field_A 20250715" "field_D 20250627" "field_D 20250706")

REPO=/cluster/project/cropsci/peugste/wheat3dgs
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
cd "$REPO"
nvidia-smi

# ── PREFLIGHT: fail LOUDLY in the first second if any target folder already exists ──
echo "=== PREFLIGHT: checking target folders are fresh ==="
COLLIDE=0
for s in "${SESSIONS[@]}"; do
  set -- $s; FIELD=$1; DATE=$2
  OUT="$REPO/results/reconstruction/phone/$FIELD/$DATE/$SUBTREE/$EXP"
  IN="$REPO/input_plots/phone/$FIELD/$DATE/full_opencv/sparse/0"
  if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "  X EXISTS: $OUT"; COLLIDE=1
  elif [ ! -d "$IN" ]; then
    echo "  X MISSING INPUT: $IN"; COLLIDE=1
  else
    echo "  ok (fresh, input present): $OUT"
  fi
done
if [ "$COLLIDE" -ne 0 ]; then
  echo "ABORTING before any training — a target exists or a full_opencv/ input is missing."
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
