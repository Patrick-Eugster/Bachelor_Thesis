#!/bin/bash -l
#SBATCH -J phone_absgrad_opencv_X
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G      # 48 GB total (6 × 8G). MEASURED peak on the past phone AbsGS run
#SBATCH --cpus-per-task=6     #   (phone_absgrad_20250715, 93 imgs) = 30.0 GiB RAM; this group has
#SBATCH --time=03:59:00       #   fD/0618 = 127 imgs (~1.37x image cache) -> ~40 GiB, so 48 GB.
# TIME: MEASURED 46m/session = 32m train + 14m RENDER + 0.5m metrics (render is slow: ~100 full-res views
#   @3M AbsGS Gaussians; metrics trivial). 4 sessions ~3h; fD/0618 (127 imgs, more views+Gaussians) ~57m ->
#   group ~3h15m -> fits 3h59m (~45m margin). (8 in one 4h job would be ~6h -> that's why it's split.)
# ⚠️ VRAM: that run peaked 22.83/24 GiB — densify_grad_threshold=0.0008 BARELY fits the 4090; fD/0618's
#    larger scene is the OOM risk here. If train_vanilla crashes with CUDA OOM, raise the threshold to 0.001.
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_absgrad_opencv_X_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_absgrad_opencv_X_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# PHONE RECON — AbsGS on the OPENCV arm, GROUP X (4 of 8 sessions).
# ----------------------------------------------------------------------------
# Experiment R3: does AbsGS densification (absolute screen-space gradient) make the phone reconstruction
# sharper than the default densify criterion? A/B is opencv-absgrad (here) vs the EXISTING opencv-default
# baseline (results/.../opencv/vanilla_3dgs/baseline), so we only train the AbsGS arm.
# opencv is the phone recon baseline (fair vs Agisoft; opencv >= agisoft, opencv > pinhole).
#
# AbsGS regime: reconstruction.absgrad=true REQUIRES densify_grad_threshold=0.0008 (AbsGS gradient
# magnitudes are ~4x larger; the default 0.0002 would over-densify and OOM the 24 GB 4090). Runs on the
# gsplat engine (the default; AbsGS is a no-op under diff-gaussian-rasterization). recon-only
# (train+render+metrics, NO seg — cropped opencv frames don't match the masks).
# Baseline config otherwise: gsplat · resolution 1 · 15000 iters · pp off.
# Writes to results/reconstruction/phone/<field>/<date>/opencv/vanilla_3dgs/absgrad (own experiment name,
# does NOT collide with the opencv-default 'baseline').
#
# ⚠️ PREREQUISITE: each session needs its OPENCV SfM variant on Euler
#   (input_plots/phone/<field>/<date>/opencv/{images,sparse/0}). Preflight ABORTS in <1s if missing
#   (rsync the opencv/ folders up first) or if a target absgrad/ folder already exists.
# ============================================================================

VARIANT=opencv
SUBTREE=opencv/vanilla_3dgs
EXP=absgrad
# ordered by image count ASCENDING (84,93,96,127) so the OOM-risk fD/0618 (127 imgs) runs LAST —
# a crash there doesn't waste the three safe sessions before it.
SESSIONS=("field_A 20250627" "field_A 20250706" "field_D 20250715" "field_D 20250618")

REPO=/cluster/project/cropsci/peugste/wheat3dgs
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
cd "$REPO"
nvidia-smi

# ── PREFLIGHT: fail LOUDLY if any target exists OR the opencv/ SfM input is missing ──
echo "=== PREFLIGHT: opencv/ inputs present + absgrad/ targets fresh ==="
COLLIDE=0
for s in "${SESSIONS[@]}"; do
  set -- $s; FIELD=$1; DATE=$2
  OUT="$REPO/results/reconstruction/phone/$FIELD/$DATE/$SUBTREE/$EXP"
  IN="$REPO/input_plots/phone/$FIELD/$DATE/opencv/sparse/0"
  if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "  X EXISTS: $OUT"; COLLIDE=1
  elif [ ! -d "$IN" ]; then
    echo "  X MISSING INPUT (rsync the opencv/ folder up first!): $IN"; COLLIDE=1
  else
    echo "  ok (fresh, opencv input present): $OUT"
  fi
done
if [ "$COLLIDE" -ne 0 ]; then
  echo "ABORTING — a target exists or an opencv/ SfM input is missing (rsync it up first)."
  exit 1
fi

# ── peak VRAM + RAM loggers ── AbsGS grows Gaussians -> watch VRAM approach the 24 GB ceiling.
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

# ── run the 4 sessions; DO NOT set -e (one failure must not kill the rest) ──
declare -A STATUS
for s in "${SESSIONS[@]}"; do
  set -- $s; FIELD=$1; DATE=$2
  echo ""
  echo "================================================================"
  echo "  opencv+AbsGS  ${FIELD} / ${DATE}   (train+render+metrics, 15k)"
  echo "================================================================"
  python src/run_reconstruction.py \
    dataset=phone plot=$FIELD date=$DATE \
    run_train=true run_render=true run_metrics=true \
    reconstruction.iterations=15000 reconstruction.resolution=1 \
    reconstruction.absgrad=true reconstruction.densify_grad_threshold=0.0008 \
    experiment_name=$EXP sfm_variant=$VARIANT
  rc=$?
  if [ $rc -eq 0 ]; then STATUS["$FIELD/$DATE"]=OK; else STATUS["$FIELD/$DATE"]="FAIL(rc=$rc)"; fi
done

# ── summary ──
echo ""
echo "================ opencv+AbsGS (group X) PER-SESSION STATUS ================"
for s in "${SESSIONS[@]}"; do
  set -- $s; FIELD=$1; DATE=$2
  echo "  $FIELD/$DATE : ${STATUS[$FIELD/$DATE]}  -> results/reconstruction/phone/$FIELD/$DATE/$SUBTREE/$EXP"
done
PEAK_VRAM_MIB=$(sort -n "$VRAM_LOG" | tail -1)
PEAK_RAM_GIB=$(awk "BEGIN{printf \"%.2f\", $(sort -n "$RAM_LOG" | tail -1)/1024/1024/1024}")
echo "----------------------------------------------------------------"
echo "Peak VRAM: ${PEAK_VRAM_MIB} MiB ($(awk "BEGIN{printf \"%.2f\", ${PEAK_VRAM_MIB}/1024}") GiB)"
echo "Peak RAM:  ${PEAK_RAM_GIB} GiB (job cgroup)"
echo "Compare vs opencv-default: results/.../opencv/vanilla_3dgs/baseline/results.json"
echo "================================================================"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
