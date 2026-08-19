#!/bin/bash -l
#SBATCH -J phone_res2_A0715
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G      # 48 GB total (6 × 8G) — A/0715 is 93 imgs; res 2 needs LESS than the
#SBATCH --cpus-per-task=6     #   res-1 baseline (half-area image cache) so 48 GB is comfortable.
#SBATCH --time=01:00:00       # single res-2 run (train+render+metrics). res-2 train on 93 imgs ~10-15 min
                              #   (FIP res2 was ~9 min on 36 imgs); render+metrics small -> 1h ceiling.
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_res2_A0715_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_res2_A0715_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# PHONE RECON — resolution-2 comparison on field_A / 20250715, OPENCV arm.
# ----------------------------------------------------------------------------
# Mirror of the FIP res-ablation: is a half-resolution phone train a worthwhile detail-vs-cost trade?
# The res-1 reference is the EXISTING opencv baseline (.../opencv/vanilla_3dgs/baseline), so we only
# train the res-2 arm here. Config = the phone recon baseline otherwise:
#   gsplat · 15000 iters · default densification (NO AbsGS) · pp OFF · sfm_variant=opencv.
# recon-only (train+render+metrics, NO seg — cropped opencv frames don't match the masks).
#
# ⚠️ INTERPRETATION CAVEAT: at res 2 the render AND the GT are 2x downscaled, so PSNR/SSIM/LPIPS are
#   scored against an easier, half-detail target and are NOT directly comparable to the res-1 baseline
#   in absolute terms. Read this as detail-vs-cost (train time / VRAM / Gaussian count), not a PSNR race.
#
# ⚠️ PREREQUISITE: opencv SfM variant on Euler at
#   input_plots/phone/field_A/20250715/opencv/{images,sparse/0}. Preflight ABORTS if missing, or if the
#   res2_baseline target already exists.
# Writes results/reconstruction/phone/field_A/20250715/opencv/vanilla_3dgs/res2_baseline.
# ============================================================================

FIELD=field_A
DATE=20250715
VARIANT=opencv
SUBTREE=opencv/vanilla_3dgs
EXP=res2_baseline

REPO=/cluster/project/cropsci/peugste/wheat3dgs
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
cd "$REPO"
export WHEAT_RENDERER=gsplat
nvidia-smi

# ── PREFLIGHT: opencv input present + res2 target fresh ──
echo "=== PREFLIGHT ==="
OUT="$REPO/results/reconstruction/phone/$FIELD/$DATE/$SUBTREE/$EXP"
IN="$REPO/input_plots/phone/$FIELD/$DATE/opencv/sparse/0"
FAIL=0
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "  X EXISTS: $OUT"; FAIL=1
elif [ ! -d "$IN" ]; then
  echo "  X MISSING INPUT (rsync the opencv/ folder up first!): $IN"; FAIL=1
else
  echo "  ok (fresh, opencv input present): $OUT"
fi
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }

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

echo ""
echo "================ gsplat  phone $FIELD/$DATE  res=2  (train+render+metrics, 15k, default densify, pp OFF) ================"
python src/run_reconstruction.py \
  dataset=phone plot=$FIELD date=$DATE \
  run_train=true run_render=true run_metrics=true \
  reconstruction.iterations=15000 reconstruction.resolution=2 \
  experiment_name=$EXP sfm_variant=$VARIANT
echo "  res=2 rc=$? -> results/reconstruction/phone/$FIELD/$DATE/$SUBTREE/$EXP"

echo ""
echo "================ STATUS ================"
echo "  res2 -> results/reconstruction/phone/$FIELD/$DATE/$SUBTREE/$EXP"
echo "  compare vs res1 baseline -> results/reconstruction/phone/$FIELD/$DATE/$SUBTREE/baseline/results.json"
PEAK_VRAM_MIB=$(sort -n "$VRAM_LOG" | tail -1)
echo "Peak VRAM: ${PEAK_VRAM_MIB} MiB ($(awk "BEGIN{printf \"%.2f\", ${PEAK_VRAM_MIB}/1024}") GiB)"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
