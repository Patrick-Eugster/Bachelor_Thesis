#!/bin/bash -l
#SBATCH -J r360_A0715_pin30k
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=16G      # 192 GB total (12 × 16G) — pin30k Render360 RAM-OOM'd at the previous
#SBATCH --cpus-per-task=12     #   64 GB cap on the 4.47M-gaussian model; true peak was above 64, so we
                              #   give it 3x headroom. Raise mem-per-cpu if it still oom_kills.
#SBATCH --time=02:00:00        # render_360 alone: ~3 min on the 15k model; the 4.47M model is heavier but
                              #   still minutes -> 2h ceiling is ample even if slow.
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/r360_A0715_pin30k_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/r360_A0715_pin30k_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# A/0715 PINHOLE colmap_dense17k (30k, ~4.47M gaussians) — RENDER_360 ONLY, high-RAM retry.
# ----------------------------------------------------------------------------
# The full seg job finished Stage 4 (seg) but its Render360 (step 5) was OOM-killed by the SLURM cgroup
# at the 64 GB RAM cap (host RAM, NOT VRAM — VRAM was only 3.6 GB). This job re-runs ONLY render_360
# against the existing seg output, with 192 GB RAM. No mask-gen, no seg, no eval, no retrain.
# Reads results/.../vanilla_3dgs/colmap_dense17k/segmentation_3d/pin30k_yolov5_pertile.
# ============================================================================

FIELD=field_A
DATE=20250715
MODEL_EXP=colmap_dense17k
ITER=30000
SEG_EXP=pin30k_yolov5_pertile

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

# ── PREFLIGHT: model + seg output must exist ──
echo "=== PREFLIGHT ==="
MODEL="$REPO/results/reconstruction/phone/$FIELD/$DATE/vanilla_3dgs/$MODEL_EXP"
FAIL=0
[ -f "$MODEL/point_cloud/iteration_$ITER/point_cloud.ply" ] || { echo "  X no model iter $ITER: $MODEL"; FAIL=1; }
[ -f "$MODEL/segmentation_3d/$SEG_EXP/gaussians.ply" ]      || { echo "  X no seg output: $MODEL/segmentation_3d/$SEG_EXP"; FAIL=1; }
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: model + seg output present"

# ── peak VRAM + RAM loggers (RAM is the one we care about here) ──
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
echo "================ RENDER_360 pinhole $MODEL_EXP (iter $ITER, seg $SEG_EXP) — 192 GB RAM ================"
python src/run_reconstruction.py \
  dataset=phone plot=$FIELD date=$DATE \
  experiment_name=$MODEL_EXP \
  reconstruction.iterations=$ITER reconstruction.resolution=1 \
  reconstruction.use_principal_point=false \
  run_render_360=true \
  segmentation_3d.frustum_cull=true \
  segmentation_3d.exp_name=$SEG_EXP
rc=$?
echo "  render_360 rc=$rc -> $MODEL/segmentation_3d/$SEG_EXP/3DSeg"

echo ""
echo "================ STATUS ================"
PEAK_VRAM_MIB=$(sort -n "$VRAM_LOG" | tail -1)
PEAK_RAM_GIB=$(awk "BEGIN{printf \"%.2f\", $(sort -n "$RAM_LOG" | tail -1)/1024/1024/1024}")
echo "Peak VRAM: ${PEAK_VRAM_MIB} MiB"
echo "Peak RAM:  ${PEAK_RAM_GIB} GiB (job cgroup)   <- compare vs the 64 GB that OOM-killed it"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
