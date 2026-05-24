#!/bin/bash -l
#SBATCH -J fip_paper_bench_retrain
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=6
#SBATCH --time=20:00:00       # ~1h40m per plot × 7 plots = ~12h; 20h gives safe headroom
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_paper_bench_retrain_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_paper_bench_retrain_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi
cd /cluster/project/cropsci/peugste/wheat3dgs

# ── VRAM logger ──
VRAM_LOG=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/vram_${SLURM_JOB_ID}.log
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    sleep 5
  done ) > "$VRAM_LOG" &
VRAM_PID=$!

# ── RAM logger ──
RAM_LOG=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/ram_${SLURM_JOB_ID}.log
CGROUP_PATH=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)
CGROUP_MEM_FILE="/sys/fs/cgroup${CGROUP_PATH}/memory.current"
( while true; do
    if [ -r "$CGROUP_MEM_FILE" ]; then
      cat "$CGROUP_MEM_FILE"
    else
      ps -u $USER -o rss= | awk '{sum+=$1} END{print sum*1024}'
    fi
    sleep 5
  done ) > "$RAM_LOG" &
RAM_PID=$!

trap "kill $VRAM_PID $RAM_PID 2>/dev/null" EXIT

for plot in plot_461 plot_462 plot_463 plot_464 plot_465 plot_466 plot_467; do
  echo ""
  echo "========================================"
  echo "  $plot  (FIP retrain, 30k iters, fixed eval split)"
  echo "========================================"
  python src/run_reconstruction.py \
    plot=$plot \
    run_train=true run_render=true run_metrics=true \
    reconstruction.iterations=30000 \
    experiment_name=paper_bench_30k
done

# ── Peak summary ──
echo ""
echo "========================================"
echo "  PEAK USAGE SUMMARY"
echo "========================================"
PEAK_VRAM_MIB=$(sort -n "$VRAM_LOG" | tail -1)
PEAK_RAM_BYTES=$(sort -n "$RAM_LOG" | tail -1)
PEAK_RAM_GIB=$(awk "BEGIN{printf \"%.2f\", $PEAK_RAM_BYTES/1024/1024/1024}")
echo "Peak VRAM: ${PEAK_VRAM_MIB} MiB ($(awk "BEGIN{printf \"%.2f\", ${PEAK_VRAM_MIB}/1024}") GiB)"
echo "Peak RAM:  ${PEAK_RAM_GIB} GiB"