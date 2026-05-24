#!/bin/bash -l
#SBATCH -J phone_bench_agisoft_fast
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G     # 48 GB total (6 × 8G)
#SBATCH --cpus-per-task=6
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_bench_agisoft_fast_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_bench_agisoft_fast_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
which python
python --version
conda info --envs
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

# ── RAM logger ── reads this job's cgroup (v2), falls back to summing RSS of our processes
RAM_LOG=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/ram_${SLURM_JOB_ID}.log
CGROUP_PATH=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)
CGROUP_MEM_FILE="/sys/fs/cgroup${CGROUP_PATH}/memory.current"
( while true; do
    if [ -r "$CGROUP_MEM_FILE" ]; then
      cat "$CGROUP_MEM_FILE"          # bytes, my-job-only
    else
      ps -u $USER -o rss= | awk '{sum+=$1} END{print sum*1024}'  # fallback: kB→bytes
    fi
    sleep 5
  done ) > "$RAM_LOG" &
RAM_PID=$!

trap "kill $VRAM_PID $RAM_PID 2>/dev/null" EXIT

for args in "field_A 20250609" "field_A 20250618" "field_D 20250523" "field_D 20250530"; do
  field=$(echo $args | cut -d' ' -f1)
  date=$(echo $args | cut -d' ' -f2)
  echo ""
  echo "========================================"
  echo "  $field / $date  (Agisoft)"
  echo "========================================"
  python src/run_reconstruction.py \
    dataset=phone plot=$field date=$date \
    run_train=true run_render=true run_metrics=true \
    use_agisoft_sfm=true experiment_name=agisoft_bench
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
echo "Peak RAM:  ${PEAK_RAM_GIB} GiB (job cgroup, not whole node)"
echo "Full logs: $VRAM_LOG  /  $RAM_LOG"