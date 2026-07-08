#!/bin/bash -l
#SBATCH -J phone_bench_20250715
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G      # 48 GB total (6 × 8G) — phone ~93 full-res images in CPU RAM
#SBATCH --cpus-per-task=6
#SBATCH --time=8:00:00
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_bench_20250715_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_bench_20250715_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# COLMAP-vs-Agisoft 3DGS RECONSTRUCTION benchmark for field_A/20250715 (train+render+metrics, NO seg).
# Answers "is Agisoft SfM better than our COLMAP for 3DGS?" — the COLMAP side (phone_sahi) scored PSNR
# 15.44 / SSIM 0.23. Both sides run the same code/params, differing only in the SfM (use_agisoft_sfm).
# No masks needed (reconstruction-only). Compare the two results.json PSNR/SSIM/LPIPS afterwards.

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi
cd /cluster/project/cropsci/peugste/wheat3dgs

# ── peak VRAM + RAM loggers ── sample every 5s to a log; peak = max at the end (we care about the
# train's memory footprint on this phone data). VRAM via nvidia-smi; RAM via this job's cgroup v2.
VRAM_LOG=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/vram_${SLURM_JOB_ID}.log
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    sleep 5
  done ) > "$VRAM_LOG" &
VRAM_PID=$!

RAM_LOG=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/ram_${SLURM_JOB_ID}.log
CGROUP_PATH=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)
CGROUP_MEM_FILE="/sys/fs/cgroup${CGROUP_PATH}/memory.current"
( while true; do
    if [ -r "$CGROUP_MEM_FILE" ]; then
      cat "$CGROUP_MEM_FILE"                                      # bytes, this job only
    else
      ps -u $USER -o rss= | awk '{sum+=$1} END{print sum*1024}'   # fallback: sum RSS (kB→bytes)
    fi
    sleep 5
  done ) > "$RAM_LOG" &
RAM_PID=$!

trap "kill $VRAM_PID $RAM_PID 2>/dev/null" EXIT

echo "========================================"
echo "  field_A / 20250715  (COLMAP)"
echo "========================================"
python src/run_reconstruction.py \
  dataset=phone plot=field_A date=20250715 \
  run_train=true run_render=true run_metrics=true \
  experiment_name=colmap_bench

echo "========================================"
echo "  field_A / 20250715  (Agisoft)"
echo "========================================"
python src/run_reconstruction.py \
  dataset=phone plot=field_A date=20250715 \
  run_train=true run_render=true run_metrics=true \
  use_agisoft_sfm=true experiment_name=agisoft_bench

echo ""
echo "========================================"
PEAK_VRAM_MIB=$(sort -n "$VRAM_LOG" | tail -1)
PEAK_RAM_BYTES=$(sort -n "$RAM_LOG" | tail -1)
PEAK_RAM_GIB=$(awk "BEGIN{printf \"%.2f\", $PEAK_RAM_BYTES/1024/1024/1024}")
echo "Peak VRAM: ${PEAK_VRAM_MIB} MiB ($(awk "BEGIN{printf \"%.2f\", ${PEAK_VRAM_MIB}/1024}") GiB)"
echo "Peak RAM:  ${PEAK_RAM_GIB} GiB (job cgroup, not whole node)"
echo "========================================"
echo ""
echo "DONE. Compare:"
echo "  COLMAP : results/reconstruction/phone/field_A/20250715/vanilla_3dgs/colmap_bench/results.json"
echo "  Agisoft: results/reconstruction/phone/field_A/20250715/agisoft/vanilla_3dgs/agisoft_bench/results.json"
