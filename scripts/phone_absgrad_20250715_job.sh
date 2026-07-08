#!/bin/bash -l
#SBATCH -J phone_absgrad_20250715
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G      # 48 GB total (6 × 8G) — phone ~93 full-res images in CPU RAM
#SBATCH --cpus-per-task=6
#SBATCH --time=8:00:00
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_absgrad_20250715_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_absgrad_20250715_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# AbsGrad A/B for field_A/20250715 (train+render+metrics, NO seg) — does AbsGS densification make the
# phone reconstruction sharper? The baselines (absgrad OFF) already exist: colmap_bench / agisoft_bench
# (PSNR 15.45 / 15.41, 1.24M Gaussians — UNDER-densified vs FIP's 2.33M). Here we run absgrad ON for
# BOTH SfMs, everything else default (15k iters, densify_until 11k).
# NOTE: absgrad=true REQUIRES densify_grad_threshold ~0.0008 (absgrad magnitudes are ~4x larger; the
# default 0.0002 would over-densify and OOM) — so the two knobs move together as the "absgrad regime".
# Compare Gaussian count + PSNR + visual sharpness vs the *_bench baselines.

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi
cd /cluster/project/cropsci/peugste/wheat3dgs

# ── peak VRAM + RAM loggers ── sample every 5s; peak at the end. absgrad adds Gaussians → watch VRAM.
VRAM_LOG=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/vram_${SLURM_JOB_ID}.log
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!
RAM_LOG=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/ram_${SLURM_JOB_ID}.log
CGROUP_PATH=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)
CGROUP_MEM_FILE="/sys/fs/cgroup${CGROUP_PATH}/memory.current"
( while true; do
    if [ -r "$CGROUP_MEM_FILE" ]; then cat "$CGROUP_MEM_FILE"
    else ps -u $USER -o rss= | awk '{sum+=$1} END{print sum*1024}'; fi
    sleep 5
  done ) > "$RAM_LOG" &
RAM_PID=$!
trap "kill $VRAM_PID $RAM_PID 2>/dev/null" EXIT

echo "========================================"
echo "  field_A / 20250715  COLMAP + absgrad"
echo "========================================"
python src/run_reconstruction.py \
  dataset=phone plot=field_A date=20250715 \
  run_train=true run_render=true run_metrics=true \
  reconstruction.absgrad=true reconstruction.densify_grad_threshold=0.0008 \
  experiment_name=colmap_absgrad

echo "========================================"
echo "  field_A / 20250715  Agisoft + absgrad"
echo "========================================"
python src/run_reconstruction.py \
  dataset=phone plot=field_A date=20250715 \
  run_train=true run_render=true run_metrics=true \
  use_agisoft_sfm=true \
  reconstruction.absgrad=true reconstruction.densify_grad_threshold=0.0008 \
  experiment_name=agisoft_absgrad

echo ""
echo "========================================"
PEAK_VRAM_MIB=$(sort -n "$VRAM_LOG" | tail -1)
PEAK_RAM_GIB=$(awk "BEGIN{printf \"%.2f\", $(sort -n "$RAM_LOG" | tail -1)/1024/1024/1024}")
echo "Peak VRAM: ${PEAK_VRAM_MIB} MiB ($(awk "BEGIN{printf \"%.2f\", ${PEAK_VRAM_MIB}/1024}") GiB)"
echo "Peak RAM:  ${PEAK_RAM_GIB} GiB (job cgroup)"
echo "========================================"
echo ""
echo "DONE. Compare vs baselines (absgrad off):"
echo "  COLMAP  off: results/reconstruction/phone/field_A/20250715/vanilla_3dgs/colmap_bench/results.json"
echo "  COLMAP  on : results/reconstruction/phone/field_A/20250715/vanilla_3dgs/colmap_absgrad/results.json"
echo "  Agisoft off: results/reconstruction/phone/field_A/20250715/agisoft/vanilla_3dgs/agisoft_bench/results.json"
echo "  Agisoft on : results/reconstruction/phone/field_A/20250715/agisoft/vanilla_3dgs/agisoft_absgrad/results.json"
