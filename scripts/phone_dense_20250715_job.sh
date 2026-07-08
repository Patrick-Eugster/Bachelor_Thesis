#!/bin/bash -l
#SBATCH -J phone_dense_20250715
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G      # 48 GB total (6 × 8G) — phone ~93 full-res images in CPU RAM
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00       # 30k iters + more Gaussians (longer densify) → slower than the 15k runs
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_dense_20250715_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_dense_20250715_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# Densification sweep for field_A/20250715 (train+render+metrics, NO seg): on top of the absgrad win,
# extend the densification window + refine longer, to see if phone sharpness keeps improving or hits a
# ceiling (wind non-rigidity / thin-awn Gaussian limit). ONE factor changed vs the *_absgrad runs: the
# densification window.
#   absgrad=true, densify_grad_threshold=0.0008   (same as *_absgrad — building on it)
#   densify_until_iter 11000 -> 17000             (longer densification = more Gaussians)
#   iterations 15000 -> 30000                     (now meaningful: densify to 17k, then 13k refine)
#   opacity_prune_threshold UNCHANGED (0.005)     (isolate the cause)
# Compare Gaussian count + WHOLE/ROI/MARKER metrics + sharpness vs *_absgrad (until 11k, 15k iters).
# metrics.py auto-runs the masked (ROI+marker) passes for COLMAP (run_reconstruction passes -s); Agisoft
# gets whole-image only.

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi
cd /cluster/project/cropsci/peugste/wheat3dgs

# ── peak VRAM + RAM loggers ── absgrad + longer densify can push the Gaussian count high; watch VRAM.
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
echo "  field_A / 20250715  COLMAP  absgrad + densify_until 17k / 30k iters"
echo "========================================"
python src/run_reconstruction.py \
  dataset=phone plot=field_A date=20250715 \
  run_train=true run_render=true run_metrics=true \
  reconstruction.absgrad=true reconstruction.densify_grad_threshold=0.0008 \
  reconstruction.densify_until_iter=17000 reconstruction.iterations=30000 \
  experiment_name=colmap_dense17k

echo "========================================"
echo "  field_A / 20250715  Agisoft  absgrad + densify_until 17k / 30k iters"
echo "========================================"
python src/run_reconstruction.py \
  dataset=phone plot=field_A date=20250715 \
  run_train=true run_render=true run_metrics=true \
  use_agisoft_sfm=true \
  reconstruction.absgrad=true reconstruction.densify_grad_threshold=0.0008 \
  reconstruction.densify_until_iter=17000 reconstruction.iterations=30000 \
  experiment_name=agisoft_dense17k

echo ""
echo "========================================"
PEAK_VRAM_MIB=$(sort -n "$VRAM_LOG" | tail -1)
PEAK_RAM_GIB=$(awk "BEGIN{printf \"%.2f\", $(sort -n "$RAM_LOG" | tail -1)/1024/1024/1024}")
echo "Peak VRAM: ${PEAK_VRAM_MIB} MiB ($(awk "BEGIN{printf \"%.2f\", ${PEAK_VRAM_MIB}/1024}") GiB)"
echo "Peak RAM:  ${PEAK_RAM_GIB} GiB (job cgroup)"
echo "========================================"
echo ""
echo "DONE. Compare vs the *_absgrad runs (until 11k, 15k iters):"
echo "  COLMAP  absgrad     : results/reconstruction/phone/field_A/20250715/vanilla_3dgs/colmap_absgrad/results.json"
echo "  COLMAP  dense17k    : results/reconstruction/phone/field_A/20250715/vanilla_3dgs/colmap_dense17k/results.json"
echo "  Agisoft absgrad     : results/reconstruction/phone/field_A/20250715/agisoft/vanilla_3dgs/agisoft_absgrad/results.json"
echo "  Agisoft dense17k    : results/reconstruction/phone/field_A/20250715/agisoft/vanilla_3dgs/agisoft_dense17k/results.json"
