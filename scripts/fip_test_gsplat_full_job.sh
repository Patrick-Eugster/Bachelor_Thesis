#!/bin/bash -l
#SBATCH -J fip_test_gsplat_full
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00        # 7 plots × full pipeline + Test A. Round4 was ~20h for train/render/metrics only; seg+eval+gsplat-compile add time. Lower/split if your partition caps walltime.
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_test_gsplat_full_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_test_gsplat_full_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ── Script 2: full pipeline with the NEW engine (gsplat) + use_principal_point  (= Test B) ──
# Then Test A: render-engine comparison (gsplat vs diff-gaussian) on this trained model.
# No WHEAT_RENDERER export here → render() defaults to gsplat.
# (First run JIT-compiles gsplat CUDA kernels for Euler's GPU — one-time, then cached.)

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi
cd /cluster/project/cropsci/peugste/wheat3dgs

EXP=test_gsplat_full

# One combined run-report for this whole sbatch (all 7 plots append to it).
# run_reconstruction.py also writes a per-plot run_report.txt inside each experiment folder.
export WHEAT_RUN_REPORT=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/run_report_${SLURM_JOB_ID}.txt

# ── VRAM logger ──
VRAM_LOG=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/vram_${SLURM_JOB_ID}.log
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!
# ── RAM logger ──
RAM_LOG=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/ram_${SLURM_JOB_ID}.log
CGROUP_PATH=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)
CGROUP_MEM_FILE="/sys/fs/cgroup${CGROUP_PATH}/memory.current"
( while true; do
    if [ -r "$CGROUP_MEM_FILE" ]; then cat "$CGROUP_MEM_FILE"; else ps -u $USER -o rss= | awk '{s+=$1} END{print s*1024}'; fi
    sleep 5
  done ) > "$RAM_LOG" &
RAM_PID=$!
trap "kill $VRAM_PID $RAM_PID 2>/dev/null" EXIT

for PLOT in plot_461 plot_462 plot_463 plot_464 plot_465 plot_466 plot_467; do
  SRC=input_plots/fip/${PLOT}
  MODEL=results/reconstruction/fip/${PLOT}/vanilla_3dgs/${EXP}

  echo ""
  echo "========================================"
  echo "  ${PLOT}  FULL pipeline — gsplat + pp (30k)   [Test B]"
  echo "========================================"
  python src/run_reconstruction.py \
    plot=${PLOT} \
    run_train=true run_render=true run_metrics=true \
    run_seg=true run_eval=true run_eval_2d=true \
    reconstruction.iterations=30000 \
    reconstruction.use_principal_point=true \
    experiment_name=${EXP}

  echo ""
  echo "========================================"
  echo "  ${PLOT}  Test A — render diff (gsplat vs diff-gaussian)"
  echo "========================================"
  python src/reconstruction/compare_renderers.py -s ${SRC} -m ${MODEL} --iteration 30000
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

# ── Combined run report (all plots, step statuses + times + error tails) ──
echo ""
echo "========================================"
echo "  COMBINED RUN REPORT"
echo "========================================"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT" || echo "(no run report written)"
