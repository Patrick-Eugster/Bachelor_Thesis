#!/bin/bash -l
#SBATCH -J fip_test_absgrad
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00        # 7 plots × full pipeline (train+render+metrics+seg+eval+eval2d), 30k. No Test A here.
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_test_absgrad_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_test_absgrad_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ── AbsGrad arm: gsplat engine + use_principal_point + AbsGS densification ──
# Identical to test_gsplat_full EXCEPT: reconstruction.absgrad=true and a raised
# densify_grad_threshold (0.0008, gsplat's recommended value for absgrad, following the
# AbsGS paper). Compare this run (test_absgrad) against the existing test_gsplat_full
# (= gsplat + default signed-gradient densification) to see if AbsGrad beats vanilla.
# No WHEAT_RENDERER export → render() defaults to gsplat (absgrad requires gsplat).
# gsplat is already installed + kernel-cached on Euler from the gsplat benchmark → no new deps.

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi
cd /cluster/project/cropsci/peugste/wheat3dgs

EXP=test_absgrad
THRESH=0.0008    # AbsGS threshold. If training OOMs / Gaussian count explodes, raise to 0.001+.

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

PLOTS="plot_461 plot_462 plot_463 plot_464 plot_465 plot_466 plot_467"

# ════════════════════════════════════════════════════════════════════
#  PHASE 1 — train AbsGrad + full pipeline (30k) on every plot
# ════════════════════════════════════════════════════════════════════
for PLOT in $PLOTS; do
  echo ""
  echo "========================================"
  echo "  ${PLOT}  FULL pipeline — gsplat + pp + AbsGrad (30k, thresh=${THRESH})"
  echo "========================================"
  python src/run_reconstruction.py \
    plot=${PLOT} \
    run_train=true run_render=true run_metrics=true \
    run_seg=true run_eval=true run_eval_2d=true \
    reconstruction.iterations=30000 \
    reconstruction.use_principal_point=true \
    reconstruction.absgrad=true \
    reconstruction.densify_grad_threshold=${THRESH} \
    experiment_name=${EXP}
done

# ════════════════════════════════════════════════════════════════════
#  PHASE 2 — 15k-vs-30k iteration eval (NO retraining), for BOTH arms
#  A 30k run already saved point_cloud/iteration_15000, and the LR + densification
#  schedules are anchored to fixed step counts, so iter-15000 of a 30k run == a 15k run.
#  We render the 15k AND 30k test views and run metrics once -> each results.json gets
#  both ours_15000 and ours_30000. Covers the new AbsGrad arm and the existing vanilla
#  gsplat arm (test_gsplat_full), so you can decide 15k-vs-30k from one job, no 2nd sbatch.
#  (Standalone re-runnable version: scripts/fip_eval_iters_job.sh)
# ════════════════════════════════════════════════════════════════════
for EVAL_EXP in test_gsplat_full test_absgrad; do
  for PLOT in $PLOTS; do
    SRC=input_plots/fip/${PLOT}
    MODEL=results/reconstruction/fip/${PLOT}/vanilla_3dgs/${EVAL_EXP}
    if [ ! -d "${MODEL}/point_cloud/iteration_15000" ] || [ ! -d "${MODEL}/point_cloud/iteration_30000" ]; then
      echo ">>> SKIP iter-eval ${EVAL_EXP}/${PLOT}: missing iteration_15000 or iteration_30000"
      continue
    fi
    echo ""
    echo "========================================"
    echo "  ITER-EVAL ${EVAL_EXP} / ${PLOT}  — render 15k + 30k test views, then metrics"
    echo "========================================"
    python src/reconstruction/render.py -s ${SRC} -m ${MODEL} --iteration 15000 --skip_train
    python src/reconstruction/render.py -s ${SRC} -m ${MODEL} --iteration 30000 --skip_train
    python src/reconstruction/metrics.py -m ${MODEL}
    echo "----- results.json (${EVAL_EXP}/${PLOT}) — has ours_15000 AND ours_30000 -----"
    cat ${MODEL}/results.json
    echo ""
  done
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
