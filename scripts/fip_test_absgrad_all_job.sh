#!/bin/bash -l
#SBATCH -J fip_absgrad_all
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00         # ONE job, loops ALL 7 plots sequentially. ~4.7h/plot × 7 ≈ 33h; 48h gives margin.
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_absgrad_all_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_absgrad_all_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ── AbsGrad arm — FULL CLEAN RERUN of ALL 7 FIP plots ──
# Supersedes both fip_test_absgrad_job.sh (the original 48h batch that died on an MPS hang)
# and fip_test_absgrad_resubmit_job.sh (only redid 464/466/467). This redoes EVERYTHING for
# all 7 plots so every plot ends with a CONSISTENT, valid results.json holding BOTH
# ours_15000 AND ours_30000.
#
# WHY a full rerun: the prior results were a mess — 461/462/463 only had a valid 30k metric
# (their Phase 2 never ran, original job died at the 21h MPS hang), and 464/466/467 had both
# 15k+30k but BOTH were garbage (~13 dB) because the standalone render.py in Phase 2 dropped
# --use_principal_point (get_combined_args bool footgun → pp-shifted renders). This script
# fixes that (Phase 2 now forces --use_principal_point) and runs all 7 uniformly.
#
# SINGLE job, sequential loop over all 7 plots (NO job array — this account can't run array
# tasks). Hang protection: each plot's pipeline is wrapped in `timeout` so an MPS/CUDA hang
# is KILLED after PLOT_TIMEOUT and the loop continues to the next plot, instead of one zombie
# step (like the 21h export hang) eating the whole job's wall budget. 7 × 6h = 42h < 48h cap.
#
# gsplat engine + use_principal_point + AbsGS densification. No WHEAT_RENDERER export →
# render() defaults to gsplat (absgrad requires gsplat). gsplat already installed on Euler.

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi
cd /cluster/project/cropsci/peugste/wheat3dgs

EXP=test_absgrad
THRESH=0.0008       # AbsGS threshold. If training OOMs / gaussian count explodes, raise to 0.001+.
PLOT_TIMEOUT=6h     # max wall per plot. Healthy plot ~4.7h → 6h won't kill a good run; a hang dies at 6h and the loop moves on. 7×6h=42h < 48h job cap.
PLOTS=(plot_461 plot_462 plot_463 plot_464 plot_465 plot_466 plot_467)

# One combined run-report for all plots (run_reconstruction.py APPENDS each plot's report here).
export WHEAT_RUN_REPORT=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/run_report_${SLURM_JOB_ID}.txt

# ── VRAM logger (whole job) ──
VRAM_LOG=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/vram_${SLURM_JOB_ID}.log
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!
# ── RAM logger (whole job) ──
RAM_LOG=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/ram_${SLURM_JOB_ID}.log
CGROUP_PATH=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)
CGROUP_MEM_FILE="/sys/fs/cgroup${CGROUP_PATH}/memory.current"
( while true; do
    if [ -r "$CGROUP_MEM_FILE" ]; then cat "$CGROUP_MEM_FILE"; else ps -u $USER -o rss= | awk '{s+=$1} END{print s*1024}'; fi
    sleep 5
  done ) > "$RAM_LOG" &
RAM_PID=$!
trap "kill $VRAM_PID $RAM_PID 2>/dev/null" EXIT

# ════════════════════════════════════════════════════════════════════
#  LOOP over all 7 plots — full pipeline + iter-eval each
# ════════════════════════════════════════════════════════════════════
for PLOT in "${PLOTS[@]}"; do
  echo ""
  echo "########################################################################"
  echo "#  ${PLOT}  FULL pipeline — gsplat + pp + AbsGrad (30k, thresh=${THRESH})"
  echo "########################################################################"

  # PHASE 1 — train + full pipeline (30k). timeout bounds any MPS/CUDA hang to PLOT_TIMEOUT.
  timeout ${PLOT_TIMEOUT} python src/run_reconstruction.py \
    plot=${PLOT} \
    run_train=true run_render=true run_metrics=true \
    run_seg=true run_render_360=true run_eval=true run_eval_2d=true \
    reconstruction.iterations=30000 \
    reconstruction.use_principal_point=true \
    reconstruction.absgrad=true \
    reconstruction.densify_grad_threshold=${THRESH} \
    experiment_name=${EXP}
  RC=$?
  [ $RC -eq 124 ] && echo ">>> WARNING: ${PLOT} pipeline HIT THE ${PLOT_TIMEOUT} TIMEOUT (likely MPS/CUDA hang) — moving on to next plot."

  # PHASE 2 — 15k-vs-30k iteration eval (NO retraining): a 30k run already saved
  # iteration_15000, and LR/densify schedules are anchored to fixed step counts, so iter-15000
  # of a 30k run == a standalone 15k run. Render 15k + 30k test views, metrics once → results.json
  # gets both ours_15000 and ours_30000. (Standalone version: scripts/fip_eval_iters_job.sh)
  SRC=input_plots/fip/${PLOT}
  MODEL=results/reconstruction/fip/${PLOT}/vanilla_3dgs/${EXP}
  if [ -d "${MODEL}/point_cloud/iteration_15000" ] && [ -d "${MODEL}/point_cloud/iteration_30000" ]; then
    echo ""
    echo "----- ITER-EVAL ${EXP} / ${PLOT} — render 15k + 30k test views, then metrics -----"
    # MUST pass --use_principal_point: this model was trained with pp, but standalone render.py
    # would otherwise DROP it. The flag defaults to False in ModelParams, and get_combined_args
    # merges with `if v != None`, so the cmdline default False overrides the True saved in cfg_args
    # → renders without pp → globally shifted vs GT → PSNR collapses to ~13 dB even though the
    # model + renders are fine. The Phase-1 pipeline (run_reconstruction.py) passes this flag; the
    # standalone calls here must too. (This is the bug that wrecked 464/466/467 in job 3246353.)
    timeout ${PLOT_TIMEOUT} python src/reconstruction/render.py -s ${SRC} -m ${MODEL} --use_principal_point --iteration 15000 --skip_train
    timeout ${PLOT_TIMEOUT} python src/reconstruction/render.py -s ${SRC} -m ${MODEL} --use_principal_point --iteration 30000 --skip_train
    timeout ${PLOT_TIMEOUT} python src/reconstruction/metrics.py -m ${MODEL}
    echo "----- results.json (${EXP}/${PLOT}) — has ours_15000 AND ours_30000 -----"
    cat ${MODEL}/results.json
    echo ""
  else
    echo ">>> SKIP iter-eval ${EXP}/${PLOT}: missing iteration_15000 or iteration_30000 (train likely failed)"
  fi
done

# ── Peak usage summary (whole job, across all plots) ──
echo ""
echo "========================================"
echo "  PEAK USAGE SUMMARY (all plots)"
echo "========================================"
PEAK_VRAM_MIB=$(sort -n "$VRAM_LOG" | tail -1)
PEAK_RAM_BYTES=$(sort -n "$RAM_LOG" | tail -1)
PEAK_RAM_GIB=$(awk "BEGIN{printf \"%.2f\", $PEAK_RAM_BYTES/1024/1024/1024}")
echo "Peak VRAM: ${PEAK_VRAM_MIB} MiB ($(awk "BEGIN{printf \"%.2f\", ${PEAK_VRAM_MIB}/1024}") GiB)"
echo "Peak RAM:  ${PEAK_RAM_GIB} GiB"

# ── Combined run report (all plots: per-step status + times + error tails) ──
echo ""
echo "========================================"
echo "  COMBINED RUN REPORT (all plots)"
echo "========================================"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT" || echo "(no run report written)"
