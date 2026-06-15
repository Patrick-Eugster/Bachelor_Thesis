#!/bin/bash -l
#SBATCH -J fip_absgrad_A
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=5G          # 5G × 6 = 30G total — fits a 32 GB node with headroom.
#SBATCH --cpus-per-task=6
#SBATCH --time=18:00:00           # 3 plots × ~4.8h ≈ 14.5h; 18h margin. PLOT_TIMEOUT 6h × 3 = 18h hard cap.
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_absgrad_A_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_absgrad_A_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ── AbsGrad arm — SPLIT A: plots 461/462/463 ──
# Half of the full 7-plot rerun (fip_test_absgrad_all_job.sh), split so it can run on its own
# GPU node in parallel with SPLIT B (464/465/466/467). Each half fits under the 24h cap and on
# a 32 GB RAM node. Every plot ends with a CONSISTENT, valid results.json holding BOTH
# ours_15000 AND ours_30000 (Phase 2 forces --use_principal_point — the bug that wrecked the
# earlier resubmit). Full pipeline incl. step 5 (360 video).
#
# SINGLE job, sequential loop (NO array). Each plot's pipeline wrapped in `timeout` so an
# MPS/CUDA hang is killed after PLOT_TIMEOUT and the loop continues. 3 × 6h = 18h < 18h... =cap.
# gsplat engine + use_principal_point + AbsGS densification. gsplat already installed on Euler.

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi
cd /cluster/project/cropsci/peugste/wheat3dgs

EXP=test_absgrad_v2
THRESH=0.0008       # AbsGS threshold. If training OOMs / gaussian count explodes, raise to 0.001+.
PLOT_TIMEOUT=6h     # max wall per plot. Healthy plot ~5h → 6h won't kill a good run; a hang dies at 6h and the loop moves on.
PLOTS=(plot_461 plot_462 plot_463)

# One combined run-report for this half (run_reconstruction.py APPENDS each plot's report here).
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
#  LOOP over this half's plots — full pipeline + iter-eval each
# ════════════════════════════════════════════════════════════════════
for PLOT in "${PLOTS[@]}"; do
  echo ""
  echo "########################################################################"
  echo "#  ${PLOT}  FULL pipeline — gsplat + pp + AbsGrad (30k, thresh=${THRESH})"
  echo "########################################################################"

  # PHASE 1 — train + full pipeline (30k) incl. step 5 (360 video). timeout bounds any MPS/CUDA hang.
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

  # PHASE 2 — 15k-vs-30k iteration eval (NO retraining): a 30k run already saved iteration_15000,
  # and LR/densify schedules are anchored to fixed step counts, so iter-15000 of a 30k run ==
  # a standalone 15k run. Render 15k + 30k test views, metrics once → results.json gets both.
  SRC=input_plots/fip/${PLOT}
  MODEL=results/reconstruction/fip/${PLOT}/vanilla_3dgs/${EXP}
  if [ -d "${MODEL}/point_cloud/iteration_15000" ] && [ -d "${MODEL}/point_cloud/iteration_30000" ]; then
    echo ""
    echo "----- ITER-EVAL ${EXP} / ${PLOT} — render 15k + 30k test views, then metrics -----"
    # MUST pass --use_principal_point: this model was trained with pp, but standalone render.py
    # would otherwise DROP it. The flag defaults to False in ModelParams, and get_combined_args
    # merges with `if v != None`, so the cmdline default False overrides the True saved in cfg_args
    # → renders without pp → globally shifted vs GT → PSNR collapses to ~13 dB even though the
    # model + renders are fine. (This is the bug that wrecked 464/466/467 in job 3246353.)
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

# ── Peak usage summary (whole job) ──
echo ""
echo "========================================"
echo "  PEAK USAGE SUMMARY (SPLIT A)"
echo "========================================"
PEAK_VRAM_MIB=$(sort -n "$VRAM_LOG" | tail -1)
PEAK_RAM_BYTES=$(sort -n "$RAM_LOG" | tail -1)
PEAK_RAM_GIB=$(awk "BEGIN{printf \"%.2f\", $PEAK_RAM_BYTES/1024/1024/1024}")
echo "Peak VRAM: ${PEAK_VRAM_MIB} MiB ($(awk "BEGIN{printf \"%.2f\", ${PEAK_VRAM_MIB}/1024}") GiB)"
echo "Peak RAM:  ${PEAK_RAM_GIB} GiB  (node has 32 GB — watch this)"

# ── Combined run report (this half) ──
echo ""
echo "========================================"
echo "  COMBINED RUN REPORT (SPLIT A)"
echo "========================================"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT" || echo "(no run report written)"
