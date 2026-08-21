#!/bin/bash -l
#SBATCH -J fip_absgrad_fix
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G          # 8G × 6 = 48G — back to the original single-node budget (the 32G
#SBATCH --cpus-per-task=6         # cap was only for the two-node split). v2 peaked ~30G, so 48G is ample.
#SBATCH --time=23:00:00           # 2 plots × ~6.5h ≈ 13h; 10h timeout × 2 = 20h < 23h.
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_absgrad_fix_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_absgrad_fix_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ── AbsGrad arm — SEG-FIX rerun of plots 463 + 467 ──
# In the test_absgrad_v2 all-7 rerun (jobs 3354206/3354217), these two — the DENSEST absgrad
# models — had their flashsplat segmentation KILLED ~6h in (no gaussians.ply / all_obj_labels /
# eval_2d), so they're missing seg + 2D metrics. Their RECON metrics (results.json, 15k+30k)
# under test_absgrad_v2 are fine; only seg is missing.
#
# This redoes the FULL pipeline for just those two under a NEW exp name (test_absgrad_v3) so
# nothing mixes with the partial v2 seg folder, with TWO changes vs the v2 job:
#   (1) run_render_360=false  — skip the 360 video (Euler ffmpeg libopenh264.so.5 is broken anyway).
#   (2) PLOT_TIMEOUT=10h       — the v2 6h cap killed seg mid-run; 10h gives healthy seg (~5h) +
#                                train (~0.85h) plenty of headroom while still bounding an MPS hang.
#
# CONFIRMED (sacct + .out grep): the v2 kill was the 6h PLOT_TIMEOUT (WARNING lines for both
# plot_463 and plot_467), NOT an OOM — both v2 jobs ended State=COMPLETED. But v2 job B did peak
# at MaxRSS ~30G (98% of its 30G two-node alloc), so this single-node run goes back to 48G.
#
# SINGLE job, sequential loop (NO array). gsplat engine + use_principal_point + AbsGS densification.

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi
cd /cluster/project/cropsci/peugste/wheat3dgs

EXP=test_absgrad_v3   # NEW name — keeps the partial v2 seg for 463/467 untouched, no mixing.
THRESH=0.0008         # AbsGS threshold (same as v2).
PLOT_TIMEOUT=10h      # raised from 6h: v2 seg needed ~5h on these dense plots and got cut at 6h.
PLOTS=(plot_463 plot_467)

# One combined run-report for this fix (run_reconstruction.py APPENDS each plot's report here).
export WHEAT_RUN_REPORT=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/run_report_${SLURM_JOB_ID}.txt

# ── VRAM logger (whole job) ──
VRAM_LOG=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/vram_${SLURM_JOB_ID}.log
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!
# ── RAM logger (whole job) — watch this vs the 32 GB node limit ──
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
#  LOOP over the 2 fix plots — full pipeline (NO 360 video) + iter-eval
# ════════════════════════════════════════════════════════════════════
for PLOT in "${PLOTS[@]}"; do
  echo ""
  echo "########################################################################"
  echo "#  ${PLOT}  FULL pipeline (no video) — gsplat + pp + AbsGrad (30k, thresh=${THRESH})"
  echo "########################################################################"

  # PHASE 1 — train + seg + eval (NO render_360). timeout bounds any MPS/CUDA hang to PLOT_TIMEOUT.
  timeout ${PLOT_TIMEOUT} python src/run_reconstruction.py \
    plot=${PLOT} \
    run_train=true run_render=true run_metrics=true \
    run_seg=true run_render_360=false run_eval=true run_eval_2d=true \
    reconstruction.iterations=30000 \
    reconstruction.use_principal_point=true \
    reconstruction.absgrad=true \
    reconstruction.densify_grad_threshold=${THRESH} \
    experiment_name=${EXP}
  RC=$?
  [ $RC -eq 124 ] && echo ">>> WARNING: ${PLOT} pipeline HIT THE ${PLOT_TIMEOUT} TIMEOUT (likely MPS/CUDA hang or seg too slow) — moving on to next plot."

  # PHASE 2 — 15k-vs-30k iteration eval (NO retraining). MUST pass --use_principal_point (else
  # standalone render.py drops it → pp-shifted renders → ~13 dB; the get_combined_args bool footgun).
  SRC=input_plots/fip/${PLOT}
  MODEL=results/reconstruction/fip/${PLOT}/vanilla_3dgs/${EXP}
  if [ -d "${MODEL}/point_cloud/iteration_15000" ] && [ -d "${MODEL}/point_cloud/iteration_30000" ]; then
    echo ""
    echo "----- ITER-EVAL ${EXP} / ${PLOT} — render 15k + 30k test views, then metrics -----"
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

# ── Peak usage summary ──
echo ""
echo "========================================"
echo "  PEAK USAGE SUMMARY (SEG-FIX 463/467)"
echo "========================================"
PEAK_VRAM_MIB=$(sort -n "$VRAM_LOG" | tail -1)
PEAK_RAM_BYTES=$(sort -n "$RAM_LOG" | tail -1)
PEAK_RAM_GIB=$(awk "BEGIN{printf \"%.2f\", $PEAK_RAM_BYTES/1024/1024/1024}")
echo "Peak VRAM: ${PEAK_VRAM_MIB} MiB ($(awk "BEGIN{printf \"%.2f\", ${PEAK_VRAM_MIB}/1024}") GiB)"
echo "Peak RAM:  ${PEAK_RAM_GIB} GiB  (48G alloc this run; v2 peaked ~30G)"

# ── Combined run report ──
echo ""
echo "========================================"
echo "  COMBINED RUN REPORT (SEG-FIX 463/467)"
echo "========================================"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT" || echo "(no run report written)"
