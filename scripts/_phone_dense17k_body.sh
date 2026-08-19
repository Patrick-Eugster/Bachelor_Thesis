# ------------------------------------------------------------------------------
# Shared body for the phone dense17k OPENCV A/B (sourced by the 4 wrappers
# phone_dense17k_{absgrad,noabsgrad}_{A,B}_job.sh, which set the SBATCH header +
# these vars before sourcing:
#   SESSIONS = bash array of "FIELD DATE" pairs (2 per wrapper, all <=96 imgs)
#   EXP      = experiment_name (dense17k | dense17k_noabsgrad)
#   ABSGRAD  = true | false
#   DGT      = densify_grad_threshold (0.0008 for absgrad, 0.0002 for default)
#
# The "more Gaussians" recipe: densify_until_iter 11000->17000 + iterations 15000->30000, on the
# opencv SfM arm (phone recon baseline). recon-only (train+render+metrics, NO seg — cropped opencv
# frames don't match the masks). resolution 1, pp off, opacity_prune UNCHANGED. gsplat engine.
# Writes to results/reconstruction/phone/<field>/<date>/opencv/vanilla_3dgs/$EXP.
#
# VRAM: absgrad+until17k is the proven 22.84/24 GiB point on a 96-img session. The no-absgrad arm's
# Gaussian count at until17k is UNMEASURED (default densified ~2.5x FEWER than absgrad at until11k, so
# likely fewer here too, but not certain) — the logger below reports the true peak either way.
#
# ⚠️ PREREQUISITE: opencv/ SfM variant present on Euler (already there from the groupX batch).
#   Preflight ABORTS if missing or if a target $EXP/ folder already exists.
# ------------------------------------------------------------------------------

VARIANT=opencv
SUBTREE=opencv/vanilla_3dgs
REPO=/cluster/project/cropsci/peugste/wheat3dgs

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
cd "$REPO"
nvidia-smi

# ── PREFLIGHT: fail LOUDLY if any target exists OR the opencv/ SfM input is missing ──
echo "=== PREFLIGHT ($EXP): opencv/ inputs present + $EXP/ targets fresh ==="
COLLIDE=0
for s in "${SESSIONS[@]}"; do
  set -- $s; FIELD=$1; DATE=$2
  OUT="$REPO/results/reconstruction/phone/$FIELD/$DATE/$SUBTREE/$EXP"
  IN="$REPO/input_plots/phone/$FIELD/$DATE/opencv/sparse/0"
  if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "  X EXISTS: $OUT"; COLLIDE=1
  elif [ ! -d "$IN" ]; then
    echo "  X MISSING INPUT (rsync the opencv/ folder up first!): $IN"; COLLIDE=1
  else
    echo "  ok (fresh, opencv input present): $OUT"
  fi
done
if [ "$COLLIDE" -ne 0 ]; then
  echo "ABORTING — a target exists or an opencv/ SfM input is missing (rsync it up first)."
  exit 1
fi

# ── peak VRAM + RAM loggers ──
VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!
RAM_LOG="$REPO/slurm_logs/ram_${SLURM_JOB_ID}.log"
CGROUP_PATH=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)
CGROUP_MEM_FILE="/sys/fs/cgroup${CGROUP_PATH}/memory.current"
( while true; do
    if [ -r "$CGROUP_MEM_FILE" ]; then cat "$CGROUP_MEM_FILE"
    else ps -u "$USER" -o rss= | awk '{sum+=$1} END{print sum*1024}'; fi
    sleep 5
  done ) > "$RAM_LOG" &
RAM_PID=$!
trap "kill $VRAM_PID $RAM_PID 2>/dev/null" EXIT

export WHEAT_RUN_REPORT="$REPO/slurm_logs/run_report_${SLURM_JOB_ID}.txt"

# ── run the sessions; DO NOT set -e (one failure must not kill the rest) ──
declare -A STATUS
for s in "${SESSIONS[@]}"; do
  set -- $s; FIELD=$1; DATE=$2
  echo ""
  echo "================================================================"
  echo "  opencv $EXP (absgrad=$ABSGRAD, dgt=$DGT)  ${FIELD} / ${DATE}   until 17k / 30k"
  echo "================================================================"
  python src/run_reconstruction.py \
    dataset=phone plot=$FIELD date=$DATE \
    run_train=true run_render=true run_metrics=true \
    reconstruction.resolution=1 \
    reconstruction.absgrad=$ABSGRAD reconstruction.densify_grad_threshold=$DGT \
    reconstruction.densify_until_iter=17000 reconstruction.iterations=30000 \
    experiment_name=$EXP sfm_variant=$VARIANT
  rc=$?
  if [ $rc -eq 0 ]; then STATUS["$FIELD/$DATE"]=OK; else STATUS["$FIELD/$DATE"]="FAIL(rc=$rc)"; fi
done

# ── summary ──
echo ""
echo "================ opencv $EXP PER-SESSION STATUS ================"
for s in "${SESSIONS[@]}"; do
  set -- $s; FIELD=$1; DATE=$2
  echo "  $FIELD/$DATE : ${STATUS[$FIELD/$DATE]}  -> results/reconstruction/phone/$FIELD/$DATE/$SUBTREE/$EXP"
done
PEAK_VRAM_MIB=$(sort -n "$VRAM_LOG" | tail -1)
PEAK_RAM_GIB=$(awk "BEGIN{printf \"%.2f\", $(sort -n "$RAM_LOG" | tail -1)/1024/1024/1024}")
echo "----------------------------------------------------------------"
echo "Peak VRAM: ${PEAK_VRAM_MIB} MiB ($(awk "BEGIN{printf \"%.2f\", ${PEAK_VRAM_MIB}/1024}") GiB)"
echo "Peak RAM:  ${PEAK_RAM_GIB} GiB (job cgroup)"
echo "================================================================"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
