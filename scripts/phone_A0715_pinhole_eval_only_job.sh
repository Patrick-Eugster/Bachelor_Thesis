#!/bin/bash -l
#SBATCH -J eval_A0715_pin
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G       # 64 GB total — eval decodes full-res GT + 2DSeg label maps (RAM-heavy);
#SBATCH --cpus-per-task=8      #   the pin30k 4.47M-gaussian model's eval peaked ~21 GB RAM previously.
#SBATCH --time=01:00:00        # eval-only: both prior runs' Eval reached ~1 min before the id2rgb crash;
                               #   13 test views x 2 experiments -> a few minutes; 1h ceiling is ample.
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/eval_A0715_pin_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/eval_A0715_pin_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# A/0715 PINHOLE — EVAL-ONLY re-run after the id2rgb / max_num_obj fix.
# ----------------------------------------------------------------------------
# The pin15k (baseline) and pin30k (colmap_dense17k) 3D-seg runs both finished Stage 4 (seg) fine but
# CRASHED in step 6 (Eval) on `ValueError: ID should be in range(0, max_num_obj)` — the id2rgb palette
# capped at 999 while phone plots have thousands of head IDs. That guard is now fixed (image_helper.py:
# id2rgb only rejects negatives; the golden-ratio hue is valid for any id). This job re-runs ONLY the
# eval steps against the EXISTING seg output — no mask-gen, no seg, no retrain.
#
# run_render_360=false ON PURPOSE: pin30k's Render360 RAM-OOM'd (cgroup oom_kill @64 GB) on the 4.47M
# model — that is a separate RAM issue, NOT this bug, so we don't re-trigger it here. Eval itself only
# needs ~4 GB VRAM / ~21 GB RAM and completes.
#
# Reads the seg output at results/.../vanilla_3dgs/<MODEL_EXP>/segmentation_3d/<SEG_EXP>. Writes eval
# overlays + binary masks (step 6) and eval_2d metrics (step 6b, GT-aligned on pinhole 4032 frames).
# ============================================================================

FIELD=field_A
DATE=20250715
# two pinhole arms: "MODEL_EXP ITER SEG_EXP MASK_EXP"
RUNS=(
  "baseline        15000 pin15k_yolov5_pertile yolov5_pertile_pin15k"
  "colmap_dense17k 30000 pin30k_yolov5_pertile yolov5_pertile_pin30k"
)

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

# ── PREFLIGHT: each model + its seg output must already exist ──
echo "=== PREFLIGHT ==="
FAIL=0
for r in "${RUNS[@]}"; do
  set -- $r; MODEL_EXP=$1; ITER=$2; SEG_EXP=$3
  MODEL="$REPO/results/reconstruction/phone/$FIELD/$DATE/vanilla_3dgs/$MODEL_EXP"
  [ -f "$MODEL/point_cloud/iteration_$ITER/point_cloud.ply" ] || { echo "  X no model iter $ITER: $MODEL"; FAIL=1; }
  [ -f "$MODEL/segmentation_3d/$SEG_EXP/gaussians.ply" ]      || { echo "  X no seg output: $MODEL/segmentation_3d/$SEG_EXP"; FAIL=1; }
done
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: both models + seg outputs present"

# ── peak VRAM logger ──
VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!; trap "kill $VRAM_PID 2>/dev/null" EXIT
export WHEAT_RUN_REPORT="$REPO/slurm_logs/run_report_${SLURM_JOB_ID}.txt"

# ── run eval (+eval_2d) for each arm; do NOT set -e (one failure must not kill the other) ──
declare -A STATUS
for r in "${RUNS[@]}"; do
  set -- $r; MODEL_EXP=$1; ITER=$2; SEG_EXP=$3; MASK_EXP=$4
  echo ""
  echo "================================================================"
  echo "  EVAL pinhole $MODEL_EXP (iter $ITER) -> seg $SEG_EXP"
  echo "================================================================"
  python src/run_reconstruction.py \
    dataset=phone plot=$FIELD date=$DATE \
    experiment_name=$MODEL_EXP \
    reconstruction.iterations=$ITER reconstruction.resolution=1 \
    reconstruction.use_principal_point=false \
    run_eval=true run_eval_2d=true \
    segmentation_3d.frustum_cull=true \
    segmentation_3d.detection_method=yolo_sam_v1 \
    segmentation_3d.mask_gen_experiment=$MASK_EXP \
    segmentation_3d.exp_name=$SEG_EXP
  rc=$?
  if [ $rc -eq 0 ]; then STATUS["$MODEL_EXP"]=OK; else STATUS["$MODEL_EXP"]="FAIL(rc=$rc)"; fi
done

echo ""
echo "================ EVAL-ONLY PER-ARM STATUS ================"
for r in "${RUNS[@]}"; do
  set -- $r; MODEL_EXP=$1; SEG_EXP=$3
  echo "  $MODEL_EXP : ${STATUS[$MODEL_EXP]}  -> results/.../vanilla_3dgs/$MODEL_EXP/segmentation_3d/$SEG_EXP/{eval,eval_2d}"
done
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
