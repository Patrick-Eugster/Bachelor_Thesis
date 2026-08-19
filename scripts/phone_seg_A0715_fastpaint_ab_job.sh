#!/bin/bash -l
#SBATCH -J seg_A0715_fastpaint_ab
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G       # 64 GB total — phone 3D seg decodes many full-res masks (RAM-heavy)
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00        # two agisoft-15k segs back to back (~8.5h each, no eval/render) fit under the 24h cap
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_fastpaint_ab_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_fastpaint_ab_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# A/B for the FAST BBOX COMMIT-PAINT optimization in run_3d_seg.py.
# Purpose: (1) measure the wall-time saved by painting only each head's 2D bbox into the 2DSeg map
# instead of sweeping the full 12MP frame per (head,camera); (2) PROVE it is lossless (byte-identical
# segmentation output). Runs the SAME model + SAME masks twice, changing ONLY the paint path:
#   arm BASELINE : WHEAT_SEG_NO_FAST_PAINT=1  -> old full-frame paint (the ~27% commit_paint bucket)
#   arm FAST     : (default)                  -> new bbox paint
# Both arms: WHEAT_SEG_TIMING=1 so the "commit paint (CPU 2DSeg)" line is printed for each -> read the
# two numbers to get the delta. At the end the two all_obj_labels.pth are md5-compared: MUST MATCH
# (lossless). We use the AGISOFT 15k model because two of its segs (~8.5h each) fit one 24h job; the
# opencv model (~13h/seg) would need two separate jobs. NO eval / NO render_360 (timing + md5 only).
# ============================================================================

FIELD=field_A
DATE=20250715
VARIANT=agisoft
MODEL_EXP=baseline
ITER=15000
MASK_EXP=yolov5_pertile_agi15k          # reuse the SAME masks the agi15k_groundfix run used
SEG_BASE=agi15k_paint_baseline          # arm 1 (full-frame paint)
SEG_FAST=agi15k_paint_fast              # arm 2 (bbox paint)

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"

MODEL="$REPO/results/reconstruction/phone/$FIELD/$DATE/$VARIANT/vanilla_3dgs/$MODEL_EXP"
MASK_OUT="$REPO/results/mask_generation/phone/$FIELD/$DATE/$VARIANT/yolo_sam_v1/$MASK_EXP"

echo "=== PREFLIGHT ==="
FAIL=0
[ -f "$MODEL/point_cloud/iteration_$ITER/point_cloud.ply" ] || { echo "  X no model at iter $ITER: $MODEL"; FAIL=1; }
[ -n "$(ls -A "$MASK_OUT/masks" 2>/dev/null)" ] || { echo "  X no masks at $MASK_OUT/masks (run the agi15k seg first)"; FAIL=1; }
for S in "$SEG_BASE" "$SEG_FAST"; do
  D="$MODEL/segmentation_3d/$S"
  [ -d "$D" ] && [ -n "$(ls -A "$D" 2>/dev/null)" ] && { echo "  X seg target exists: $D"; FAIL=1; }
done
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: agisoft model + masks present, both seg targets fresh"

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!; trap "kill $VRAM_PID 2>/dev/null" EXIT

run_arm () {
  # $1 = SEG_EXP, $2 = value for WHEAT_SEG_NO_FAST_PAINT ("" = fast/default, "1" = baseline full-frame)
  local SEG_EXP="$1"; local NOFAST="$2"
  local LOG="$REPO/slurm_logs/fastpaint_${SLURM_JOB_ID}_${SEG_EXP}.log"
  echo ""; echo "======== ARM: $SEG_EXP  (WHEAT_SEG_NO_FAST_PAINT='${NOFAST}') ========"
  WHEAT_SEG_TIMING=1 WHEAT_SEG_NO_FAST_PAINT="$NOFAST" \
  python src/run_reconstruction.py \
    dataset=phone plot=$FIELD date=$DATE sfm_variant=$VARIANT \
    experiment_name=$MODEL_EXP \
    reconstruction.iterations=$ITER reconstruction.resolution=1 \
    reconstruction.use_principal_point=false \
    run_seg=true \
    segmentation_3d.detection_method=yolo_sam_v1 \
    segmentation_3d.mask_gen_experiment=$MASK_EXP \
    segmentation_3d.exp_name=$SEG_EXP 2>&1 | tee "$LOG"
  echo "  -- $SEG_EXP commit-paint bucket --"
  grep -i "commit paint" "$LOG" || echo "  (no commit-paint line found — check $LOG)"
}

# arm 1: baseline full-frame paint
run_arm "$SEG_BASE" "1"
# arm 2: fast bbox paint (default)
run_arm "$SEG_FAST" ""

echo ""; echo "======== LOSSLESS CHECK (all_obj_labels.pth md5 — MUST MATCH) ========"
MD5_BASE=$(md5sum "$MODEL/segmentation_3d/$SEG_BASE/all_obj_labels.pth" 2>/dev/null | awk '{print $1}')
MD5_FAST=$(md5sum "$MODEL/segmentation_3d/$SEG_FAST/all_obj_labels.pth" 2>/dev/null | awk '{print $1}')
echo "  baseline: $MD5_BASE"
echo "  fast    : $MD5_FAST"
[ -n "$MD5_BASE" ] && [ "$MD5_BASE" = "$MD5_FAST" ] && echo "  -> LOSSLESS ✅ (identical)" || echo "  -> DIFFER ❌ (fast paint is NOT lossless — investigate)"
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
echo ""
echo "Compare the two 'commit paint (CPU 2DSeg)' seconds above for the time saved."
