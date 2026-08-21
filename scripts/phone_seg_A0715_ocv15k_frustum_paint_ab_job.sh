#!/bin/bash -l
#SBATCH -J seg_A0715_ocv15k_frust_paintab
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G       # 64 GB total — phone 3D seg decodes many full-res masks (RAM-heavy)
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00        # ONE opencv frustum seg (seg-only, no eval/render) fits well under 24h
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv15k_frust_paintab_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv15k_frust_paintab_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# FAST-PAINT A/B on the OPENCV 15k model under the FULL cull config (roi_cull + marker_exclude +
# frustum_cull) — the same config as job 11199101, so this A/B lands INSIDE the opencv cull ablation.
# ONE ARM PER JOB (pass 'baseline' or 'fast' as the sbatch argument) so each job is a single ~seg-only
# run that fits 24h with no truncation risk. Submit BOTH; they queue and start whenever nodes free:
#     sbatch /cluster/project/cropsci/peugste/wheat3dgs/scripts/phone_seg_A0715_ocv15k_frustum_paint_ab_job.sh baseline
#     sbatch /cluster/project/cropsci/peugste/wheat3dgs/scripts/phone_seg_A0715_ocv15k_frustum_paint_ab_job.sh fast
#
#   baseline arm : WHEAT_SEG_NO_FAST_PAINT=1  -> OLD full-frame commit paint (the slow bucket)
#   fast arm     : (unset)                    -> NEW bbox commit paint (default)
# Both arms: WHEAT_SEG_TIMING=1 -> the "commit paint (CPU 2DSeg)" seconds are printed for each arm; the
# fast-vs-baseline delta = the time saved. Seg output is provably lossless, so the two arms' all_obj_labels.pth
# md5 MUST match each other (and 11199101's) — see the compare block at the very end.
#
# WHY a fresh baseline arm instead of reusing 11199101: 11199101 was submitted from a script snapshot BEFORE
# WHEAT_SEG_TIMING was added AND started ~4 min before the fast-paint code landed on Euler, so it ran OLD paint
# with NO timing output. It gives an md5 cross-check only, not a slow-paint time — hence this self-contained A/B.
# NO eval / NO render_360 (timing + md5 only).
# ============================================================================

ARM="${1:-}"
case "$ARM" in
  baseline) NOFAST=1;  SEG_EXP=ocv15k_frust_paint_baseline ;;
  fast)     NOFAST="";  SEG_EXP=ocv15k_frust_paint_fast ;;
  *) echo "usage: sbatch $0 [baseline|fast]"; exit 1 ;;
esac

FIELD=field_A
DATE=20250715
VARIANT=opencv
MODEL_EXP=baseline
ITER=15000
MASK_EXP=yolov5_pertile_ocv15k          # SAME masks 11199101 used

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"

MODEL="$REPO/results/reconstruction/phone/$FIELD/$DATE/$VARIANT/vanilla_3dgs/$MODEL_EXP"
MASK_OUT="$REPO/results/mask_generation/phone/$FIELD/$DATE/$VARIANT/yolo_sam_v1/$MASK_EXP"
SEG_OUT="$MODEL/segmentation_3d/$SEG_EXP"
IN_MARK="$REPO/input_plots/phone/$FIELD/$DATE/$VARIANT/logs/marker_points3d.json"

echo "=== PREFLIGHT (arm=$ARM -> $SEG_EXP) ==="
FAIL=0
[ -f "$MODEL/point_cloud/iteration_$ITER/point_cloud.ply" ] || { echo "  X no model at iter $ITER: $MODEL"; FAIL=1; }
[ -n "$(ls -A "$MASK_OUT/masks" 2>/dev/null)" ] || { echo "  X no masks at $MASK_OUT/masks"; FAIL=1; }
[ -f "$IN_MARK" ] || { echo "  X no marker_points3d.json (roi_cull + marker_exclude NEED it): $IN_MARK"; FAIL=1; }
[ -d "$SEG_OUT" ] && [ -n "$(ls -A "$SEG_OUT" 2>/dev/null)" ] && { echo "  X seg target exists: $SEG_OUT"; FAIL=1; }
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: opencv model + reused masks + markers present, seg target fresh"

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!; trap "kill $VRAM_PID 2>/dev/null" EXIT

LOG="$REPO/slurm_logs/frust_paint_${SLURM_JOB_ID}_${ARM}.log"
echo ""; echo "======== ARM: $ARM  (WHEAT_SEG_NO_FAST_PAINT='${NOFAST}')  roi+marker+frustum ON ========"
WHEAT_SEG_TIMING=1 WHEAT_SEG_NO_FAST_PAINT="$NOFAST" \
python src/run_reconstruction.py \
  dataset=phone plot=$FIELD date=$DATE sfm_variant=$VARIANT \
  experiment_name=$MODEL_EXP \
  reconstruction.iterations=$ITER reconstruction.resolution=1 \
  reconstruction.use_principal_point=false \
  run_seg=true \
  segmentation_3d.frustum_cull=true \
  segmentation_3d.roi_cull=true \
  segmentation_3d.marker_exclude=true \
  segmentation_3d.detection_method=yolo_sam_v1 \
  segmentation_3d.mask_gen_experiment=$MASK_EXP \
  segmentation_3d.exp_name=$SEG_EXP 2>&1 | tee "$LOG"

echo ""; echo "======== $ARM DONE ========"
echo "  -- commit-paint bucket for this arm --"
grep -i "commit paint" "$LOG" | tail || echo "  (no commit-paint line — check timing)"
echo "  md5(this arm all_obj_labels.pth): $(md5sum "$SEG_OUT/all_obj_labels.pth" 2>/dev/null | awk '{print $1}')"
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
echo ""
echo "AFTER BOTH ARMS DONE, compare (run locally or on Euler):"
echo "  md5sum $MODEL/segmentation_3d/ocv15k_frust_paint_{baseline,fast}/all_obj_labels.pth   # must MATCH"
echo "  grep -i 'commit paint' slurm_logs/frust_paint_*_baseline.log slurm_logs/frust_paint_*_fast.log  # baseline vs fast seconds"
