#!/bin/bash -l
#SBATCH -J seg_A0715_ocv30k
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G       # 64 GB total — phone 3D seg decodes many full-res masks (RAM-heavy)
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00        # 30k dense17k model (large): seg ~20-22h + eval; 24h ceiling
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv30k_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv30k_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# A/0715 seg — arm 4 of the 2x2: OPENCV dense17k, 30k (absgrad, matches pinhole colmap_dense17k; VRAM-heavy, frustum_cull ON). NOTE: point_cloud is on Euler only (excluded from local pull) - preflight will abort if absent.
# Self-contained: Stage 1 (wheat-maskgen) full-res YOLOv5 + per_tile SAM1 + ROI on the OPENCV undistorted
# images (target 3904 = no downscale on the ~3898px frame) -> Stage 2 (wheat3dgs) 3D seg of the opencv 15k
# model + render_360 + eval. NO eval_2d: GT masks are pinhole-framed (4032), misaligned to the cropped
# opencv frame. Its own MASK_EXP so all 4 jobs can be sbatch'd at once. frustum_cull ON.
# ============================================================================

FIELD=field_A
DATE=20250715
VARIANT=opencv
MODEL_EXP=dense17k
ITER=30000
TARGET=3904                              # opencv is ~3898px -> 3904 (stride-32) gives r~=1.0 (no downscale)
MASK_EXP=yolov5_pertile_ocv30k
SEG_EXP=ocv30k_yolov5_pertile

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"

MODEL="$REPO/results/reconstruction/phone/$FIELD/$DATE/$VARIANT/vanilla_3dgs/$MODEL_EXP"
MASK_OUT="$REPO/results/mask_generation/phone/$FIELD/$DATE/$VARIANT/yolo_sam_v1/$MASK_EXP"
SEG_OUT="$MODEL/segmentation_3d/$SEG_EXP"
IN_IMG="$REPO/input_plots/phone/$FIELD/$DATE/$VARIANT/images"
IN_MARK="$REPO/input_plots/phone/$FIELD/$DATE/$VARIANT/logs/marker_points3d.json"

echo "=== PREFLIGHT ==="
FAIL=0
[ -f "$MODEL/point_cloud/iteration_$ITER/point_cloud.ply" ] || { echo "  X no model at iter $ITER: $MODEL"; FAIL=1; }
[ -d "$IN_IMG" ]  || { echo "  X no opencv images: $IN_IMG"; FAIL=1; }
[ -f "$IN_MARK" ] || { echo "  X no marker_points3d.json: $IN_MARK"; FAIL=1; }
[ -d "$SEG_OUT" ] && [ -n "$(ls -A "$SEG_OUT" 2>/dev/null)" ] && { echo "  X seg target exists: $SEG_OUT"; FAIL=1; }
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: model(iter $ITER) + opencv images + markers present, seg target fresh"

module purge
source ~/miniconda3/etc/profile.d/conda.sh

# ---------------- STAGE 1: MASK GENERATION (wheat-maskgen) ----------------
echo ""; echo "======== STAGE 1 — full-res YOLOv5 + per_tile SAM1 + ROI (opencv, target $TARGET) ========"
conda activate wheat-maskgen
module load eth_proxy 2>/dev/null || true
python src/mask_generation/run_mask_generation.py \
  dataset=phone method=yolo_sam_v1 \
  dataset.plot_glob=$FIELD/$DATE/$VARIANT \
  method.target_image_size=$TARGET method.batch_size_yolo=2 \
  method.sam_crop_mode=per_tile method.sam_backend=sam1 \
  roi.enabled=true \
  experiment_name=$MASK_EXP
MASK_RC=$?
conda deactivate
echo "  mask-gen rc=$MASK_RC -> $MASK_OUT"
if [ $MASK_RC -ne 0 ] || [ -z "$(ls -A "$MASK_OUT/masks" 2>/dev/null)" ]; then
  echo "ABORTING — mask generation failed or produced no masks."; exit 1
fi

# ---------------- STAGE 2: 3D SEG + render_360 + eval (wheat3dgs) ----------------
echo ""; echo "======== STAGE 2 — 3D seg (opencv $MODEL_EXP, iter $ITER) + render_360 + eval ========"
module purge
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!; trap "kill $VRAM_PID 2>/dev/null" EXIT
export WHEAT_RUN_REPORT="$REPO/slurm_logs/run_report_${SLURM_JOB_ID}.txt"

# opencv -> sfm_variant=opencv; NO eval_2d (pinhole-framed GT misaligned to cropped opencv). frustum_cull ON.
python src/run_reconstruction.py \
  dataset=phone plot=$FIELD date=$DATE sfm_variant=$VARIANT \
  experiment_name=$MODEL_EXP \
  reconstruction.iterations=$ITER reconstruction.resolution=1 \
  reconstruction.use_principal_point=false \
  run_seg=true run_eval=true run_render_360=true \
  segmentation_3d.frustum_cull=true \
  segmentation_3d.detection_method=yolo_sam_v1 \
  segmentation_3d.mask_gen_experiment=$MASK_EXP \
  segmentation_3d.exp_name=$SEG_EXP
SEG_RC=$?

echo ""; echo "======== STATUS ========"
echo "  mask-gen: rc=$MASK_RC -> $MASK_OUT"
echo "  seg+eval: rc=$SEG_RC -> $SEG_OUT"
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
