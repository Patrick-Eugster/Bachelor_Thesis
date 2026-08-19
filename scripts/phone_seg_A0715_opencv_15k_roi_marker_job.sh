#!/bin/bash -l
#SBATCH -J seg_A0715_ocv15k_roimark_gf
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G       # 64 GB total — phone 3D seg decodes many full-res masks (RAM-heavy)
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00        # seg-only (masks reused); ROI cull removes background so fewer candidate Gaussians -> likely a bit faster than the tiltfix run, but keep the same ceiling
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv15k_roimark_gf_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv15k_roimark_gf_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# A/0715 OPENCV 15k re-seg WITH roi_cull + marker_exclude (F1 + F2), ON TOP of the tilt-corrected cull.
# Follow-up to phone_seg_A0715_opencv_15k_tiltfix_job.sh: same opencv model, same reused full-res masks,
# but now also (F1) restrict to the marker-hull plot ROI so the ground/canopy BEHIND the plot can't form
# one large background blob, and (F2) drop the coded-marker plates so they aren't segmented as heads.
# roi_cull already cuts along the fitted marker plane, so the tilt correction is included. Compare against:
#   - ocv15k_yolov5_pertile   (OLD, broken diagonal cut)
#   - ocv15k_tiltfix          (tilt fix only, no ROI/marker)  <- isolates what ROI+marker add on top
# SEG-ONLY: reuses results/.../opencv/yolo_sam_v1/yolov5_pertile_ocv15k. frustum_cull ON. NO eval_2d
# (pinhole-framed GT misaligned to the cropped opencv frame). marker plates are ~13 cm circle / 15 cm
# square -> default marker_radius_m=0.075 covers the plate; roi_buffer_m=0.25 spares plot-edge heads.
# ============================================================================

FIELD=field_A
DATE=20250715
VARIANT=opencv
MODEL_EXP=baseline
ITER=15000
MASK_EXP=yolov5_pertile_ocv15k          # reuse the SAME full-res masks as the tiltfix / old runs
SEG_EXP=ocv15k_roimark_groundfix                 # new seg output name -> does NOT overwrite the tiltfix / old runs

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"

MODEL="$REPO/results/reconstruction/phone/$FIELD/$DATE/$VARIANT/vanilla_3dgs/$MODEL_EXP"
MASK_OUT="$REPO/results/mask_generation/phone/$FIELD/$DATE/$VARIANT/yolo_sam_v1/$MASK_EXP"
SEG_OUT="$MODEL/segmentation_3d/$SEG_EXP"
IN_MARK="$REPO/input_plots/phone/$FIELD/$DATE/$VARIANT/logs/marker_points3d.json"

echo "=== PREFLIGHT ==="
FAIL=0
[ -f "$MODEL/point_cloud/iteration_$ITER/point_cloud.ply" ] || { echo "  X no model at iter $ITER: $MODEL"; FAIL=1; }
[ -n "$(ls -A "$MASK_OUT/masks" 2>/dev/null)" ] || { echo "  X no masks at $MASK_OUT/masks — run phone_seg_A0715_opencv_15k_job.sh first (it makes them)"; FAIL=1; }
[ -f "$IN_MARK" ] || { echo "  X no marker_points3d.json (roi_cull + marker_exclude NEED it): $IN_MARK"; FAIL=1; }
[ -d "$SEG_OUT" ] && [ -n "$(ls -A "$SEG_OUT" 2>/dev/null)" ] && { echo "  X seg target exists: $SEG_OUT"; FAIL=1; }
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: opencv model(iter $ITER) + reused masks + markers present, seg target fresh"

module purge
source ~/miniconda3/etc/profile.d/conda.sh

# ---------------- 3D SEG (roi_cull + marker_exclude) + render_360 + eval ----------------
echo ""; echo "======== 3D seg (opencv $MODEL_EXP, iter $ITER) — roi_cull + marker_exclude ON ========"
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!; trap "kill $VRAM_PID 2>/dev/null" EXIT
export WHEAT_RUN_REPORT="$REPO/slurm_logs/run_report_${SLURM_JOB_ID}.txt"
export WHEAT_SEG_TIMING=1   # print find_match render / match-IoU / commit-paint buckets (lossless) so Run A is directly comparable to the frustum Run B

# roi_cull (F1) + marker_exclude (F2) ON; defaults roi_buffer_m=0.25, marker_radius_m=0.075. frustum_cull OFF
# (this is Run A of the speed A/B; the frustum-on twin is phone_seg_A0715_opencv_15k_roi_marker_frustum_job.sh).
python src/run_reconstruction.py \
  dataset=phone plot=$FIELD date=$DATE sfm_variant=$VARIANT \
  experiment_name=$MODEL_EXP \
  reconstruction.iterations=$ITER reconstruction.resolution=1 \
  reconstruction.use_principal_point=false \
  run_seg=true run_eval=true run_render_360=true \
  segmentation_3d.frustum_cull=true \
  segmentation_3d.roi_cull=true \
  segmentation_3d.marker_exclude=true \
  segmentation_3d.detection_method=yolo_sam_v1 \
  segmentation_3d.mask_gen_experiment=$MASK_EXP \
  segmentation_3d.exp_name=$SEG_EXP
SEG_RC=$?

echo ""; echo "======== STATUS ========"
echo "  seg+eval: rc=$SEG_RC -> $SEG_OUT"
echo "  compare vs: $MODEL/segmentation_3d/{ocv15k_yolov5_pertile (old broken), ocv15k_tiltfix (tilt only)}"
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
