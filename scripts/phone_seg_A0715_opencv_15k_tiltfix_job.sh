#!/bin/bash -l
#SBATCH -J seg_A0715_ocv15k_gfix
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G       # 64 GB total — phone 3D seg decodes many full-res masks (RAM-heavy)
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00        # seg-only (masks reused); tilt fix fills the whole plot -> ~2x heads vs the broken half-plot run, so allow more than the old 15k
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv15k_gfix_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv15k_gfix_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# A/0715 OPENCV 15k re-seg WITH THE TILT-CORRECTED GROUND CULL (docs/segmentation_3d/TILT_GROUND_CULL_FIX.md).
# Clean A/B vs the existing (broken) seg "ocv15k_yolov5_pertile": SAME opencv model, SAME masks, ONLY the
# seg code changed. The old run used z<z_mean on a ~51deg-tilted COLMAP frame -> diagonal slice -> half
# the plot empty. The tilt fix is now the DEFAULT (auto, needs only markers), so this run just picks it up.
# NO roi_cull / marker_exclude here — we isolate the tilt fix alone. SEG-ONLY: reuses the masks the old run
# already generated (results/.../opencv/yolo_sam_v1/yolov5_pertile_ocv15k). frustum_cull ON. NO eval_2d
# (pinhole-framed GT misaligned to the cropped opencv frame).
# ============================================================================

FIELD=field_A
DATE=20250715
VARIANT=opencv
MODEL_EXP=baseline
ITER=15000
MASK_EXP=yolov5_pertile_ocv15k          # reuse the EXACT masks the broken ocv15k run used (clean A/B)
SEG_EXP=ocv15k_groundfix                    # new seg output name -> does NOT overwrite the old broken run

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"

MODEL="$REPO/results/reconstruction/phone/$FIELD/$DATE/$VARIANT/vanilla_3dgs/$MODEL_EXP"
MASK_OUT="$REPO/results/mask_generation/phone/$FIELD/$DATE/$VARIANT/yolo_sam_v1/$MASK_EXP"
SEG_OUT="$MODEL/segmentation_3d/$SEG_EXP"
IN_MARK="$REPO/input_plots/phone/$FIELD/$DATE/$VARIANT/logs/marker_points3d.json"

echo "=== PREFLIGHT ==="
FAIL=0
[ -f "$MODEL/point_cloud/iteration_$ITER/point_cloud.ply" ] || { echo "  X no model at iter $ITER: $MODEL"; FAIL=1; }
[ -n "$(ls -A "$MASK_OUT/masks" 2>/dev/null)" ] || { echo "  X no masks at $MASK_OUT/masks — run scripts/phone_seg_A0715_opencv_15k_job.sh first (it makes them), or use the agisoft-style 2-stage variant"; FAIL=1; }
[ -f "$IN_MARK" ] || { echo "  X no marker_points3d.json (the tilt fix NEEDS it): $IN_MARK"; FAIL=1; }
[ -d "$SEG_OUT" ] && [ -n "$(ls -A "$SEG_OUT" 2>/dev/null)" ] && { echo "  X seg target exists: $SEG_OUT"; FAIL=1; }
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: opencv model(iter $ITER) + reused masks + markers present, seg target fresh"

module purge
source ~/miniconda3/etc/profile.d/conda.sh

# ---------------- 3D SEG (tilt-corrected default) + render_360 + eval ----------------
echo ""; echo "======== 3D seg (opencv $MODEL_EXP, iter $ITER) — TILT-CORRECTED default ground cull ========"
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!; trap "kill $VRAM_PID 2>/dev/null" EXIT
export WHEAT_RUN_REPORT="$REPO/slurm_logs/run_report_${SLURM_JOB_ID}.txt"

# tilt fix is the DEFAULT when markers exist — nothing to pass. frustum_cull ON. NO eval_2d (opencv crop).
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
echo "  seg+eval: rc=$SEG_RC -> $SEG_OUT"
echo "  compare heads-found vs the OLD broken run: $MODEL/segmentation_3d/ocv15k_yolov5_pertile/seg_summary.json"
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
