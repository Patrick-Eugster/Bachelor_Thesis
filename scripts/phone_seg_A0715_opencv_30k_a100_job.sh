#!/bin/bash -l
#SBATCH -J seg_A0715_ocv30k_a100
#SBATCH --gpus=a100-pcie-40gb:1 # A100 40 GB — the rtx_4090 (24 GB) CUDA-OOM'd in flashsplat seg: ~14.5 GiB
                               #   in use + a 10.66 GiB alloc = ~25 GiB peak, so 40 GB clears it with ~15 GiB margin.
#SBATCH --mem-per-cpu=16G      # 128 GB total (8 × 16G) — seg decodes many full-res masks AND this job also
#SBATCH --cpus-per-task=8      #   runs render_360, which RAM-OOM'd at 64 GB on the sibling 4.47M model (2x margin).
#SBATCH --time=23:00:00        # 30k dense17k model: seg ~15-22h + eval + render_360; 24h ceiling.
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv30k_a100_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/seg_A0715_ocv30k_a100_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# A/0715 seg — arm 4 RETRY on A100: OPENCV dense17k, 30k (~4.47M gaussians).
# ----------------------------------------------------------------------------
# The rtx_4090 run (job 10667743) CRASHED in Stage 4 (seg) with torch.cuda.OutOfMemoryError inside the
# flashsplat rasterizer — the undistorted opencv frames render larger than the pinhole ones, so the
# per-view buffers exceeded 24 GB. This retry uses an A100 40 GB. Stage 1 (mask-gen) already succeeded on
# the failed run (24586 masks on disk), so we SKIP it if the masks are present. Stage 2 = 3D seg +
# render_360 + eval (NO eval_2d: pinhole-framed GT misaligned to the cropped opencv frame). frustum_cull ON.
#
# ⚠️ ARCH NOTE: the wheat3dgs env's compiled CUDA submodules (flashsplat / diff-gaussian) must include
#   sm_80 in their build. If seg dies immediately with "no kernel image is available for execution on the
#   device", the submodules were 4090-only (sm_89) and need a rebuild with TORCH_CUDA_ARCH_LIST including 8.0.
# ⚠️ point_cloud is Euler-only (excluded from local pull) — preflight aborts if the model is absent.
# ============================================================================

FIELD=field_A
DATE=20250715
VARIANT=opencv
MODEL_EXP=dense17k
ITER=30000
TARGET=3904                              # opencv is ~3898px -> 3904 (stride-32) gives r~=1.0 (no downscale)
MASK_EXP=yolov5_pertile_ocv30k
SEG_EXP=ocv30k_yolov5_pertile_a100       # separate folder — leaves the crashed run's partial untouched

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
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
# SEG_EXP is a dedicated _a100 folder, so the crashed 4090 run (ocv30k_yolov5_pertile) is left untouched.
# This only fires on a repeat A100 attempt — wipe its own stale partial so seg starts clean.
if [ -d "$SEG_OUT" ] && [ -n "$(ls -A "$SEG_OUT" 2>/dev/null)" ]; then
  echo "  ! removing stale partial seg output from a prior A100 attempt: $SEG_OUT"; rm -rf "$SEG_OUT"
fi
echo "  ok: model(iter $ITER) + opencv images + markers present"

module purge
source ~/miniconda3/etc/profile.d/conda.sh

# ---------------- STAGE 1: MASK GENERATION — SKIP if masks already present ----------------
if [ -n "$(ls -A "$MASK_OUT/masks" 2>/dev/null)" ]; then
  echo ""; echo "======== STAGE 1 — SKIPPED (masks already exist: $MASK_OUT/masks) ========"
  MASK_RC=0
else
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
fi

# ---------------- STAGE 2: 3D SEG + render_360 + eval (wheat3dgs) ----------------
echo ""; echo "======== STAGE 2 — 3D seg (opencv $MODEL_EXP, iter $ITER) + render_360 + eval — A100 40 GB ========"
module purge
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

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
echo "  seg+eval+r360: rc=$SEG_RC -> $SEG_OUT"
PEAK_VRAM_MIB=$(sort -n "$VRAM_LOG" | tail -1)
PEAK_RAM_GIB=$(awk "BEGIN{printf \"%.2f\", $(sort -n "$RAM_LOG" | tail -1)/1024/1024/1024}")
echo "Peak VRAM: ${PEAK_VRAM_MIB} MiB ($(awk "BEGIN{printf \"%.2f\", ${PEAK_VRAM_MIB}/1024}") GiB)"
echo "Peak RAM:  ${PEAK_RAM_GIB} GiB (job cgroup)"
[ -f "$WHEAT_RUN_REPORT" ] && cat "$WHEAT_RUN_REPORT"
