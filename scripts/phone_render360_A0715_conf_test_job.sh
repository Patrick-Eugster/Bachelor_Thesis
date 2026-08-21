#!/bin/bash -l
#SBATCH -J render360_A0715_conftest
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=10G      # 80 GB total — render_360 loads all_obj_labels + per-head plys into RAM (~42 GB); this is the RAM-heavy step we deferred out of the seg jobs
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00         # 2 palettes, full res, fast_render ~9 min each + model load
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/render360_A0715_conftest_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/render360_A0715_conftest_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# RENDER-LATER for the 3D-seg CONF TEST — the RAM-heavy render step, decoupled from the seg-only jobs and
# run AFTER a conf's seg has finished (reuses gaussians.ply + all_obj_labels.pth, no re-seg). One arm per
# job, argument = the conf you already segged:
#     sbatch .../phone_render360_A0715_conf_test_job.sh 0.40
#     sbatch .../phone_render360_A0715_conf_test_job.sh 0.55
#     sbatch .../phone_render360_A0715_conf_test_job.sh 0.70
#
# Renders the flyaround TWICE for that conf's seg output:
#   ORIGINAL palette -> wheat_field_360_original.mp4
#   HIGH-CONTRAST    -> wheat_field_360_contrast.mp4   (WHEAT_SEG_CONTRAST_PALETTE=1)
# Full res (downscale=1), fast_render. Euler encodes mpeg4 (no libx264) -> transcode to H.264 locally.
# NOTE: needs the WHEAT_SEG_CONTRAST_PALETTE flag in src/gaussians/utils/wheatgs_helper.py — push code first.
# ============================================================================
set -euo pipefail
CONF="${1:-}"
case "$CONF" in
  0.40) CC=040 ;;
  0.55) CC=055 ;;
  0.70) CC=070 ;;
  *) echo "usage: sbatch $0 <0.40|0.55|0.70>"; exit 1 ;;
esac

FIELD=field_A
DATE=20250715
VARIANT=opencv
MODEL_EXP=baseline
ITER=15000
SEG_EXP=ocv15k_conf${CC}                  # the seg output produced by phone_seg_A0715_conf_test_job.sh $CONF

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"
M="$REPO/results/reconstruction/phone/$FIELD/$DATE/$VARIANT/vanilla_3dgs/$MODEL_EXP"
SEGDIR="$M/segmentation_3d/$SEG_EXP"
MP4="$SEGDIR/wheat_field_360.mp4"
OUT_ORIG="$SEGDIR/wheat_field_360_original.mp4"
OUT_CONTRAST="$SEGDIR/wheat_field_360_contrast.mp4"

echo "=== PREFLIGHT (conf=$CONF -> $SEG_EXP) ==="
FAIL=0
[ -f "$M/point_cloud/iteration_$ITER/point_cloud.ply" ] || { echo "  X no training model at iter $ITER"; FAIL=1; }
[ -f "$SEGDIR/gaussians.ply" ]                          || { echo "  X no seg gaussians.ply ($SEG_EXP) — seg not done?"; FAIL=1; }
[ -f "$SEGDIR/all_obj_labels.pth" ]                     || { echo "  X no all_obj_labels.pth ($SEG_EXP)"; FAIL=1; }
[ -e "$OUT_ORIG" ]     && { echo "  X exists: $OUT_ORIG";     FAIL=1; }
[ -e "$OUT_CONTRAST" ] && { echo "  X exists: $OUT_CONTRAST"; FAIL=1; }
grep -q "WHEAT_SEG_CONTRAST_PALETTE" src/gaussians/utils/wheatgs_helper.py || { echo "  X wheatgs_helper.py lacks the contrast flag — push code first"; FAIL=1; }
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: model + seg($SEG_EXP) present, output names free, contrast flag present"

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!; trap "kill $VRAM_PID 2>/dev/null" EXIT

render_pass () {
  python src/run_reconstruction.py \
    dataset=phone plot=$FIELD date=$DATE sfm_variant=$VARIANT \
    experiment_name=$MODEL_EXP \
    reconstruction.iterations=$ITER reconstruction.resolution=1 \
    reconstruction.use_principal_point=false \
    run_render_360=true \
    render_360_downscale=1 \
    segmentation_3d.exp_name=$SEG_EXP
}

echo ""; echo "======== PASS 1: ORIGINAL palette (conf=$CONF) ========"
unset WHEAT_SEG_CONTRAST_PALETTE || true
render_pass
mv -v "$MP4" "$OUT_ORIG"

echo ""; echo "======== PASS 2: HIGH-CONTRAST palette (conf=$CONF) ========"
export WHEAT_SEG_CONTRAST_PALETTE=1
render_pass
mv -v "$MP4" "$OUT_CONTRAST"

echo ""; echo "======== DONE (conf=$CONF) ========"
echo "  original : $OUT_ORIG"
echo "  contrast : $OUT_CONTRAST"
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
