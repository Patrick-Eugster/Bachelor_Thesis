#!/bin/bash -l
#SBATCH -J render360_A0715_paletteAB
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=10G      # 80 GB total — render_360 loads all_obj_labels + model into RAM (~50-58 GB, resolution-independent)
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00         # fast_render, 2 palettes, full res ~5-10 min each + model load
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/render360_A0715_paletteAB_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/render360_A0715_paletteAB_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# Render the A/0715 opencv 360 flyaround TWICE on the same seg (ocv15k_frust_paint_fast), so the
# supervisor can compare colourings side by side:
#   1. ORIGINAL palette  -> wheat_field_360_original.mp4   (thin hue=(i-1)/n ramp; adjacent heads look same)
#   2. HIGH-CONTRAST     -> wheat_field_360_contrast.mp4   (WHEAT_SEG_CONTRAST_PALETTE=1: golden-ratio hop +
#                                                           alternating S/V -> adjacent heads clearly differ)
# The contrast video answers "are neighbours the same ID?": a patch that STAYS one colour is one merged
# head; separate heads break into distinct colours.
# RENDER ONLY — reuses the existing seg (no re-seg). fast_render, full res (downscale=1). Euler encodes
# mpeg4 (no libx264 there); transcode to H.264 locally for VS Code playback.
# NOTE: requires the WHEAT_SEG_CONTRAST_PALETTE flag in src/gaussians/utils/wheatgs_helper.py — PUSH the
# code to Euler before submitting.
# ============================================================================
set -euo pipefail
FIELD=field_A
DATE=20250715
VAR=opencv
SEG=ocv15k_frust_paint_fast
ITER=15000
REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"
M="$REPO/results/reconstruction/phone/$FIELD/$DATE/$VAR/vanilla_3dgs/baseline"
SEGDIR="$M/segmentation_3d/$SEG"
MP4="$SEGDIR/wheat_field_360.mp4"
OUT_ORIG="$SEGDIR/wheat_field_360_original.mp4"
OUT_CONTRAST="$SEGDIR/wheat_field_360_contrast.mp4"

echo "=== PREFLIGHT ==="
FAIL=0
[ -f "$M/point_cloud/iteration_$ITER/point_cloud.ply" ] || { echo "  X no training model at iter $ITER"; FAIL=1; }
[ -f "$SEGDIR/gaussians.ply" ]                          || { echo "  X no seg gaussians.ply"; FAIL=1; }
[ -f "$SEGDIR/all_obj_labels.pth" ]                     || { echo "  X no all_obj_labels.pth"; FAIL=1; }
[ -e "$OUT_ORIG" ]     && { echo "  X exists: $OUT_ORIG";     FAIL=1; }
[ -e "$OUT_CONTRAST" ] && { echo "  X exists: $OUT_CONTRAST"; FAIL=1; }
grep -q "WHEAT_SEG_CONTRAST_PALETTE" src/gaussians/utils/wheatgs_helper.py || { echo "  X wheatgs_helper.py lacks the contrast flag — push code first"; FAIL=1; }
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: model + seg present, output names free, contrast flag present"

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!; trap "kill $VRAM_PID 2>/dev/null" EXIT

render_pass () {   # $1 = human label
  python src/run_reconstruction.py \
    dataset=phone plot=$FIELD date=$DATE sfm_variant=$VAR \
    experiment_name=baseline \
    reconstruction.iterations=$ITER reconstruction.resolution=1 \
    reconstruction.use_principal_point=false \
    run_render_360=true \
    render_360_downscale=1 \
    segmentation_3d.exp_name=$SEG
}

echo ""; echo "======== PASS 1: ORIGINAL palette ========"
unset WHEAT_SEG_CONTRAST_PALETTE || true
render_pass "original"
mv -v "$MP4" "$OUT_ORIG"

echo ""; echo "======== PASS 2: HIGH-CONTRAST palette ========"
export WHEAT_SEG_CONTRAST_PALETTE=1
render_pass "contrast"
mv -v "$MP4" "$OUT_CONTRAST"

echo ""; echo "======== DONE ========"
echo "  original : $OUT_ORIG"
echo "  contrast : $OUT_CONTRAST"
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
