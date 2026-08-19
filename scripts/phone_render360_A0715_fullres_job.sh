#!/bin/bash -l
#SBATCH -J render360_A0715_fullres
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=10G      # 80 GB total — render_360 loads all_obj_labels + model into RAM (measured peak ~50-58 GB; resolution-INDEPENDENT, so downscale=1 needs the same)
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00         # fast_render, 2 models, full res ~5-10 min each + model load
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/render360_A0715_fullres_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/render360_A0715_fullres_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# Re-render the A/0715 360 flyaround videos at FULL RESOLUTION for BOTH runs (opencv + agisoft).
# Fixes the two quality problems + the agisoft-blank bug:
#   1. render_360_downscale=1  -> full training-image resolution (was 2 = half -> blurry).
#   2. encoder now libx264/mpeg4 WITH quality flags (crf 16 / qscale 2) -> was default mpeg4 = "hard compressed".
#   3. orbit fix in wheatgs_helper.get_camera_path_fixed_elevation: the world-Z path now orbits the PLOT
#      (look_at), not the origin -> agisoft's metric frame (plot ~500 units out) was rendering BLANK white.
# RENDER ONLY: reuses the existing seg outputs (ocv15k_groundfix / agi15k_groundfix) — NO re-seg.
# fast_render (single colored pass/frame) as in the seg jobs. Euler encodes high-quality mpeg4 (no libx264
# there); the full-res mp4 is clean — optionally transcode to H.264 locally for player compatibility.
# ============================================================================

FIELD=field_A
DATE=20250715
ITER=15000
REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"

# variant : seg_exp  (the two finished runs)
RUNS=("opencv:ocv15k_groundfix" "agisoft:agi15k_groundfix")

echo "=== PREFLIGHT ==="
FAIL=0
for r in "${RUNS[@]}"; do
  VAR=${r%%:*}; SEG=${r##*:}
  M="$REPO/results/reconstruction/phone/$FIELD/$DATE/$VAR/vanilla_3dgs/baseline"
  [ -f "$M/point_cloud/iteration_$ITER/point_cloud.ply" ] || { echo "  X $VAR: no training model at iter $ITER"; FAIL=1; }
  [ -f "$M/segmentation_3d/$SEG/gaussians.ply" ]          || { echo "  X $VAR: no seg gaussians.ply ($SEG)"; FAIL=1; }
  [ -f "$M/segmentation_3d/$SEG/all_obj_labels.pth" ]     || { echo "  X $VAR: no all_obj_labels.pth ($SEG)"; FAIL=1; }
done
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: both models + seg outputs present"

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
nvidia-smi

VRAM_LOG="$REPO/slurm_logs/vram_${SLURM_JOB_ID}.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$VRAM_LOG" &
VRAM_PID=$!; trap "kill $VRAM_PID 2>/dev/null" EXIT

for r in "${RUNS[@]}"; do
  VAR=${r%%:*}; SEG=${r##*:}
  echo ""; echo "======== RENDER360 full-res: $VAR ($SEG) ========"
  # render_360 only (no re-seg); render_360_downscale=1 = full res; fast_render_360 stays true (default)
  python src/run_reconstruction.py \
    dataset=phone plot=$FIELD date=$DATE sfm_variant=$VAR \
    experiment_name=baseline \
    reconstruction.iterations=$ITER reconstruction.resolution=1 \
    reconstruction.use_principal_point=false \
    run_render_360=true \
    render_360_downscale=1 \
    segmentation_3d.exp_name=$SEG
  echo "  -> results/reconstruction/phone/$FIELD/$DATE/$VAR/vanilla_3dgs/baseline/segmentation_3d/$SEG/wheat_field_360.mp4"
done

echo ""; echo "======== DONE ========"
echo "Peak VRAM: $(sort -n "$VRAM_LOG" | tail -1) MiB"
