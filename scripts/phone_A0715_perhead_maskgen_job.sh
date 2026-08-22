#!/bin/bash -l
#SBATCH -J perhead_maskgen_A0715
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G       # 48 GB total — per_head mask-gen is RAM-heavier than per_tile (many per-head full-res masks); covers the ~42 GB peak with margin. Still below the 64 GB the seg jobs used.
#SBATCH --cpus-per-task=8
#SBATCH --time=03:59:00        # short 4h tier: per_head SAM2 ~1h + SAM1 ~2.5h on ~96 imgs (SAM2 first so a SAM1 timeout still leaves a complete SAM2 set)
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/perhead_maskgen_A0715_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/perhead_maskgen_A0715_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# A/0715 — PER_HEAD mask generation ONLY (no seg), opencv, full-res YOLOv5 (target 3904), conf 0.35, ROI.
# Pre-bakes the per_head masks so the two per_head SEG jobs (SAM1, SAM2) can launch seg-ONLY next wave with
# no mask-gen inside the 24h seg slot. Matches the per_tile control (opencv, target 3904, roi, conf 0.35);
# the ONLY axis changed vs per_tile is sam_crop_mode=per_head. Two mask sets in one job:
#   pertile-style names -> perhead_sam2_conf035  (SAM2, ~1h)  and  perhead_sam1_conf035  (SAM1, ~2.5h).
# SAM2 runs FIRST (faster) so a wall-time cut on the slow SAM1 still leaves a complete SAM2 set; each writes
# to its OWN dir, so a partial SAM1 only affects the SAM1 dir (rerun that one).
#
# conf 0.35 matches the existing per_tile anchor (ocv15k_frust_paint SAM1 0.369, ocv15k_conf035 SAM2 0.335)
# so the later per_head segs give the clean per_tile-vs-per_head granularity A/B at matched conf + SAM version.
# ============================================================================

FIELD=field_A
DATE=20250715
VARIANT=opencv
TARGET=3904                              # full-res on the ~3890px opencv frame (stride-32 safe) — matches the per_tile control
CONF=0.35                                # matched to the per_tile anchor
MASK_EXP_SAM2=perhead_sam2_conf035
MASK_EXP_SAM1=perhead_sam1_conf035

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"

MASK_BASE="$REPO/results/mask_generation/phone/$FIELD/$DATE/$VARIANT/yolo_sam_v1"
IN_IMG="$REPO/input_plots/phone/$FIELD/$DATE/$VARIANT/images"
IN_MARK="$REPO/input_plots/phone/$FIELD/$DATE/$VARIANT/logs/marker_points3d.json"
SAM1_W="$REPO/src/mask_generation/weights/sam_vit_h_4b8939.pth"
SAM2_W="$REPO/src/mask_generation/weights/sam2.1_l.pt"

echo "=== PREFLIGHT ==="
FAIL=0
[ -d "$IN_IMG" ]  || { echo "  X no opencv images: $IN_IMG"; FAIL=1; }
[ -f "$IN_MARK" ] || { echo "  X no opencv marker_points3d.json (roi NEEDs it): $IN_MARK"; FAIL=1; }
[ -f "$SAM1_W" ]  || { echo "  X missing SAM1 ViT-H weights: $SAM1_W"; FAIL=1; }
[ -f "$SAM2_W" ]  || { echo "  X missing SAM2 large weights: $SAM2_W"; FAIL=1; }
[ -e "$MASK_BASE/$MASK_EXP_SAM2" ] && { echo "  X mask target exists: $MASK_BASE/$MASK_EXP_SAM2"; FAIL=1; }
[ -e "$MASK_BASE/$MASK_EXP_SAM1" ] && { echo "  X mask target exists: $MASK_BASE/$MASK_EXP_SAM1"; FAIL=1; }
[ "$FAIL" -ne 0 ] && { echo "ABORTING (see X above)."; exit 1; }
echo "  ok: opencv images + markers + both SAM weights present, both mask targets fresh"

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat-maskgen
module load eth_proxy 2>/dev/null || true

run_maskgen () {   # $1 = sam_backend, $2 = experiment_name
  echo ""; echo "======== per_head mask-gen — $1 @ conf $CONF (target $TARGET) ========"
  python src/mask_generation/run_mask_generation.py \
    dataset=phone method=yolo_sam_v1 \
    dataset.plot_glob=$FIELD/$DATE/$VARIANT \
    method.target_image_size=$TARGET method.batch_size_yolo=2 \
    method.sam_crop_mode=per_head method.sam_backend=$1 \
    method.conf_threshold_good_box=$CONF \
    roi.enabled=true \
    experiment_name=$2
  local rc=$?
  echo "  $1 mask-gen rc=$rc -> $MASK_BASE/$2"
  return $rc
}

# SAM2 first (faster) — a wall-time cut on SAM1 below still leaves this set complete.
run_maskgen sam2 "$MASK_EXP_SAM2"; RC_SAM2=$?
run_maskgen sam1 "$MASK_EXP_SAM1"; RC_SAM1=$?

echo ""; echo "======== STATUS ========"
CNT2=$(ls -A "$MASK_BASE/$MASK_EXP_SAM2/masks" 2>/dev/null | wc -l)
CNT1=$(ls -A "$MASK_BASE/$MASK_EXP_SAM1/masks" 2>/dev/null | wc -l)
echo "  SAM2 per_head: rc=$RC_SAM2  masks=$CNT2  -> $MASK_BASE/$MASK_EXP_SAM2"
echo "  SAM1 per_head: rc=$RC_SAM1  masks=$CNT1  -> $MASK_BASE/$MASK_EXP_SAM1"
echo "  (next wave: seg-only jobs point --seg_dir at these; no mask-gen in the 24h seg slot)"
