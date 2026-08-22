#!/bin/bash
# ============================================================================
# E2 SAM3 @4032 — LOCAL box-prep (NO GPU, no detection, non-destructive).
#
# Completes the full-res YOLOv5 grid block for SAM3. SAM3 can only run on Euler
# (26 GB encoder), so here we just stage the boxes: copy the EXISTING full-res
# YOLOv5 boxes from the SAM2 cells (e2_{ff,pt,ph}_sam2) into fresh SAM3 folders
# (e2_{ff,pt,ph}_sam3). Reusing the identical boxes keeps the SAM3 cells a pure
# segmenter A/B against SAM1/SAM2 (same boxes, only SAM changes) — the documented
# box-reuse workflow. Detection is granularity-independent, so each granularity's
# own SAM2 boxes are copied to its SAM3 counterpart.
#
# SAFETY: ABORTS if any e2_*_sam3 target already exists (never overwrites). Only
# writes bboxes/ under new e2_*_sam3 names; touches nothing else.
#
# After this: rsync ONLY the new box folders up to Euler (same include/exclude
# pattern the SAM3 box-reuse workflow uses, just e2_*_sam3 instead of gt_*_sam3),
# then submit scripts/e2_sam3_fullres_job.sh there. From repo root:
#   rsync -av --include='*/' --include='**/e2_*_sam3/bboxes/**' --exclude='*' \
#     results/mask_generation/phone/ \
#     <euler>:/cluster/project/cropsci/peugste/wheat3dgs/results/mask_generation/phone/
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"

SESSIONS=( "field_A/20250627" "field_A/20250706" "field_A/20250715"
           "field_D/20250627" "field_D/20250706" "field_D/20250715" )
declare -A GRAN=( [ff]=full_frame [pt]=per_tile [ph]=per_head )
BASE="results/mask_generation/phone"

echo "=== E2 SAM3 PREP PREFLIGHT ==="
FAIL=0
for sess in "${SESSIONS[@]}"; do
  for g in ff pt ph; do
    src="$BASE/$sess/yolo_sam_v1/e2_${g}_sam2/bboxes"
    dst_exp="$BASE/$sess/yolo_sam_v1/e2_${g}_sam3"
    [ -d "$src" ] || { echo "  X missing SAM2 boxes: $src"; FAIL=1; }
    [ -e "$dst_exp" ] && { echo "  X target already exists (would overwrite): $dst_exp"; FAIL=1; }
  done
done
[ "$FAIL" -ne 0 ] && { echo "ABORTING (nothing written)."; exit 1; }
echo "  ok: all SAM2 boxes present, all e2_*_sam3 names fresh"

echo ""; echo "=== copying boxes SAM2 -> SAM3 (bboxes/ only) ==="
N=0
for sess in "${SESSIONS[@]}"; do
  for g in ff pt ph; do
    src="$BASE/$sess/yolo_sam_v1/e2_${g}_sam2/bboxes"
    dst="$BASE/$sess/yolo_sam_v1/e2_${g}_sam3/bboxes"
    mkdir -p "$dst"
    cp -a "$src/." "$dst/"
    # verify byte-identical so the SAM3 A/B is provably on the same boxes
    if ! diff -rq "$src" "$dst" >/dev/null; then
      echo "  X copy mismatch: $dst"; exit 1
    fi
    n=$(find "$dst" -type f | wc -l)
    N=$((N + 1))
    echo "  ok  $sess  e2_${g}_sam3/bboxes  ($n files)"
  done
done
echo ""; echo "=== DONE: staged $N box folders (6 sessions x 3 granularities) ==="
echo "Next: rsync the e2_*_sam3/bboxes up to Euler, then submit e2_sam3_fullres_job.sh"
