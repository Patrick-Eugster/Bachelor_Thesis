#!/usr/bin/env bash
# Phone mask-generation GRID runner (local) — the G01–G18 cells of PHONE_MASKGEN_EXPERIMENTS.md.
#   3 detectors × 3 granularities × 2 SAM versions = 18 cells, on the 6 GT images.
# Each cell: run_mask_generation (ROI on, only_labeled_images) → eval_masks_instance (instance F1 + boundary).
# yolo11_seg (G19) is a separate direct-seg method, already scored — not in this grid.
#
# Usage:  scripts/run_maskgen_grid.sh [A|B|C|all]
#   A = full_frame  (6 cheap cells, ~20 min)     <- run + eyeball this first
#   B = per_tile    (6 cells, ~25 min)
#   C = per_head    (6 cells, ~90 min, dominant cost)
#   all = A then B then C
#
# PROGRESS + ETA: a running status line (cell i/N, per-cell seconds, elapsed, rolling ETA) is printed to
# stdout AND appended to a STABLE file you can watch live in another terminal:
#     tail -f results/mask_generation/phone/evaluation/_grid_logs/progress.log
# Per-cell full stdout goes to _grid_logs/<det>_<exp>.log
set -u
cd "$(dirname "$0")/.."   # repo root

BATCH="${1:-A}"
DETECTORS=(yolo_sam_v1 sahi_yolo_sam yolo11_sam)
SAMS=(sam1 sam2)
LOGDIR="results/mask_generation/phone/evaluation/_grid_logs"
PROG="$LOGDIR/progress.log"
mkdir -p "$LOGDIR"

declare -A TAG=( [full_frame]=ff [per_tile]=tile [per_head]=head )

# which granularities this batch runs, and the total cell count N (for i/N + ETA)
case "$BATCH" in
  A)   GRANS=(full_frame) ;;
  B)   GRANS=(per_tile) ;;
  C)   GRANS=(per_head) ;;
  all) GRANS=(full_frame per_tile per_head) ;;
  *)   echo "usage: $0 [A|B|C|all]"; exit 1 ;;
esac
N=$(( ${#GRANS[@]} * ${#DETECTORS[@]} * ${#SAMS[@]} ))

# hms <seconds> -> "Hh Mm Ss" (compact)
hms() { local s=$1; printf '%dm%02ds' $((s/60)) $((s%60)); }

log() { echo "$*" | tee -a "$PROG"; }

RUN_START=$(date +%s)
log "===================================================================================="
log "GRID BATCH '$BATCH' — $N cells — started $(date '+%Y-%m-%d %H:%M:%S')"
log "watch:  tail -f $PROG"
log "===================================================================================="

i=0
for gran in "${GRANS[@]}"; do
  for det in "${DETECTORS[@]}"; do
    for sam in "${SAMS[@]}"; do
      i=$((i+1))
      exp="gt_${TAG[$gran]}_${sam}"
      log_cell="$LOGDIR/${det}_${exp}.log"
      log "[$i/$N] START $(date '+%H:%M:%S')  det=$det gran=$gran sam=$sam  exp=$exp"
      c0=$(date +%s)
      if python src/mask_generation/run_mask_generation.py dataset=phone method="$det" \
           only_labeled_images=true roi.enabled=true \
           method.sam_crop_mode="$gran" method.sam_backend="$sam" \
           experiment_name="$exp" > "$log_cell" 2>&1 \
         && python src/mask_generation/evaluation/eval_masks_instance.py dataset=phone method="$det" \
              method_name="$det" mask_gen_experiment="$exp" eval_experiment="$exp" >> "$log_cell" 2>&1
      then status="OK  "; else status="FAIL"; fi
      c1=$(date +%s)
      cell_s=$((c1-c0)); elapsed=$((c1-RUN_START))
      avg=$(( elapsed / i )); remaining=$(( (N - i) * avg ))
      log "[$i/$N] $status  cell $(hms $cell_s) | elapsed $(hms $elapsed) | avg $(hms $avg) | ETA ~$(hms $remaining) | -> $log_cell"
      [ "$status" = "FAIL" ] && tail -6 "$log_cell" | sed 's/^/       /' | tee -a "$PROG"
    done
  done
done

log "===================================================================================="
log "GRID BATCH '$BATCH' DONE in $(hms $(( $(date +%s) - RUN_START )))  ($N cells)"
log "aggregate:  python src/analysis/aggregate_maskgen_grid.py"
log "===================================================================================="
