#!/usr/bin/env bash
# EVAL-ONLY re-score of the phone mask-gen grid — runs ONLY eval_masks_instance on masks already on disk.
# Use this (not run_maskgen_grid.sh) when the masks already exist and you only changed the EVALUATOR
# (e.g. added the F1-vs-IoU curve, matched-pair IoU histogram, or the dynamic boundary band): it does NOT
# re-run SAM, and it covers sam3 too (run_maskgen_grid.sh only does sam1/sam2 and re-runs inference).
#
# Grid = 3 detectors x 3 granularities x {sam1,sam2,sam3} = 27 cells, on the 6 GT images. Plus yolo11_seg/gt_eval.
# A cell is skipped (not failed) if its mask-gen experiment folder has no masks/ on disk.
#
# Usage:  scripts/rescore_maskgen_grid.sh [clean]
#   clean = FIRST wipe results/mask_generation/phone/evaluation/*/masks_instance/ + grid_summary.csv, so no
#           OLD-format JSONs (no dynamic/curve) linger and mix into the aggregator glob. RECOMMENDED here,
#           because the existing sam1/sam2 eval JSONs predate the new metrics.
#   (no arg) = re-score in place (overwrites each cell's JSON; safe, but leftover unrelated eval folders remain).
set -u
cd "$(dirname "$0")/.."   # repo root

EVROOT="results/mask_generation/phone/evaluation"
DETECTORS=(yolo_sam_v1 sahi_yolo_sam yolo11_sam)
SAMS=(sam1 sam2 sam3)
declare -A TAG=( [full_frame]=ff [per_tile]=tile [per_head]=head )
GRANS=(full_frame per_tile per_head)
LOGDIR="$EVROOT/_grid_logs"
mkdir -p "$LOGDIR"
PROG="$LOGDIR/rescore.log"
log() { echo "$*" | tee -a "$PROG"; }

if [ "${1:-}" = "clean" ]; then
  log "CLEAN: wiping $EVROOT/*/masks_instance/ + grid_summary.csv (masks untouched — they live under <method>/<exp>/masks/)"
  rm -rf "$EVROOT"/*/masks_instance
  rm -f  "$EVROOT/grid_summary.csv"
fi

# path to a cell's masks (any GT plot has them; presence of masks/ under a plot is the marker).
# phone plots are TWO levels deep: phone/<field>/<date>/<method>/<exp>/masks/  -> two wildcards before method.
masks_exist() {  # $1=det $2=exp  -> 0 if at least one plot has masks/ for this cell
  compgen -G "results/mask_generation/phone/*/*/$1/$2/masks/*.png" >/dev/null 2>&1
}

N=$(( ${#GRANS[@]} * ${#DETECTORS[@]} * ${#SAMS[@]} + 1 ))   # +1 for yolo11_seg/gt_eval
i=0; ok=0; skip=0; fail=0
RUN_START=$(date +%s)
log "===================================================================================="
log "RE-SCORE grid (eval only) — up to $N cells — $(date '+%Y-%m-%d %H:%M:%S')"
log "===================================================================================="

score() {  # $1=method_name $2=exp
  local det="$1" exp="$2" cell_log="$LOGDIR/rescore_${1}_${2}.log"
  i=$((i+1))
  if ! masks_exist "$det" "$exp"; then
    log "[$i/$N] SKIP  $det/$exp  (no masks on disk)"; skip=$((skip+1)); return
  fi
  log "[$i/$N] score $det/$exp"
  if python src/mask_generation/evaluation/eval_masks_instance.py dataset=phone \
        method_name="$det" mask_gen_experiment="$exp" eval_experiment="$exp" \
        > "$cell_log" 2>&1
  then ok=$((ok+1)); else fail=$((fail+1)); log "       FAIL -> $cell_log"; tail -6 "$cell_log" | sed 's/^/       /' | tee -a "$PROG"; fi
}

for gran in "${GRANS[@]}"; do
  for det in "${DETECTORS[@]}"; do
    for sam in "${SAMS[@]}"; do
      score "$det" "gt_${TAG[$gran]}_${sam}"
    done
  done
done
score "yolo11_seg" "gt_eval"   # G19 direct-seg cell (own method folder)

log "===================================================================================="
log "RE-SCORE DONE in $(( $(date +%s) - RUN_START ))s  |  OK=$ok  SKIP=$skip  FAIL=$fail"
log "aggregate:  python src/analysis/aggregate_maskgen_grid.py"
log "===================================================================================="
