#!/usr/bin/env bash
# A1 front-end re-run: SIFT at max_image_size=2048 (MATCHED to the ALIKED baseline's downscale), so the
# SIFT-vs-ALIKED comparison is resolution-fair. Writes ONLY to each session's sift_2048/ — the existing
# sift/ (SIFT at COLMAP-default 3200) is left untouched for the record. Then compares vs Agisoft.
set -u
cd /workspace
LOG=/workspace/scripts/sift2048_a1.log
: > "$LOG"
SESSIONS=(
  "field_A 20250618" "field_A 20250627" "field_A 20250706" "field_A 20250715"
  "field_D 20250618" "field_D 20250627" "field_D 20250706" "field_D 20250715"
)
for s in "${SESSIONS[@]}"; do
  set -- $s; F=$1; P=$2
  echo "==== $(date '+%H:%M:%S')  SIFT@2048  $F/$P ====" | tee -a "$LOG"
  python src/preprocessing/run_colmap.py field=$F plot=$P \
    front_end=sift sift_max_image_size=2048 variant_dir=sift_2048 >>"$LOG" 2>&1 \
    && echo "  SfM OK  $F/$P" | tee -a "$LOG" \
    || { echo "  SfM FAIL $F/$P" | tee -a "$LOG"; continue; }
  # compare vs Agisoft (default native ref; fD/0627 also gets the merged special case)
  python src/preprocessing/compare_to_agisoft.py field=$F plot=$P \
    ours_sparse_dir=sift_2048/sparse/0 \
    output_file=logs/compare_to_agisoft_sift2048.json >>"$LOG" 2>&1 \
    && echo "  compare OK  $F/$P" | tee -a "$LOG" \
    || echo "  compare FAIL $F/$P (model may have fragmented)" | tee -a "$LOG"
done
echo "==== DONE $(date '+%H:%M:%S') ====" | tee -a "$LOG"
# registration summary (one model? how many images?)
for s in "${SESSIONS[@]}"; do
  set -- $s; F=$1; P=$2
  nsub=$(ls -d input_plots/phone/$F/$P/sift_2048/distorted/sparse/*/ 2>/dev/null | wc -l)
  nreg=$(grep -c '^[0-9]' input_plots/phone/$F/$P/sift_2048/sparse/0/images.txt 2>/dev/null || echo NA)
  echo "  $F/$P : sub-models=$nsub  registered(images.txt lines)=$nreg" | tee -a "$LOG"
done
