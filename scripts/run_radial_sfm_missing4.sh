#!/usr/bin/env bash
# Build the radial (SIMPLE_RADIAL) ALIKED SfM variant for the 4 sessions that lack it,
# so the phone camera-model recon table can be completed. Writes only to each session's radial/.
# Replicates the existing radial config: ALIKED + exhaustive + single_camera.
set -u
cd /workspace
LOG=/workspace/scripts/radial_sfm_missing4.log
: > "$LOG"
SESSIONS=(
  "field_A 20250627"
  "field_A 20250706"
  "field_D 20250618"
  "field_D 20250715"
)
for s in "${SESSIONS[@]}"; do
  set -- $s; F=$1; P=$2
  echo "==== $(date '+%H:%M:%S')  radial SfM  $F/$P ====" | tee -a "$LOG"
  python src/preprocessing/run_colmap.py field=$F plot=$P \
    camera=SIMPLE_RADIAL front_end=aliked variant_dir=radial >>"$LOG" 2>&1 \
    && echo "  OK  $F/$P" | tee -a "$LOG" \
    || echo "  FAIL $F/$P (see log)" | tee -a "$LOG"
done
echo "==== DONE $(date '+%H:%M:%S') ====" | tee -a "$LOG"
# quick registration check
for s in "${SESSIONS[@]}"; do
  set -- $s; F=$1; P=$2
  n=$(grep -c '^[0-9]' input_plots/phone/$F/$P/radial/sparse/0/images.txt 2>/dev/null || echo NA)
  echo "  $F/$P radial images.txt lines=$n" | tee -a "$LOG"
done
