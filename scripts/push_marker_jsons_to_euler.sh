#!/usr/bin/env bash
# Push ONLY the triangulated marker files (logs/marker_points3d.json) for every phone arm up to Euler, so the
# metrics-only rerun (scripts/rerun_metrics_phone_euler_job.sh) can compute the ROI + MARKERS regions there.
# These were generated LOCALLY (detect_markers_v8_cct + triangulate_markers, 2026-08-13) and are the only
# metrics input not already on Euler — sparse/0 (cameras.txt/images.txt) + test/renders are there from the
# recon jobs. 33 tiny JSON files (32 arms + fD/0627 agisoft_2group_old); nothing else is transferred.
#
# Run FROM THE REPO ROOT (/home/patrick/Bachelor_Thesis). The include/exclude filter transfers exactly the
# marker_points3d.json files at any depth and prunes empty dirs — no images, no sparse, no overwrites of
# anything else on Euler.
set -eu
cd /home/patrick/Bachelor_Thesis

rsync -avz --prune-empty-dirs \
  --include='*/' \
  --include='marker_points3d.json' \
  --exclude='*' \
  input_plots/phone/ \
  peugste@euler.ethz.ch:/cluster/project/cropsci/peugste/wheat3dgs/input_plots/phone/
