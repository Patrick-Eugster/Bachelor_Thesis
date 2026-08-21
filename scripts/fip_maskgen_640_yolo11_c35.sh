#!/usr/bin/env bash
# Local FIP mask-gen for the 3D-seg detector experiment (bad / baseline / best triple).
#
# Generates ALL THREE seed-mask sets fresh so they share the same SAM (ViT-H / sam1), the same conf
# operating point (0.35), the same crop mode, and cover all 7 FIP plots + all 36 views each:
#   - BAD      : YOLOv5 @ target_image_size=640   + SAM1 -> yolo_sam_v1/fipseg_yv5_640_c35
#   - BASELINE : YOLOv5 @ target_image_size=1280  + SAM1 -> yolo_sam_v1/fipseg_yv5_1280_c35
#   - BEST     : YOLO11 (yolo11_sam, imgsz 3008)  + SAM1 -> yolo11_sam/fipseg_yolo11_c35
# (1280 is re-run cleanly rather than reusing the older yolo_sam_v1/initial, so all three share one
# identical invocation and only the detector/imgsz differs — clean provenance for the thesis.)
#
# Why sam1 (ViT-H): match SAM across the trio so the ONLY variable is the detector. Why conf 0.35: the
# box-AP table uses 0.35, so all three share the same operating point (fair; not each detector's F1 peak).
# only_labeled_images stays false -> masks for every view (3D seg needs all 36, not just the GT cam).
#
# Run locally (mask-gen is GPU-light; NOT 3D seg): bash scripts/fip_maskgen_640_yolo11_c35.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=src/mask_generation/run_mask_generation.py
COMMON=(dataset=fip dataset.plot_glob='*' only_labeled_images=false)   # all 7 plots, all views

echo "==================================================================="
echo " FIP mask-gen trio: YOLOv5@640 / YOLOv5@1280 / YOLO11+SAM, conf 0.35, sam1"
echo "==================================================================="

echo; echo "### [1/3] YOLOv5 @ 640 + SAM1  -> yolo_sam_v1/fipseg_yv5_640_c35"
python "$RUN" "${COMMON[@]}" method=yolo_sam_v1 \
  method.target_image_size=640 \
  method.sam_backend=sam1 \
  method.conf_threshold_good_box=0.35 \
  experiment_name=fipseg_yv5_640_c35

echo; echo "### [2/3] YOLOv5 @ 1280 + SAM1 -> yolo_sam_v1/fipseg_yv5_1280_c35"
python "$RUN" "${COMMON[@]}" method=yolo_sam_v1 \
  method.target_image_size=1280 \
  method.sam_backend=sam1 \
  method.conf_threshold_good_box=0.35 \
  experiment_name=fipseg_yv5_1280_c35

echo; echo "### [3/3] YOLO11 (yolo11_sam, imgsz 3008) + SAM1 -> yolo11_sam/fipseg_yolo11_c35"
python "$RUN" "${COMMON[@]}" method=yolo11_sam \
  method.sam_backend=sam1 \
  method.conf_threshold_good_box=0.35 \
  experiment_name=fipseg_yolo11_c35

echo; echo "DONE. Seg-ready masks under results/mask_generation/fip/plot_46*/{yolo_sam_v1/fipseg_yv5_640_c35,yolo_sam_v1/fipseg_yv5_1280_c35,yolo11_sam/fipseg_yolo11_c35}/masks/"
