#!/usr/bin/env bash
# Extracts the first frame of the 3D-segmentation 360 fly-around videos used as
# thesis figures: the phone run (fig:phone-seg-render) and the FIP run
# (fig:fip-seg-render). Each colored head is one per-instance 3D segment.
set -euo pipefail
PHONE="results/reconstruction/phone/field_A/20250715/opencv/vanilla_3dgs/baseline/segmentation_3d/ocv15k_yolov5_pertile/wheat_field_360_h264.mp4"
FIP="results/reconstruction/fip/plot_461/vanilla_3dgs/initial/segmentation_3d/run_1/wheat_field_360.mp4"
ffmpeg -y -i "$PHONE" -vf "select=eq(n\,0)" -vframes 1 thesis/figures/phone_seg_render.png -loglevel error
ffmpeg -y -i "$FIP"   -vf "select=eq(n\,0)" -vframes 1 thesis/figures/fip_seg_render.png   -loglevel error
echo "wrote thesis/figures/{phone,fip}_seg_render.png"
