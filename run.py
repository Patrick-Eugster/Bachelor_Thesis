import subprocess
import sys

# --- TOGGLE PIPELINE STAGES ---
RUN_YOLO_SAM       = False  # Phase 1: YOLO detection + SAM segmentation
RUN_RECONSTRUCTION = False  # Phase 2: 3DGS train, segment, eval, viewer


def run(cmd):
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"!!! ERROR: {' '.join(cmd)} failed with code {result.returncode}")
        sys.exit(1)


if RUN_YOLO_SAM:
    run(["python", "src/mask_generation/yolo_sam_v1/main_v1.py"])

if RUN_RECONSTRUCTION:
    run(["python", "src/run_reconstruction.py"])
