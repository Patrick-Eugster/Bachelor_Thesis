import subprocess
import sys

# --- TOGGLE PIPELINE STAGES ---
RUN_YOLO_SAM       = False  # Phase 1: YOLO detection + SAM segmentation
RUN_RECONSTRUCTION = False  # Phase 2: 3DGS train, segment, eval, viewer

# Any extra CLI args (e.g. dataset=phone experiment_name=my_run) are passed through to both scripts
extra_args = sys.argv[1:]


def run(cmd):
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"!!! ERROR: {' '.join(cmd)} failed with code {result.returncode}")
        sys.exit(1)


if RUN_YOLO_SAM:
    run(["python", "src/mask_generation/yolo_sam_v1/main_v1.py"] + extra_args)

if RUN_RECONSTRUCTION:
    run(["python", "src/run_reconstruction.py"] + extra_args)
