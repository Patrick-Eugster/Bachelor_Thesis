#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import subprocess
import time
from argparse import ArgumentParser
import shutil

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true', default=True)
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="SIMPLE_PINHOLE", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
# sequential: matches each image only against the next N images — low RAM, good for walk/video sequences
# exhaustive: matches every image against every other — high RAM, use only for unordered image sets
parser.add_argument("--matcher", default="sequential", choices=["sequential", "exhaustive"], type=str)
parser.add_argument("--sequential_overlap", default=25, type=int,
                    help="sequential matcher only: how many next images each image is matched against")
parser.add_argument("--num_threads", default=8, type=int,
                    help="threads for SIFT extraction + matching. Lower = less RAM (default 8, set -1 for all cores)")
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

log_dir = os.path.join(args.source_path, "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "colmap.log")
log_file = open(log_path, "w")
print(f"Logging all output to {log_path}")

def run_cmd(cmd):
    """Run a shell command, printing output to terminal and saving to log file simultaneously.
    Uses shell `tee -a` so COLMAP output bypasses Python — same speed as native, no per-line overhead."""
    log_file.write(f"\n$ {cmd}\n")
    log_file.flush()
    full_cmd = f"({cmd}) 2>&1 | tee -a {log_path}"
    return subprocess.call(full_cmd, shell=True, executable="/bin/bash")

t_start = time.time()

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    print("Step 1/3: Feature extraction...")
    t0 = time.time()
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu) + " \
        --SiftExtraction.num_threads " + str(args.num_threads)
    exit_code = run_cmd(feat_extracton_cmd)
    if exit_code != 0:
        print(f"ERROR: Feature extraction failed with code {exit_code}. Exiting.")
        log_file.close()
        exit(exit_code)
    print(f"  Feature extraction done in {time.time() - t0:.1f}s")

    ## Feature matching
    print(f"Step 2/3: Feature matching ({args.matcher})...")
    t0 = time.time()
    if args.matcher == "sequential":
        feat_matching_cmd = colmap_command + " sequential_matcher \
            --database_path " + args.source_path + "/distorted/database.db \
            --SiftMatching.use_gpu " + str(use_gpu) + " \
            --SiftMatching.num_threads " + str(args.num_threads) + " \
            --SequentialMatching.overlap " + str(args.sequential_overlap)
    else:
        feat_matching_cmd = colmap_command + " exhaustive_matcher \
            --database_path " + args.source_path + "/distorted/database.db \
            --SiftMatching.use_gpu " + str(use_gpu) + " \
            --SiftMatching.num_threads " + str(args.num_threads)
    exit_code = run_cmd(feat_matching_cmd)
    if exit_code != 0:
        print(f"ERROR: Feature matching failed with code {exit_code}. Exiting.")
        log_file.close()
        exit(exit_code)
    print(f"  Feature matching done in {time.time() - t0:.1f}s")

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    print("Step 3/3: Mapping (SfM + bundle adjustment)...")
    t0 = time.time()
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = run_cmd(mapper_cmd)
    if exit_code != 0:
        print(f"ERROR: Mapper failed with code {exit_code}. Exiting.")
        log_file.close()
        exit(exit_code)
    print(f"  Mapper done in {time.time() - t0:.1f}s")

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
print("Undistorting images...")
t0 = time.time()
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = run_cmd(img_undist_cmd)
if exit_code != 0:
    print(f"ERROR: Image undistortion failed with code {exit_code}. Exiting.")
    log_file.close()
    exit(exit_code)
print(f"  Undistortion done in {time.time() - t0:.1f}s")

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = run_cmd(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            print(f"ERROR: 50% resize failed with code {exit_code}. Exiting.")
            log_file.close()
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = run_cmd(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            print(f"ERROR: 25% resize failed with code {exit_code}. Exiting.")
            log_file.close()
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = run_cmd(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            print(f"ERROR: 12.5% resize failed with code {exit_code}. Exiting.")
            log_file.close()
            exit(exit_code)

elapsed = time.time() - t_start
msg = f"Done. Total time: {elapsed/60:.1f} min ({elapsed:.0f}s)"
print(msg)
log_file.write(msg + "\n")
log_file.close()
