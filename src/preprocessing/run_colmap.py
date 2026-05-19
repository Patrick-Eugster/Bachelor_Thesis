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
# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
# Adapted to use Hydra for configuration. See configs/preprocessing/colmap.yaml for parameters.
# (Originally named convert.py — renamed to run_colmap.py since it runs the full COLMAP SfM pipeline,
# not just a format conversion.)

import os
import subprocess
import time
import shutil

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing/colmap")
def main(cfg: DictConfig):
    """Run COLMAP SfM on phone images: feature_extractor -> matcher -> mapper -> image_undistorter.
    Produces images/ + sparse/0/ in source_path, ready for 3DGS training."""
    print("--- COLMAP config ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

    colmap_command = f'"{cfg.colmap_executable}"' if len(cfg.colmap_executable) > 0 else "colmap"
    magick_command = f'"{cfg.magick_executable}"' if len(cfg.magick_executable) > 0 else "magick"
    use_gpu = 0 if cfg.no_gpu else 1
    source_path = cfg.source_path
    # full path to the folder COLMAP reads images from
    image_path = os.path.join(source_path, cfg.image_subdir)

    # ensure log folder exists, open the log file we'll tee into
    log_dir = os.path.join(source_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "colmap.log")
    log_file = open(log_path, "w")
    print(f"Logging all output to {log_path}")

    # save the config snapshot next to the log so each run is self-documenting
    with open(os.path.join(log_dir, "colmap_config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    def run_cmd(cmd):
        """Run a shell command, printing output to terminal and saving to log file simultaneously.
        Uses shell `tee -a` so COLMAP output bypasses Python — same speed as native, no per-line overhead."""
        log_file.write(f"\n$ {cmd}\n")
        log_file.flush()
        full_cmd = f"({cmd}) 2>&1 | tee -a {log_path}"
        return subprocess.call(full_cmd, shell=True, executable="/bin/bash")

    t_start = time.time()

    if not cfg.skip_matching:
        os.makedirs(source_path + "/distorted/sparse", exist_ok=True)

        # 1. Feature extraction
        print("Step 1/3: Feature extraction...")
        t0 = time.time()
        single_camera_flag = "1" if cfg.single_camera else "0"
        feat_extracton_cmd = colmap_command + " feature_extractor "\
            "--database_path " + source_path + "/distorted/database.db \
            --image_path " + image_path + " \
            --ImageReader.camera_model " + cfg.camera + " \
            --ImageReader.single_camera " + single_camera_flag + " \
            --FeatureExtraction.use_gpu " + str(use_gpu) + " \
            --FeatureExtraction.num_threads " + str(cfg.num_threads)
        exit_code = run_cmd(feat_extracton_cmd)
        if exit_code != 0:
            print(f"ERROR: Feature extraction failed with code {exit_code}. Exiting.")
            log_file.close()
            exit(exit_code)
        print(f"  Feature extraction done in {time.time() - t0:.1f}s")

        # 2. Feature matching
        print(f"Step 2/3: Feature matching ({cfg.matcher})...")
        t0 = time.time()
        if cfg.matcher == "sequential":
            feat_matching_cmd = colmap_command + " sequential_matcher \
                --database_path " + source_path + "/distorted/database.db \
                --FeatureMatching.use_gpu " + str(use_gpu) + " \
                --FeatureMatching.num_threads " + str(cfg.num_threads) + " \
                --SequentialMatching.overlap " + str(cfg.sequential_overlap)
        else:
            feat_matching_cmd = colmap_command + " exhaustive_matcher \
                --database_path " + source_path + "/distorted/database.db \
                --FeatureMatching.use_gpu " + str(use_gpu) + " \
                --FeatureMatching.num_threads " + str(cfg.num_threads)
        exit_code = run_cmd(feat_matching_cmd)
        if exit_code != 0:
            print(f"ERROR: Feature matching failed with code {exit_code}. Exiting.")
            log_file.close()
            exit(exit_code)
        print(f"  Feature matching done in {time.time() - t0:.1f}s")

        # 3. Bundle adjustment / mapper. The default Mapper tolerance is unnecessarily large;
        # decreasing it speeds up bundle adjustment.
        print("Step 3/3: Mapping (SfM + bundle adjustment)...")
        t0 = time.time()
        mapper_cmd = (colmap_command + " mapper \
            --database_path " + source_path + "/distorted/database.db \
            --image_path " + image_path + " \
            --output_path " + source_path + "/distorted/sparse \
            --Mapper.ba_global_function_tolerance=0.000001")
        exit_code = run_cmd(mapper_cmd)
        if exit_code != 0:
            print(f"ERROR: Mapper failed with code {exit_code}. Exiting.")
            log_file.close()
            exit(exit_code)
        print(f"  Mapper done in {time.time() - t0:.1f}s")

    # 4. Image undistortion — rewrites images to ideal pinhole intrinsics.
    print("Undistorting images...")
    t0 = time.time()
    img_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + image_path + " \
        --input_path " + source_path + "/distorted/sparse/0 \
        --output_path " + source_path + "\
        --output_type COLMAP")
    exit_code = run_cmd(img_undist_cmd)
    if exit_code != 0:
        print(f"ERROR: Image undistortion failed with code {exit_code}. Exiting.")
        log_file.close()
        exit(exit_code)
    print(f"  Undistortion done in {time.time() - t0:.1f}s")

    # COLMAP writes sparse files directly into sparse/ — 3DGS expects sparse/0/ so we move them.
    files = os.listdir(source_path + "/sparse")
    os.makedirs(source_path + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(source_path, "sparse", file)
        destination_file = os.path.join(source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    # Also export sparse/0/ as text alongside the .bin files (human-readable, easier to diff vs Agisoft).
    # COLMAP's model_converter writes cameras.txt, images.txt, points3D.txt next to the .bin files —
    # both formats coexist; .bin stays the canonical input for 3DGS.
    if cfg.export_text:
        print("Exporting sparse/0/ as text (cameras.txt, images.txt, points3D.txt)...")
        sparse0 = os.path.join(source_path, "sparse", "0")
        export_cmd = f"{colmap_command} model_converter --input_path {sparse0} --output_path {sparse0} --output_type TXT"
        exit_code = run_cmd(export_cmd)
        if exit_code != 0:
            print(f"WARNING: text export failed with code {exit_code} (continuing anyway — .bin still produced)")

    if cfg.resize:
        print("Copying and resizing...")

        os.makedirs(source_path + "/images_2", exist_ok=True)
        os.makedirs(source_path + "/images_4", exist_ok=True)
        os.makedirs(source_path + "/images_8", exist_ok=True)
        files = os.listdir(source_path + "/images")
        for file in files:
            source_file = os.path.join(source_path, "images", file)

            destination_file = os.path.join(source_path, "images_2", file)
            shutil.copy2(source_file, destination_file)
            exit_code = run_cmd(magick_command + " mogrify -resize 50% " + destination_file)
            if exit_code != 0:
                print(f"ERROR: 50% resize failed with code {exit_code}. Exiting.")
                log_file.close()
                exit(exit_code)

            destination_file = os.path.join(source_path, "images_4", file)
            shutil.copy2(source_file, destination_file)
            exit_code = run_cmd(magick_command + " mogrify -resize 25% " + destination_file)
            if exit_code != 0:
                print(f"ERROR: 25% resize failed with code {exit_code}. Exiting.")
                log_file.close()
                exit(exit_code)

            destination_file = os.path.join(source_path, "images_8", file)
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


if __name__ == "__main__":
    main()
