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

import json
import os
import struct
import subprocess
import time
import shutil

import hydra
from omegaconf import DictConfig, OmegaConf


def _count_images_in_submodel(submodel_dir):
    """Return the image count in a COLMAP sub-model folder. Reads the binary header from images.bin
    (faster than parsing text) — first 8 bytes are uint64 num_images. Falls back to 0 if missing."""
    images_bin = os.path.join(submodel_dir, "images.bin")
    if not os.path.isfile(images_bin):
        return 0
    with open(images_bin, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def _pick_largest_submodel(distorted_sparse_dir):
    """Scan distorted/sparse/<n>/ folders and return (best_subdir, best_count, all_counts).
    The mapper sometimes spawns a stray small sub-model (e.g. 2 outliers) alongside the real
    reconstruction — we always want the largest one for undistortion."""
    candidates = []
    for name in sorted(os.listdir(distorted_sparse_dir)):
        sub = os.path.join(distorted_sparse_dir, name)
        if os.path.isdir(sub):
            candidates.append((name, sub, _count_images_in_submodel(sub)))
    if not candidates:
        return None, 0, []
    best = max(candidates, key=lambda c: c[2])
    return best[1], best[2], [(c[0], c[2]) for c in candidates]


def _registered_image_names(sparse0_dir):
    """Read the names of the images COLMAP actually registered, as stems (no extension).
    Prefers images.txt (export_text writes it); the non-comment lines alternate header/points2D,
    so every 2nd line's last token is the image filename. Returns a set; empty if unreadable."""
    txt = os.path.join(sparse0_dir, "images.txt")
    if os.path.isfile(txt):
        with open(txt) as f:
            nonc = [l for l in f if l.strip() and not l.startswith("#")]
        return {os.path.basename(nonc[i].split()[-1]).split(".")[0] for i in range(0, len(nonc), 2)}
    return set()


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
        # Mapper.num_threads caps bundle adjustment + triangulation worker threads — without it
        # the mapper grabs every core, which can spike RAM on dense scenes. We reuse cfg.num_threads
        # (default 8) so SIFT and mapper share the same cap.
        mapper_cmd = (colmap_command + " mapper \
            --database_path " + source_path + "/distorted/database.db \
            --image_path " + image_path + " \
            --output_path " + source_path + "/distorted/sparse \
            --Mapper.num_threads " + str(cfg.num_threads) + " \
            --Mapper.ba_global_function_tolerance=0.000001")
        exit_code = run_cmd(mapper_cmd)
        if exit_code != 0:
            print(f"ERROR: Mapper failed with code {exit_code}. Exiting.")
            log_file.close()
            exit(exit_code)
        print(f"  Mapper done in {time.time() - t0:.1f}s")

    # 4. Image undistortion — rewrites images to ideal pinhole intrinsics.
    # Pick the largest sub-model: the mapper occasionally spawns a stray small sub-model
    # (e.g. 2-image outlier blob) alongside the real reconstruction. Hardcoding "0" would
    # undistort the wrong one on those sessions.
    distorted_sparse = os.path.join(source_path, "distorted", "sparse")
    best_sub, best_count, all_subs = _pick_largest_submodel(distorted_sparse)
    if best_sub is None:
        print(f"ERROR: no sub-models in {distorted_sparse} — mapper produced no reconstruction.")
        log_file.close()
        exit(1)
    submodel_summary = ", ".join(f"{name}={n}" for name, n in all_subs)
    if len(all_subs) > 1:
        print(f"WARNING: mapper produced {len(all_subs)} sub-models ({submodel_summary}) — undistorting the largest one ({os.path.basename(best_sub)}, {best_count} images).")
    else:
        print(f"Mapper produced 1 sub-model ({submodel_summary}). Undistorting it.")
    # remember for the summary at the end
    mapper_submodels = all_subs
    undistorted_count = best_count
    print("Undistorting images...")
    t0 = time.time()
    img_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + image_path + " \
        --input_path " + best_sub + " \
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

    # input image count (what we fed to feature_extractor) — symlinks are followed by os.listdir
    try:
        input_files = [f for f in os.listdir(image_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    except OSError:
        input_files = []
    n_input = len(input_files)

    # Which input images did COLMAP FAIL to register? Mark them explicitly: a missing image silently
    # drops out of the downstream train/test split, so two methods that register different sets are
    # no longer comparable. We list the names so the drift is visible (and check_split.py can flag it).
    input_stems = {f.split(".")[0] for f in input_files}
    registered_stems = _registered_image_names(os.path.join(source_path, "sparse", "0"))
    missing = sorted(input_stems - registered_stems) if registered_stems else []

    minutes, seconds = divmod(int(elapsed), 60)
    print("\n" + "="*50)
    print("      COLMAP SUMMARY")
    print("="*50)
    print(f"{'Plot:':<28} {cfg.field}/{cfg.plot}")
    print(f"{'Camera model:':<28} {cfg.camera}  (single_camera={cfg.single_camera})")
    print(f"{'Matcher:':<28} {cfg.matcher}")
    print(f"{'GPU enabled (SIFT+match):':<28} {not cfg.no_gpu}")
    print(f"{'Threads (feat+match+map):':<28} {cfg.num_threads}")
    print("-" * 50)
    print(f"{'Input images:':<28} {n_input}")
    print(f"{'Sub-models from mapper:':<28} {len(mapper_submodels)}  ({', '.join(f'{n}={c}' for n, c in mapper_submodels)})")
    print(f"{'Registered in largest:':<28} {undistorted_count} / {n_input}")
    if n_input > 0:
        print(f"{'Registration rate:':<28} {100.0*undistorted_count/n_input:.1f}%")
    if missing:
        print("-" * 50)
        print(f"WARNING: {len(missing)} input image(s) NOT registered (excluded from the model + split):")
        print(f"  {missing}")
    print("-" * 50)
    print(f"{'TOTAL TIME:':<28} {minutes}m {seconds}s  ({elapsed:.0f}s)")
    print("="*50 + "\n")

    log_file.write(f"Total time: {elapsed/60:.1f} min ({elapsed:.0f}s)\n")
    log_file.write(f"Registered {undistorted_count}/{n_input} in {len(mapper_submodels)} sub-models\n")
    log_file.close()

    # drop a JSON summary the orchestrator can pick up
    summary = {
        "step": "colmap",
        "field": cfg.field,
        "plot": cfg.plot,
        "input_images": n_input,
        "submodels": [{"name": n, "images": c} for n, c in mapper_submodels],
        "registered": undistorted_count,
        "missing_images": missing,
        "camera": cfg.camera,
        "single_camera": cfg.single_camera,
        "matcher": cfg.matcher,
        "gpu": not cfg.no_gpu,
        "num_threads": cfg.num_threads,
        "elapsed_s": elapsed,
    }
    with open(os.path.join(log_dir, "colmap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    main()
