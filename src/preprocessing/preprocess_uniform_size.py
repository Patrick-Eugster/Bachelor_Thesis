"""Preprocess phone images so they all share the same resolution.

Phone cameras (especially with HDR auto-switching) often output a mix of
slightly different sizes — e.g. 3850x2928 for most shots but 3852x2936 for HDR.
This breaks COLMAP's --single_camera mode and makes the mapper split the
reconstruction into separate sub-models (one per intrinsic group).

Fix: center-crop the larger images down to the majority size. Cropping (not
resizing) preserves focal length and optical center, only trims a few border
pixels. Output goes to a new folder so the originals are untouched.

Output format (cfg.output_format, default "png"): input_uniform/ is written as LOSSLESS PNG so that
COLMAP's image_undistorter (which mirrors the input extension) writes images/ as PNG too — stopping
the generation-2 JPEG recompression that otherwise hits the images 3DGS + mask-gen + seg all read.
This costs disk (PNG is ~5x the JPEG here) and drops the zero-cost symlink shortcut. Set
output_format=keep for the old behaviour (symlink when uniform / quality="keep" JPEG crop).

Usage:
    python src/preprocessing/preprocess_uniform_size.py source=input_plots/phone/field_A/20250618/input
    python src/preprocessing/preprocess_uniform_size.py source=path/to/input output=path/to/output

Config: configs/preprocessing/uniform_size.yaml
"""

import json
import os
import shutil
import time
from collections import Counter

import hydra
from omegaconf import DictConfig, OmegaConf
from PIL import Image


def _write_summary_json(source_path, summary):
    """Drop a small summary JSON in {source_path}/logs/ so the orchestrator can read it back."""
    log_dir = os.path.join(source_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "uniform_size_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _print_summary(field, plot, n_images, mode, target_size, n_copied, n_cropped, size_counts, elapsed):
    """Yolo_sam-style summary block. mode is one of: 'symlink_created', 'symlink_existed',
    'real_dir_existed', 'cropped', 'png_converted', 'png_cropped'."""
    minutes, seconds = divmod(int(elapsed), 60)
    print("\n" + "="*50)
    print("      UNIFORM-SIZE SUMMARY")
    print("="*50)
    print(f"{'Plot:':<28} {field}/{plot}")
    print(f"{'Total images:':<28} {n_images}")
    print(f"{'Source sizes:':<28} {dict(size_counts) if size_counts else '-'}")
    print(f"{'Target size:':<28} {target_size if target_size else '-'}")
    print(f"{'Outcome:':<28} {mode}")
    if n_cropped > 0 or n_copied > 0:
        print(f"{'Copied as-is:':<28} {n_copied}")
        print(f"{'Center-cropped:':<28} {n_cropped}")
    print("-" * 50)
    print(f"{'TOTAL TIME:':<28} {minutes}m {seconds}s  ({elapsed:.1f}s)")
    print("="*50 + "\n")


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing/uniform_size")
def main(cfg: DictConfig):
    """Read images from cfg.source, find the majority resolution, center-crop the outliers
    to match it, and write everything to cfg.output. Copies of already-correct images are not re-encoded."""
    print("--- uniform_size config ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------")

    t_start = time.time()
    src = os.path.abspath(cfg.source)
    dst = cfg.output if cfg.output else src.rstrip("/") + "_uniform"
    field = cfg.get("field", "?")
    plot = cfg.get("plot", "?")
    # the source_path used by the rest of the pipeline is src's parent (input/ lives inside it)
    source_path = os.path.dirname(src)

    def _save_summary(mode, target_size, n_copied, n_cropped):
        """Write the per-step JSON the orchestrator will pick up."""
        _write_summary_json(source_path, {
            "step": "uniform_size",
            "field": field, "plot": plot,
            "n_images": len(files),
            "mode": mode,
            "target_size": target_size,
            "source_sizes": {f"{w}x{h}": c for (w, h), c in sizes.items()},
            "n_copied": n_copied,
            "n_cropped": n_cropped,
            "elapsed_s": time.time() - t_start,
        })

    # 1. scan source folder to find the majority dimension (most common = target)
    sizes = Counter()
    files = [f for f in sorted(os.listdir(src)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for f in files:
        sizes.update([Image.open(os.path.join(src, f)).size])

    fmt = str(cfg.get("output_format", "png")).lower()   # "png" (lossless default) or "keep"
    uniform = (len(sizes) == 1)
    target_w, target_h = sizes.most_common(1)[0][0]
    target_str = f"{target_w}x{target_h}"

    # --- "keep" mode + already uniform: symlink input_uniform→input (zero disk), the old fast path.
    # PNG mode can't take this shortcut because it must actually re-encode each image to PNG. ---
    if fmt == "keep" and uniform:
        only_size = (target_w, target_h)
        if os.path.islink(dst):
            print(f"All {len(files)} images already uniform ({only_size}). Symlink already exists: {dst}")
            _print_summary(field, plot, len(files), "symlink_existed", target_str, 0, 0, sizes, time.time() - t_start)
            _save_summary("symlink_existed", target_str, 0, 0)
            return
        if os.path.exists(dst):
            print(f"All {len(files)} images already uniform ({only_size}). {dst} exists as a real folder — leaving it as-is.")
            _print_summary(field, plot, len(files), "real_dir_existed", target_str, 0, 0, sizes, time.time() - t_start)
            _save_summary("real_dir_existed", target_str, 0, 0)
            return
        src_parent = os.path.dirname(src)
        dst_parent = os.path.dirname(os.path.abspath(dst))
        # relative symlink if same parent (more portable), absolute path otherwise
        link_target = os.path.basename(src) if src_parent == dst_parent else src
        os.symlink(link_target, dst)
        print(f"All {len(files)} images already uniform ({only_size}). Created symlink: {dst} → {link_target}")
        _print_summary(field, plot, len(files), "symlink_created", target_str, 0, 0, sizes, time.time() - t_start)
        _save_summary("symlink_created", target_str, 0, 0)
        return

    # --- otherwise we materialize real files in dst ---
    # PNG mode: nuke any stale dst first, so the folder COLMAP reads never mixes leftover .jpg with the
    # new .png (COLMAP would otherwise extract features from BOTH). A stale symlink is also removed so we
    # don't write through it onto the originals.
    if os.path.islink(dst):
        print(f"Removing stale symlink at {dst} before writing images.")
        os.unlink(dst)
    elif fmt == "png" and os.path.isdir(dst):
        print(f"Clearing existing {dst} before writing fresh PNGs.")
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)

    if not uniform:
        print(f"Found {len(sizes)} different sizes: {dict(sizes)}")
        print(f"Majority (target): {target_w}x{target_h}")
        print(f"Cropping all others to {target_w}x{target_h}...")
    if fmt == "png":
        print("Writing input_uniform/ as LOSSLESS PNG — stops COLMAP's generation-2 JPEG recompression "
              "so images/ (used by 3DGS + mask-gen + seg) stays lossless from the phone original.")

    # 2. copy / crop / convert each image into dst
    n_copied, n_cropped = 0, 0
    crop_px_total = 0  # accumulate trimmed pixels for the summary
    for f in files:
        src_path = os.path.join(src, f)
        # resolve symlinks so we read the real file (and don't write through a link)
        real_path = os.path.realpath(src_path)
        stem, ext = os.path.splitext(f)
        ext = ext.lower()
        # PNG mode changes the extension; downstream (COLMAP names, mask-gen glob, 3DGS) is ext-agnostic
        out_name = stem + ".png" if fmt == "png" else f
        dst_path = os.path.join(dst, out_name)

        img = Image.open(real_path)
        w, h = img.size
        need_crop = (w, h) != (target_w, target_h)
        if need_crop:
            # center crop: trim equally from left/right and top/bottom (preserves focal + optical center)
            left = (w - target_w) // 2
            top = (h - target_h) // 2
            img = img.crop((left, top, left + target_w, top + target_h))
            n_cropped += 1
            crop_px_total += (w - target_w) + (h - target_h)  # px trimmed per image (w_diff + h_diff)
        else:
            n_copied += 1

        # Save. Goal: never ADD lossy compression on top of the phone's already-lossy JPEG.
        # - PNG mode: always re-encode to lossless PNG (a JPEG→PNG decode is lossless; PNG stores the
        #   decoded pixels exactly). This is the whole point — the resulting images/ won't be re-JPEGed.
        # - keep mode, no crop: byte-identical copy (no re-encode at all).
        # - keep mode + crop: JPEG uses quality="keep" (inherit source quant tables); PNG lossless.
        if fmt == "png":
            img.convert("RGB").save(dst_path, compress_level=6)   # lossless deflate, decent size
        elif not need_crop:
            shutil.copy2(real_path, dst_path)
        elif ext in (".jpg", ".jpeg"):
            img.save(dst_path, quality="keep")
        elif ext == ".png":
            img.save(dst_path, compress_level=0)  # zero PNG deflate = lossless + fast
        else:
            img.save(dst_path)  # let PIL pick lossless defaults for other formats

    print(f"Done. Copied/converted as-is: {n_copied}, center-cropped: {n_cropped}")
    if n_cropped > 0:
        print(f"  Avg trim per cropped image: {crop_px_total / n_cropped:.1f} px (width+height combined)")
    print(f"Output folder: {dst}")
    mode = ("png_cropped" if n_cropped else "png_converted") if fmt == "png" else "cropped"
    _print_summary(field, plot, len(files), mode, target_str, n_copied, n_cropped, sizes, time.time() - t_start)
    _save_summary(mode, target_str, n_copied, n_cropped)


if __name__ == "__main__":
    main()
