"""Preprocess phone images so they all share the same resolution.

Phone cameras (especially with HDR auto-switching) often output a mix of
slightly different sizes — e.g. 3850x2928 for most shots but 3852x2936 for HDR.
This breaks COLMAP's --single_camera mode and makes the mapper split the
reconstruction into separate sub-models (one per intrinsic group).

Fix: center-crop the larger images down to the majority size. Cropping (not
resizing) preserves focal length and optical center, only trims a few border
pixels. Output goes to a new folder so the originals are untouched.

Usage:
    python src/preprocessing/preprocess_uniform_size.py source=input_plots/phone/field_A/20250618/input
    python src/preprocessing/preprocess_uniform_size.py source=path/to/input output=path/to/output

Config: configs/preprocessing/uniform_size.yaml
"""

import os
import shutil
from collections import Counter

import hydra
from omegaconf import DictConfig, OmegaConf
from PIL import Image


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing/uniform_size")
def main(cfg: DictConfig):
    """Read images from cfg.source, find the majority resolution, center-crop the outliers
    to match it, and write everything to cfg.output. Copies of already-correct images are not re-encoded."""
    print("--- uniform_size config ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------")

    src = os.path.abspath(cfg.source)
    dst = cfg.output if cfg.output else src.rstrip("/") + "_uniform"

    # 1. scan source folder to find the majority dimension (most common = target)
    sizes = Counter()
    files = [f for f in sorted(os.listdir(src)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for f in files:
        sizes.update([Image.open(os.path.join(src, f)).size])

    if len(sizes) == 1:
        # all images already uniform — point dst at src via symlink (zero disk cost) so convert.py's
        # default image_subdir=input_uniform just works without forcing the user to override.
        if os.path.islink(dst):
            print(f"All {len(files)} images already uniform ({list(sizes)[0]}). Symlink already exists: {dst}")
            return
        if os.path.exists(dst):
            print(f"All {len(files)} images already uniform ({list(sizes)[0]}). {dst} exists as a real folder — leaving it as-is.")
            return
        src_parent = os.path.dirname(src)
        dst_parent = os.path.dirname(os.path.abspath(dst))
        # relative symlink if same parent (more portable), absolute path otherwise
        link_target = os.path.basename(src) if src_parent == dst_parent else src
        os.symlink(link_target, dst)
        print(f"All {len(files)} images already uniform ({list(sizes)[0]}). Created symlink: {dst} → {link_target}")
        return

    # if a stale symlink exists from a previous "all uniform" run, remove it before writing real files —
    # otherwise we'd overwrite the source through the link.
    if os.path.islink(dst):
        print(f"Removing stale symlink at {dst} before writing cropped images.")
        os.unlink(dst)
    os.makedirs(dst, exist_ok=True)

    target_w, target_h = sizes.most_common(1)[0][0]
    print(f"Found {len(sizes)} different sizes: {dict(sizes)}")
    print(f"Majority (target): {target_w}x{target_h}")
    print(f"Cropping all others to {target_w}x{target_h}...")

    # 2. copy or center-crop each image into dst
    n_copied, n_cropped = 0, 0
    for f in files:
        src_path = os.path.join(src, f)
        # resolve symlinks so we read the real file (and don't write through a link)
        real_path = os.path.realpath(src_path)
        dst_path = os.path.join(dst, f)
        img = Image.open(real_path)
        w, h = img.size
        if (w, h) == (target_w, target_h):
            shutil.copy2(real_path, dst_path)
            n_copied += 1
        else:
            # center crop: trim equally from left/right and top/bottom
            left = (w - target_w) // 2
            top = (h - target_h) // 2
            img.crop((left, top, left + target_w, top + target_h)).save(dst_path, quality=95)
            n_cropped += 1

    print(f"Done. Copied as-is: {n_copied}, center-cropped: {n_cropped}")
    print(f"Output folder: {dst}")


if __name__ == "__main__":
    main()
