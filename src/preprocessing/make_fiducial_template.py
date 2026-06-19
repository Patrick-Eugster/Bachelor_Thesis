"""Crop a REAL fiducial from one photo and save it as a grayscale template for v6 matching.

Why: v5 matched a SYNTHETIC bullseye (drawn in code) — too crude, so real fiducials only reached
NCC ~0.73 (barely above canopy). A patch of REAL pixels (true grey, soft edges, the partial coded
ring, sensor noise) correlates far more strongly with real fiducials and far less with canopy,
widening the margin and landing the peak on the fiducial center.

How it stays scale-able: we crop a window of (template_margin × fiducial_radius) around a known-
good center, then resize so the saved template has a CANONICAL disk radius (canon_radius px). v6
then resizes this one template to every size in its bank (resize factor = wanted_radius/canon).

The center + radius come from a v4 detection (v4 centers are reliable — "100% fiducial-snapped"),
so no manual clicking is needed.

Usage (defaults to the clean frontal fiducial we picked from the v4 contact sheet):
    python src/preprocessing/make_fiducial_template.py
    python src/preprocessing/make_fiducial_template.py image=IMG_20250609_112214.jpg cx=2367.9 cy=2067.8 radius=36.7
"""

import os

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing/make_fiducial_template")
def main(cfg: DictConfig):
    """Crop the fiducial window around (cx, cy), normalize it to a canonical disk radius, save it."""
    src_img = os.path.join(cfg.dataset.input_dir, str(cfg.field), str(cfg.plot), "images", cfg.image)
    img = cv2.imread(src_img)
    if img is None:
        print(f"ERROR: could not read {src_img}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # crop a square window = template_margin × radius around the center (same margin v6 expects)
    half = int(round(cfg.radius * cfg.template_margin))
    x, y = int(round(cfg.cx)), int(round(cfg.cy))
    H, W = gray.shape[:2]
    x0, y0 = max(0, x - half), max(0, y - half)
    x1, y1 = min(W, x + half), min(H, y + half)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        print("ERROR: empty crop (center off-image?)")
        return

    # resize so the saved template's disk radius == canon_radius (so v6 can rescale by a clean ratio)
    canon_half = int(round(cfg.canon_radius * cfg.template_margin))
    canon_size = 2 * canon_half + 1
    tmpl = cv2.resize(patch, (canon_size, canon_size), interpolation=cv2.INTER_AREA)

    out_path = os.path.join(hydra.utils.get_original_cwd(), cfg.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, tmpl)
    # also save a 4x preview so it's easy to eyeball
    cv2.imwrite(out_path.replace(".png", "_preview.png"),
                cv2.resize(tmpl, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
    print(f"Saved real fiducial template ({canon_size}x{canon_size}, canonical disk radius "
          f"{cfg.canon_radius} px) → {out_path}")
    print(f"  source: {cfg.image} @ ({cfg.cx:.1f}, {cfg.cy:.1f}), measured radius {cfg.radius} px")
    print(f"  preview (4x): {out_path.replace('.png', '_preview.png')}")


if __name__ == "__main__":
    main()
