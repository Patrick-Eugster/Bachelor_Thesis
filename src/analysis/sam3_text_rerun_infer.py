"""ONE-SHOT local re-run of SAM3 text-prompt on a single phone frame, to recover the per-instance
masks that the original probe discarded (it saved only union overlays + counts). Runs ``wheat`` at
conf 0.25 on the full frame and on each of the four non-overlapping 2x2 tiles, and saves the RAW
per-instance masks, boxes, and scores to .npz --- one file per panel --- so the appendix figure can
be rendered per-instance and re-rendered later without ever touching the GPU again.

Reuses the exact inference path from sam3_text_probe.py (build_predictor / result_masks / tiles_of),
which already ran on Euler, so nothing about the model call changes. Runs FP16 (quantize=16) to fit
the local 16 GB card; SAM3 encodes at imgsz 1008 regardless of panel size, so every panel costs the
same VRAM. Each panel is saved immediately after its inference, so a crash on a later panel still
keeps the earlier results.

Run ONCE: python src/analysis/sam3_text_rerun_infer.py
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

# reuse the proven probe helpers (result_masks / tiles_of ran on Euler unchanged); build the
# predictor here with half=True. The probe's build_predictor used quantize=16 for FP16, which this
# ultralytics version (8.4.21) does not accept --- its FP16 flag is half=True.
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from sam3_text_probe import result_masks, tiles_of  # noqa: E402
from ultralytics.models.sam import SAM3SemanticPredictor  # noqa: E402


def build_fp16_predictor(weight):
    """Builds the SAM3 text/concept predictor in FP16 so it fits the local 16 GB card."""
    overrides = {"model": weight, "task": "segment", "mode": "predict", "conf": CONF,
                 "save": False, "verbose": False, "project": "runs_sam3_probe",
                 "imgsz": IMGSZ, "half": True}
    return SAM3SemanticPredictor(overrides=overrides)

IMG = "input_plots/phone/field_A/20250715/images/IMG_20250715_153912.jpg"
PHRASE = "wheat"
CONF = 0.25
IMGSZ = 1008
OUT = "results/sam3_text_probe_rerun"
TMP = os.path.join(OUT, "_tmp")


def panels(img_rgb):
    """The five panels we need: the whole frame, then the four 2x2 tiles, each tagged."""
    yield from tiles_of(img_rgb, 1)   # ('full', 0, 0, whole)
    yield from tiles_of(img_rgb, 2)   # ('t00'..'t11', y0, x0, tile)


def main():
    """Runs SAM3 text=wheat@0.25 on each panel and saves raw masks/boxes/scores per panel."""
    assert os.path.exists(IMG), f"missing input image {IMG}"
    os.makedirs(TMP, exist_ok=True)
    img_rgb = np.array(Image.open(IMG).convert("RGB"))
    H, W = img_rgb.shape[:2]
    print(f"image {IMG}  {W}x{H}")

    # FP16 to fit the local card, imgsz 1008 as in the probe
    predictor = build_fp16_predictor("src/mask_generation/weights/sam3.pt")

    for tag, y0, x0, panel_rgb in panels(img_rgb):
        ph, pw = panel_rgb.shape[:2]
        panel_bgr = panel_rgb[:, :, ::-1].copy()
        tile_path = os.path.join(TMP, f"{tag}.png")
        cv2.imwrite(tile_path, panel_bgr)              # set_image reads a path (matches the probe)
        predictor.set_image(tile_path)                 # encode this panel once
        predictor.args.conf = float(CONF)
        res = predictor(text=[PHRASE])
        masks, boxes, conf = result_masks(res)         # masks (N,ph,pw) uint8, boxes (N,4), conf (N)
        out = os.path.join(OUT, f"{tag}.npz")
        np.savez_compressed(out, masks=masks.astype(np.uint8), boxes=boxes.astype(np.float32),
                            conf=conf.astype(np.float32), y0=y0, x0=x0, ph=ph, pw=pw,
                            phrase=PHRASE, panel=tag)
        print(f"  [{tag:5s}] {pw}x{ph} -> {len(masks):4d} instances | saved {out} "
              f"(masks {masks.shape}, boxes {boxes.shape})")

    print(f"\nDONE. Raw per-instance outputs in {OUT}/  (full.npz + t00/t01/t10/t11.npz)")
    print("Next: render the figure from these .npz with sam3_text_rerun_figure.py (no GPU).")


if __name__ == "__main__":
    main()
