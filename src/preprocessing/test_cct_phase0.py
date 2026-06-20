"""Phase 0 smoke test for the vendored CCTDecode core (Option C marker detection).

Goal: answer ONE question before we build any front-end — "can CCTDecode's decode
core actually read one of our markers?" We test the core in isolation (we hand it
clean, marker-filling images, doing the front-end's job manually), so any failure is
the core's, not a region-finder's.

Three isolated checks:
  A1  DrawCCT synthetic marker (CCTDecode's OWN geometry) -> must decode.
        Proves the vendored code runs correctly in our env.
  A2  Marker #1 rendered from the Agisoft spec PDF (real Agisoft geometry, perfect
        image) -> tests whether Agisoft geometry decodes with stock params.
  B   A few real markers hand-cropped from a field photo (using v6's candidate
        locations only as crop seeds) -> tests the core on real lighting/tilt/blur.
        Crops are saved so a human can eyeball which are genuine markers.

Run:
  python src/preprocessing/test_cct_phase0.py
Nothing is modified in the dataset; all outputs go to a scratch folder under /tmp
and to OUT_DIR below.
"""

import os
# headless matplotlib (CCTDecodeRelease imports pyplot at module top)
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
import glob
import json
import shutil
import subprocess

import cv2
import numpy as np

# --- make the vendored package importable (its files use bare `from X import *`) ---
HERE = os.path.dirname(os.path.abspath(__file__))
CCT_DIR = os.path.join(HERE, "cctdecode")
sys.path.insert(0, CCT_DIR)

import CCTDecodeRelease as cct   # noqa: E402  (the decode core)
import DrawCCT as draw           # noqa: E402  (synthetic marker generator + B2I/I2B)

# ----------------------------- config -----------------------------
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PLOT_DIR = os.path.join(REPO, "input_plots", "phone", "field_A", "20250609")
IMAGES_DIR = os.path.join(PLOT_DIR, "images")
V6_JSON = os.path.join(PLOT_DIR, "logs", "marker_detections_v6.json")
SPEC_PDF = os.path.join(REPO, "reference", "agisoft",
                        "Coded_12bit_15cm-square_13cm-outer-circle_.pdf")

OUT_DIR = os.path.join(PLOT_DIR, "marker_vis_cct_phase0")
SCRATCH = "/tmp/cct_phase0"

N_BITS = 12
COLOR = "black"          # our physical markers are black marks on a white plate
CIRC_THRESH = 0.6        # CCT_extract's R (circularity); upstream default 0.85
STEP_B_IMAGE = "IMG_20250609_112223.jpg"   # richest v6 image
STEP_B_TOPK = 6          # decode the top-K v6 candidates
CROP_HALFWIDTH_FACTOR = 5.0  # crop half-width = factor * v6 fiducial radius


def banner(t):
    print("\n" + "=" * 64 + f"\n  {t}\n" + "=" * 64)


def decode_image(bgr, tag, save_path=None):
    """Run the CCTDecode core on one (already cropped/clean) image and report.
    Returns the CodeTable list [[code, cx, cy], ...]."""
    try:
        code_table, vis = cct.CCT_extract(bgr, N_BITS, CIRC_THRESH, COLOR)
    except Exception as e:
        print(f"  [{tag}] ERROR during decode: {type(e).__name__}: {e}")
        return []
    if save_path is not None:
        cv2.imwrite(save_path, vis)
    if code_table:
        for code, cx, cy in code_table:
            print(f"  [{tag}] DECODED id={code}  center=({cx:.1f},{cy:.1f})")
    else:
        print(f"  [{tag}] no CCT decoded")
    return code_table


# ------------------------- Step A1: synthetic -------------------------
def step_a1():
    banner("STEP A1 - DrawCCT synthetic marker (CCTDecode's own geometry)")
    os.makedirs(SCRATCH, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(SCRATCH)  # DrawCCT_black writes to ./CCT_IMG_12_Black/<value>.png
    try:
        pattern = [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
        expected = draw.B2I(pattern, N_BITS)  # canonical (rotation-min) id
        draw.DrawCCT_black(N_BITS, 600, pattern)
        gen = glob.glob(os.path.join(SCRATCH, "CCT_IMG_12_Black", "*.png"))
        if not gen:
            print("  could not generate synthetic marker")
            return
        path = gen[0]
        print(f"  generated {os.path.basename(path)} (expected canonical id={expected})")
        bgr = cv2.imread(path)
        # pad so the marker doesn't touch the border (helps contour/affine)
        bgr = cv2.copyMakeBorder(bgr, 100, 100, 100, 100,
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
        save = os.path.join(OUT_DIR, "A1_synthetic_decoded.png")
        table = decode_image(bgr, "A1", save)
        got = {int(c[0]) for c in table}
        print(f"  --> {'PASS' if expected in got else 'FAIL'} "
              f"(expected {expected}, got {sorted(got) if got else 'none'})")
    finally:
        os.chdir(cwd)


# --------------------- Step A2: Agisoft PDF marker --------------------
def step_a2():
    banner("STEP A2 - Agisoft spec-PDF marker #1 (real Agisoft geometry)")
    if not os.path.isfile(SPEC_PDF):
        print(f"  spec PDF not found: {SPEC_PDF}")
        return
    os.makedirs(SCRATCH, exist_ok=True)
    prefix = os.path.join(SCRATCH, "agisoft_marker")
    # render page 1 at 300 dpi to PNG
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-f", "1", "-l", "1",
                    SPEC_PDF, prefix], check=False)
    rendered = sorted(glob.glob(prefix + "*.png"))
    if not rendered:
        print("  pdftoppm produced no PNG")
        return
    bgr = cv2.imread(rendered[0])
    print(f"  rendered {os.path.basename(rendered[0])}  shape={bgr.shape}")
    save = os.path.join(OUT_DIR, "A2_agisoft_pdf_decoded.png")
    decode_image(bgr, "A2", save)
    print("  (if 'no CCT decoded' here but A1 passed -> Agisoft geometry differs "
          "from stock 2.5x ring; recalibrate before front-end work)")


# ----------------------- Step B: real cropped markers -----------------
def step_b():
    banner("STEP B - real markers hand-cropped from a field photo")
    img_path = os.path.join(IMAGES_DIR, STEP_B_IMAGE)
    if not os.path.isfile(img_path):
        print(f"  image not found: {img_path}")
        return
    if not os.path.isfile(V6_JSON):
        print(f"  v6 json not found: {V6_JSON}")
        return
    full = cv2.imread(img_path)
    H, W = full.shape[:2]
    per_image = json.load(open(V6_JSON))["per_image"]
    dets = per_image.get(STEP_B_IMAGE, [])
    if not isinstance(dets, list):
        dets = dets.get("markers", [])
    dets = sorted(dets, key=lambda d: -d.get("score", 0))[:STEP_B_TOPK]
    print(f"  {STEP_B_IMAGE}: feeding top {len(dets)} v6 candidates to the core")
    print("  (v6 is mostly wrong - real markers should decode, junk should be rejected)")

    for i, d in enumerate(dets):
        cx, cy = d["center"]
        r = max(float(d.get("fid_radius", 30.0)), 12.0)
        half = int(CROP_HALFWIDTH_FACTOR * r)
        x0, y0 = max(0, int(cx) - half), max(0, int(cy) - half)
        x1, y1 = min(W, int(cx) + half), min(H, int(cy) + half)
        crop = full[y0:y1, x0:x1]
        raw = os.path.join(OUT_DIR, f"B{i}_crop_raw_s{d.get('score',0):.2f}.png")
        cv2.imwrite(raw, crop)   # saved so a human can eyeball which are real
        save = os.path.join(OUT_DIR, f"B{i}_crop_decoded.png")
        print(f"  - candidate {i}: center=({cx:.0f},{cy:.0f}) r={r:.0f} score={d.get('score',0):.2f}")
        decode_image(crop, f"B{i}", save)


def main():
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Vendored core: {CCT_DIR}")
    print(f"Outputs (overlays + raw crops): {OUT_DIR}")
    step_a1()
    step_a2()
    step_b()
    print("\nDone. Inspect the overlays + raw crops in:\n  " + OUT_DIR)


if __name__ == "__main__":
    main()
