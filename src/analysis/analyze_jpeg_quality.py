"""
Diagnostic: how aggressively were the phone JPEGs compressed?

JPEG is lossy — the 8x8 DCT quantization deletes high-frequency detail, and chroma
subsampling halves colour resolution. For tiny objects (wheat heads only a few pixels
wide) this smears/shifts edges, which hurts YOLO+SAM mask precision and therefore the
downstream 3D segmentation far more than it hurts 3DGS. This script reads the evidence
the JPEG stores in its own header (quantization tables, subsampling, EXIF) plus a measured
block-artifact ratio, so we know how bad the compression actually is. Read-only.

Run:
    python src/analysis/analyze_jpeg_quality.py                 # auto: all phone sessions' input/ JPEGs
    python src/analysis/analyze_jpeg_quality.py --glob "<pattern>"
    python src/analysis/analyze_jpeg_quality.py --sample 12     # images sampled per session (default 8)
Output:
    docs/analysis_results/jpeg_quality.json + printed table
"""

import os
import glob
import json
import argparse
import numpy as np
from PIL import Image
from PIL.JpegImagePlugin import get_sampling

REPO = "/workspace"
OUT_JSON = os.path.join(REPO, "docs", "analysis_results", "jpeg_quality.json")

# Standard IJG Annex-K luminance quantization table (quality 50 baseline). We invert the
# image's actual luma table against this to estimate the quality factor the encoder used.
STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68,109,103, 77,
    24, 35, 55, 64, 81,104,113, 92,
    49, 64, 78, 87,103,121,120,101,
    72, 92, 95, 98,112,100,103, 99,
], dtype=np.float64)

SUBSAMPLING_NAMES = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0", -1: "unknown"}


def estimate_quality(luma_table):
    """Estimate the JPEG quality factor (0-100) from the luma quantization table.

    Inverts the standard IJG scaling per coefficient and averages. Approximate (assumes
    IJG-derived tables, which most phone encoders / OpenCamera use), but good enough to tell
    'near-lossless ~95' from 'destructive ~75'. Higher = better / less loss."""
    q = np.array(luma_table, dtype=np.float64)
    # IJG: quant = (STD * scale + 50) / 100, where scale = 5000/Q (Q<50) or 200-2Q (Q>=50).
    # invert per coefficient -> scale_i, then map back to Q, average the estimates.
    scales = np.clip((q * 100.0 - 50.0) / STD_LUMA, 1e-6, None)
    qs = np.where(scales > 100.0, 5000.0 / scales, (200.0 - scales) / 2.0)
    return float(np.clip(np.mean(qs), 1, 100))


def block_artifact_ratio(gray):
    """Ratio of mean gradient ON the 8-pixel block grid vs OFF it.

    JPEG blocking puts extra discontinuities exactly on the 8x8 boundaries, so a ratio > ~1
    means visible blocking is actually present in this image (>1.15 = strong). Measured on a
    center crop to keep it cheap."""
    h, w = gray.shape
    # center crop up to 1024x1024 aligned to the 8-grid so boundary indexing is exact
    ch, cw = min(1024, (h // 8) * 8), min(1024, (w // 8) * 8)
    y0 = ((h - ch) // 2 // 8) * 8
    x0 = ((w - cw) // 2 // 8) * 8
    g = gray[y0:y0 + ch, x0:x0 + cw].astype(np.float64)

    dx = np.abs(np.diff(g, axis=1))    # horizontal gradient -> vertical block edges
    dy = np.abs(np.diff(g, axis=0))    # vertical gradient   -> horizontal block edges
    # columns/rows at multiples of 8 (the block boundaries) vs all the others
    on_cols = dx[:, 7::8]
    off_cols = np.delete(dx, np.s_[7::8], axis=1)
    on_rows = dy[7::8, :]
    off_rows = np.delete(dy, np.s_[7::8], axis=0)
    on = (on_cols.mean() + on_rows.mean()) / 2.0
    off = (off_cols.mean() + off_rows.mean()) / 2.0
    return float(on / off) if off > 1e-9 else float("nan")


def analyze_one_image(path):
    """Read header (quant tables, subsampling, EXIF software) + measure block ratio for one JPEG."""
    img = Image.open(path)
    quant = img.quantization                       # {0: luma[64], 1: chroma[64], ...} - header only
    luma = quant.get(0)
    quality = estimate_quality(luma) if luma else float("nan")
    # high-frequency aggressiveness = mean of the bottom-right quadrant of the luma table
    luma_hf = float(np.array(luma).reshape(8, 8)[4:, 4:].mean()) if luma else float("nan")
    try:
        subs = SUBSAMPLING_NAMES.get(get_sampling(img), "unknown")
    except Exception:
        subs = "unknown"
    software = ""
    try:
        exif = img.getexif()
        software = str(exif.get(305, ""))          # 305 = Software tag
    except Exception:
        pass
    gray = np.asarray(img.convert("L"))
    block = block_artifact_ratio(gray)
    return {"quality": quality, "luma_hf": luma_hf, "subsampling": subs,
            "software": software, "block_ratio": block,
            "size": f"{img.width}x{img.height}"}


def discover_sessions():
    """Find phone sessions that have an input/ folder of original-capture JPEGs."""
    sessions = []
    for d in sorted(glob.glob(os.path.join(REPO, "input_plots", "phone", "*", "*", "input"))):
        parts = d.split(os.sep)
        sessions.append((d, f"phone/{parts[-3]}/{parts[-2]}"))
    return sessions


def main():
    parser = argparse.ArgumentParser(description="Measure JPEG compression aggressiveness of phone captures.")
    parser.add_argument("--glob", default=None, help="Custom glob of JPEGs (overrides auto phone-session discovery).")
    parser.add_argument("--sample", type=int, default=8, help="Images sampled per session (default 8).")
    args = parser.parse_args()

    if args.glob:
        groups = [("custom", sorted(glob.glob(args.glob)))]
    else:
        groups = []
        for folder, label in discover_sessions():
            files = sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                           glob.glob(os.path.join(folder, "*.jpeg")) +
                           glob.glob(os.path.join(folder, "*.JPG")))
            groups.append((label, files))

    results = []
    print("\nReading JPEG headers + measuring block artifacts...\n")
    for label, files in groups:
        if not files:
            print(f"  [skip] {label}: no JPEGs found")
            continue
        # sample evenly across the session
        idx = np.linspace(0, len(files) - 1, min(args.sample, len(files))).astype(int)
        per = [analyze_one_image(files[i]) for i in idx]
        agg = {
            "label": label,
            "n_total": len(files),
            "n_sampled": len(per),
            "size": per[0]["size"],
            "subsampling": max(set(p["subsampling"] for p in per), key=[p["subsampling"] for p in per].count),
            "software": max(set(p["software"] for p in per), key=[p["software"] for p in per].count),
            "mean_quality": float(np.nanmean([p["quality"] for p in per])),
            "mean_luma_hf": float(np.nanmean([p["luma_hf"] for p in per])),
            "mean_block_ratio": float(np.nanmean([p["block_ratio"] for p in per])),
        }
        results.append(agg)

    # ── table ──
    print("\n" + "=" * 104)
    print(f"{'session':<26}{'n':>5}{'size':>12}{'subsamp':>9}{'qual~':>7}{'lumaHF':>8}{'blockR':>8}  software")
    print("-" * 104)
    for m in results:
        print(f"{m['label']:<26}{m['n_total']:>5}{m['size']:>12}{m['subsampling']:>9}"
              f"{m['mean_quality']:>7.0f}{m['mean_luma_hf']:>8.1f}{m['mean_block_ratio']:>8.2f}  {m['software']}")
    print("=" * 104)
    print("qual~  : approx JPEG quality factor (higher=better; ~95 near-lossless, ~75 destructive)")
    print("lumaHF : mean high-freq luma quantizer (higher=more fine detail destroyed)")
    print("blockR : 8px block-boundary gradient / off-boundary (>1.15 = visible blocking present)")
    print("subsamp: 4:2:0 halves colour resolution; 4:4:4 keeps it\n")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote raw metrics -> {OUT_JSON}\n")


if __name__ == "__main__":
    main()
