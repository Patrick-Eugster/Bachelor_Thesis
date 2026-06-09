"""
Diagnostic: how big (in pixels) are the SAM wheat-head masks?

The smaller a head is on screen, the more JPEG compression hurts it: an 8x8 DCT block or the
halved 4:2:0 colour resolution smears a larger *fraction* of a tiny head's edge. So the head
size distribution tells us how exposed the masks are to compression. We report mask AREA
(foreground pixel count) and, more importantly for JPEG, the bounding-box MIN DIMENSION
(how many 8-px blocks span the head). Read-only.

NOTE: masks currently exist only for FIP (YOLO+SAM not yet run on phone). This measures FIP
head sizes; phone would need its own mask run.

Run:
    python src/analysis/analyze_mask_sizes.py                 # auto: all FIP .../initial/masks
    python src/analysis/analyze_mask_sizes.py --masks <dir>   # a specific masks/ folder
    python src/analysis/analyze_mask_sizes.py --sample 3000   # masks sampled per folder (default 1500; 0 = all)
Output:
    docs/analysis_results/mask_sizes.json + printed table
"""

import os
import glob
import json
import argparse
import numpy as np
from PIL import Image

REPO = "/workspace"
OUT_JSON = os.path.join(REPO, "docs", "analysis_results", "mask_sizes.json")


def measure_mask(path):
    """Return (area, width, length, perimeter, img_shape) for one binary mask PNG, or None if empty.
    area = foreground pixel count (the TRUE head silhouette, background excluded);
    width/length = shorter/longer side of the tight bbox; perimeter = boundary pixel count
    (foreground pixels touching background) — used to estimate how much edge JPEG can disturb."""
    m = np.asarray(Image.open(path).convert("L"))
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    # crop to the bbox first so the perimeter calc is cheap (mask PNGs are full 12 MP frames)
    crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0
    area = int(crop.sum())
    bh, bw = crop.shape
    width, length = (bw, bh) if bw <= bh else (bh, bw)
    # perimeter = foreground pixels that have at least one 4-neighbour in the background
    pad = np.pad(crop, 1)
    interior = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:] & crop
    perimeter = int(area - int(interior.sum()))
    return area, width, length, perimeter, m.shape


def pct(arr, p):
    """Percentile helper that tolerates empty input."""
    return float(np.percentile(arr, p)) if len(arr) else float("nan")


def analyze_folder(masks_dir, label, sample):
    """Measure the size distribution of all (or a sample of) masks in one folder."""
    files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    if not files:
        print(f"  [skip] {label}: no mask PNGs")
        return None
    if sample and sample < len(files):
        idx = np.linspace(0, len(files) - 1, sample).astype(int)
        files = [files[i] for i in idx]

    areas, widths, lengths, perims, img_shape = [], [], [], [], None
    for f in files:
        r = measure_mask(f)
        if r is None:
            continue
        area, width, length, perim, shape = r
        areas.append(area); widths.append(width); lengths.append(length); perims.append(perim)
        img_shape = shape
    areas = np.array(areas, dtype=float)
    widths = np.array(widths, dtype=float)
    lengths = np.array(lengths, dtype=float)
    perims = np.array(perims, dtype=float)
    n = len(areas)
    if n == 0:
        return None

    # JPEG edge-impact: a boundary uncertainty of t px disturbs ~ perimeter*t pixels of the mask,
    # i.e. a fraction (perimeter*t / area) of the head's area. Average that fraction over all heads.
    # t=1px ~ the 2x2 chroma colour-edge blur; t=2px ~ chroma + 8x8 DCT ring (conservative upper end).
    frac_per_px = perims / np.maximum(areas, 1)
    return {
        "label": label,
        "masks_dir": masks_dir,
        "image_size": f"{img_shape[1]}x{img_shape[0]}",
        "n_total": len(glob.glob(os.path.join(masks_dir, "*.png"))),
        "n_measured": int(n),
        "area_px_median": float(np.median(areas)),
        "area_px_mean": float(np.mean(areas)),
        "width_px_mean": float(np.mean(widths)),    # shorter bbox side
        "length_px_mean": float(np.mean(lengths)),  # longer bbox side
        "perimeter_px_mean": float(np.mean(perims)),
        "fill_ratio_mean": float(np.mean(areas / np.maximum(widths * lengths, 1))),
        # quantified JPEG impact (mean over heads), as a fraction of mask area:
        "jpeg_impact_1px_frac": float(np.mean(frac_per_px)),       # ~chroma 2x2 edge blur
        "jpeg_impact_2px_frac": float(np.mean(2.0 * frac_per_px)), # ~chroma + DCT ring (upper)
    }


def discover():
    """Auto-find all 'initial' mask folders (FIP is nested fip/plot_461, phone is phone/field_A/date)."""
    root = os.path.join(REPO, "results", "mask_generation")
    out = []
    # recursive: handles both FIP (2 levels) and phone (3 levels) under mask_generation/
    for d in sorted(glob.glob(os.path.join(root, "**", "yolo_sam_v1", "initial", "masks"), recursive=True)):
        rel = os.path.relpath(d, root).split(os.sep)   # e.g. [fip, plot_461, yolo_sam_v1, initial, masks]
        label = "/".join(rel[:-3])                     # drop yolo_sam_v1/initial/masks
        out.append((d, label))
    return out


def main():
    parser = argparse.ArgumentParser(description="Measure SAM mask pixel sizes (JPEG exposure).")
    parser.add_argument("--masks", default=None, help="A specific masks/ folder (overrides auto-discovery).")
    parser.add_argument("--sample", type=int, default=1500, help="Masks sampled per folder (default 1500; 0 = all).")
    args = parser.parse_args()

    targets = [(args.masks, args.masks)] if args.masks else discover()
    if not targets:
        print("No mask folders found.")
        return

    results = []
    print(f"\nMeasuring mask sizes ({len(targets)} folder(s), sample={args.sample or 'all'})...\n")
    for d, label in targets:
        m = analyze_folder(d, label, args.sample)
        if m:
            results.append(m)
            print(f"  done: {label}  ({m['n_measured']} masks)")

    # ── table ──
    print("\n" + "=" * 112)
    print(f"{'folder':<24}{'masks':>7}{'area mean':>10}{'W x L (px)':>14}{'perim':>8}{'fill':>7}"
          f"{'JPEG 1px':>10}{'JPEG 2px':>10}")
    print("-" * 112)
    for m in results:
        print(f"{m['label']:<24}{m['n_total']:>7}{m['area_px_mean']:>10.0f}"
              f"{m['width_px_mean']:>7.0f}x{m['length_px_mean']:<6.0f}{m['perimeter_px_mean']:>8.0f}"
              f"{m['fill_ratio_mean']:>7.2f}{m['jpeg_impact_1px_frac']*100:>9.1f}%{m['jpeg_impact_2px_frac']*100:>9.1f}%")
    print("=" * 112)
    print("area mean : mean foreground (silhouette) pixels per head")
    print("W x L     : mean shorter x longer bbox side (px)")
    print("perim     : mean boundary pixel count per head")
    print("fill      : silhouette area / bbox area (how much of the box is really head)")
    print("JPEG 1px  : mean % of mask AREA disturbed by a 1px edge uncertainty (~2x2 chroma blur)")
    print("JPEG 2px  : same for 2px (~chroma + 8x8 DCT ring, conservative upper bound)\n")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote raw metrics -> {OUT_JSON}\n")


if __name__ == "__main__":
    main()
