#!/usr/bin/env python3
"""phone_psnr_texture_probe.py — proof that low phone 3DGS PSNR is a HIGH-FREQUENCY TEXTURE artifact,
not a reconstruction/geometry/SfM failure.

For a trained experiment it reads the saved render/GT pairs (renders/NNNNN.png vs gt/NNNNN.png under
train/ and test/ ours_<iter>/) and, per split, reports:
  - PSNR (should match results.json for the test split — sanity that we measure the same thing)
  - PSNR after a 2px Gaussian blur on BOTH images: if this JUMPS, the per-pixel error lives in the
    finest detail (wheat awns) → PSNR is punishing high-freq texture the model got structurally right.
  - global render->GT shift via FFT phase-correlation: ~0 => no geometric misalignment (no pixel-shift
    bug), so the low PSNR is NOT a pose/principal-point problem.

Run (any experiment, FIP or phone):
    python src/analysis/phone_psnr_texture_probe.py \
      -m results/reconstruction/phone/field_A/20250715/vanilla_3dgs/colmap_bench
    python src/analysis/phone_psnr_texture_probe.py -m <model_path> --iteration 15000 --json out.json

Writes the raw per-split numbers to --json if given (docs/analysis_results/ by convention).
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def psnr(a, b):
    """Standard PSNR between two 0-255 float arrays (same shape)."""
    mse = ((a - b) ** 2).mean()
    return float(10 * np.log10(255.0 ** 2 / mse)) if mse > 0 else float("inf")


def fft_shift(r, g, ds=4):
    """Global render->GT translation (dx,dy) in pixels via FFT phase-correlation on a ds-downsampled
    grayscale (fast, and integer-pixel shift is all we need). ~0 means the render is aligned to the GT."""
    rg = r[::ds, ::ds].mean(2)
    gg = g[::ds, ::ds].mean(2)
    R = np.fft.fft2(rg)
    G = np.fft.fft2(gg)
    X = R * np.conj(G)
    X /= np.abs(X) + 1e-8
    c = np.fft.ifft2(X).real
    pk = np.unravel_index(np.argmax(c), c.shape)
    dy = pk[0] - (c.shape[0] if pk[0] > c.shape[0] // 2 else 0)
    dx = pk[1] - (c.shape[1] if pk[1] > c.shape[1] // 2 else 0)
    return int(dx * ds), int(dy * ds)


def analyze_split(split_dir, blur_sigma=2.0):
    """Loop over every renders/NNNNN.png + its gt/NNNNN.png in one split dir. Returns a dict of the
    aggregate PSNR / blurred-PSNR / shift stats (or None if the split has no renders)."""
    renders = sorted(glob.glob(os.path.join(split_dir, "renders", "*.png")))
    if not renders:
        return None
    P, PB, SH = [], [], []
    for rp in renders:
        gp = rp.replace(os.sep + "renders" + os.sep, os.sep + "gt" + os.sep)
        if not os.path.exists(gp):
            continue
        r = np.asarray(Image.open(rp).convert("RGB"), np.float64)
        g = np.asarray(Image.open(gp).convert("RGB"), np.float64)
        P.append(psnr(r, g))
        # blur BOTH the same way so we only remove the shared finest detail, not add information
        PB.append(psnr(gaussian_filter(r, (blur_sigma, blur_sigma, 0)),
                       gaussian_filter(g, (blur_sigma, blur_sigma, 0))))
        SH.append(fft_shift(r, g))
    P, PB, SH = np.array(P), np.array(PB), np.abs(np.array(SH))
    return {
        "n": len(P),
        "psnr_mean": round(float(P.mean()), 3), "psnr_min": round(float(P.min()), 3),
        "psnr_max": round(float(P.max()), 3),
        "psnr_blur2px_mean": round(float(PB.mean()), 3),
        "blur_gain_db": round(float(PB.mean() - P.mean()), 3),
        "max_shift_px": [int(SH[:, 0].max()), int(SH[:, 1].max())],
        "n_views_with_shift": int((SH.sum(1) > 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_path", required=True,
                    help="experiment dir, e.g. results/reconstruction/phone/field_A/20250715/vanilla_3dgs/colmap_bench")
    ap.add_argument("--iteration", type=int, default=15000)
    ap.add_argument("--blur_sigma", type=float, default=2.0)
    ap.add_argument("--splits", nargs="+", default=["test"], choices=["test", "train"],
                    help="which split(s) to probe (default: test = the reported-metric split; "
                         "train is slower — ~84 phone views)")
    ap.add_argument("--json", default=None, help="optional path to save the raw numbers")
    args = ap.parse_args()

    out = {"model_path": args.model_path, "iteration": args.iteration, "blur_sigma": args.blur_sigma, "splits": {}}
    print(f"\n{args.model_path}  (iter {args.iteration})")
    print(f"{'split':6} {'n':>3} {'PSNR':>7} {'[min,max]':>15} {'+2px-blur':>10} {'Δblur':>7} {'maxShift':>10} {'shifted':>8}")
    for split in args.splits:
        d = os.path.join(args.model_path, split, f"ours_{args.iteration}")
        s = analyze_split(d, args.blur_sigma)
        out["splits"][split] = s
        if s is None:
            print(f"{split:6}  (no renders — run render.py first)")
            continue
        print(f"{split:6} {s['n']:>3} {s['psnr_mean']:>7.2f} "
              f"{f'[{s['psnr_min']:.1f},{s['psnr_max']:.1f}]':>15} "
              f"{s['psnr_blur2px_mean']:>10.2f} {'+'+format(s['blur_gain_db'],'.2f'):>7} "
              f"{str(tuple(s['max_shift_px'])):>10} {f'{s['n_views_with_shift']}/{s['n']}':>8}")

    print("\nRead: test PSNR should match results.json. A large Δblur (+dB) means the error is in the finest")
    print("detail (wheat awns) → PSNR is texture-limited, not a geometry failure. maxShift≈0 => aligned.\n")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"→ wrote {args.json}")


if __name__ == "__main__":
    main()
