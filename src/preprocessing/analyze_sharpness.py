"""Diagnostic sharpness analysis for a captured plot. READ-ONLY.

For each image in {source_path}/{image_subdir}/ this script computes the
Laplacian variance (Pech-Pacheco et al. 2000) — the standard quick sharpness
score: sharp images have lots of strong edges, so the variance of the Laplacian
output is high; blurry images smooth those edges out, so the variance is low.

This script does NOT modify any files. It only prints a report and writes a
JSON to {source_path}/logs/sharpness_report.json. Use the JSON to decide
whether a session is uniformly sharp or has problematic outliers.

Typical usage:
    python src/preprocessing/analyze_sharpness.py field=field_D plot=20250530

Config: configs/preprocessing/analyze_sharpness.yaml
"""

import json
import os
import time

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


def laplacian_variance(gray):
    """Standard sharpness score: variance of the Laplacian applied to a grayscale image.
    Sharp images → strong edges → high variance. Blurry images → smooth → low variance.
    Returns a float in arbitrary units (only meaningful when comparing within the same dataset)."""
    # cv2.CV_64F so negative gradients aren't clipped to 0 — they contribute to variance
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def classify(score):
    """Translate raw Laplacian variance into a coarse human label.
    Thresholds are empirical for 12 MP phone photos; treat as a rough guide, not absolute truth.
    The per-session DISTRIBUTION (which images are below 0.5x the session median) is more reliable."""
    if score > 500:   return "very sharp"
    if score > 200:   return "sharp"
    if score > 80:    return "slightly soft"
    if score > 30:    return "visibly blurry"
    return "heavy blur / out of focus"


def histogram_str(scores, n_bins=20, width=40):
    """ASCII histogram of scores — quick visual of the distribution shape.
    Width is the max bar width in characters."""
    if not scores:
        return "(no images)"
    s_min, s_max = min(scores), max(scores)
    if s_max == s_min:
        return f"  all scores = {s_min:.1f}"
    bins = np.linspace(s_min, s_max, n_bins + 1)
    counts, _ = np.histogram(scores, bins=bins)
    max_count = counts.max() if counts.max() > 0 else 1
    lines = []
    for i, c in enumerate(counts):
        lo, hi = bins[i], bins[i + 1]
        bar = "█" * int(round(width * c / max_count))
        lines.append(f"  {lo:7.1f} – {hi:7.1f}  | {bar} {c}")
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing/analyze_sharpness")
def main(cfg: DictConfig):
    """Score every image's sharpness, print a report (mean/median/distribution + worst/best),
    and write a JSON for later inspection. Touches no images."""
    print("--- analyze_sharpness config ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------------------------")
    t_start = time.time()

    image_dir = os.path.join(cfg.source_path, cfg.image_subdir)
    if not os.path.isdir(image_dir):
        print(f"ERROR: image dir not found: {image_dir}")
        return

    files = sorted(f for f in os.listdir(image_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")))
    if not files:
        print(f"ERROR: no images found in {image_dir}")
        return
    print(f"Analyzing {len(files)} images from {image_dir}...")

    per_image = []
    for i, f in enumerate(files):
        path = os.path.join(image_dir, f)
        # IMREAD_GRAYSCALE is faster + correct (Laplacian on color = same per-channel work)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [skip] could not decode {f}")
            continue
        if cfg.downscale < 1.0:
            h, w = img.shape
            img = cv2.resize(img, (int(w * cfg.downscale), int(h * cfg.downscale)),
                             interpolation=cv2.INTER_AREA)
        score = laplacian_variance(img)
        per_image.append({"name": f, "score": score, "label": classify(score)})
        # tiny progress indicator every 25 images
        if (i + 1) % 25 == 0 or (i + 1) == len(files):
            print(f"  [{i+1}/{len(files)}] {f:<35} → {score:7.1f}  ({classify(score)})")

    scores = np.array([p["score"] for p in per_image])
    sorted_by_score = sorted(per_image, key=lambda p: p["score"])
    median = float(np.median(scores))
    # how many fall below 50% of the session median — practical "obvious outliers" count
    n_outliers_below_half_median = int(np.sum(scores < 0.5 * median))

    # build the report
    elapsed = time.time() - t_start
    print("\n" + "="*60)
    print("      SHARPNESS REPORT")
    print("="*60)
    print(f"{'Plot:':<28} {cfg.field}/{cfg.plot}")
    print(f"{'Image folder:':<28} {cfg.image_subdir}/")
    print(f"{'Images analyzed:':<28} {len(per_image)}")
    print("-" * 60)
    print(f"{'min  / max  score:':<28} {scores.min():>8.1f} / {scores.max():.1f}")
    print(f"{'median score:':<28} {median:>8.1f}  ({classify(median)})")
    print(f"{'mean   score:':<28} {scores.mean():>8.1f}")
    print(f"{'std    score:':<28} {scores.std():>8.1f}")
    print(f"{'10th / 90th percentile:':<28} {np.percentile(scores, 10):>8.1f} / {np.percentile(scores, 90):.1f}")
    print(f"{'below 0.5x median:':<28} {n_outliers_below_half_median} images  (candidate outliers)")
    print("-" * 60)
    print("Distribution:")
    print(histogram_str(scores.tolist()))
    print("-" * 60)

    if cfg.worst_n > 0:
        print(f"Worst {cfg.worst_n} (blurriest):")
        for p in sorted_by_score[:cfg.worst_n]:
            print(f"  {p['score']:7.1f}  {p['label']:<25} {p['name']}")
    if cfg.best_n > 0:
        print(f"\nBest {cfg.best_n} (sharpest):")
        for p in sorted_by_score[-cfg.best_n:][::-1]:
            print(f"  {p['score']:7.1f}  {p['label']:<25} {p['name']}")

    print("-" * 60)
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"{'TOTAL TIME:':<28} {minutes}m {seconds}s  ({elapsed:.1f}s)")
    print("="*60 + "\n")

    # write the JSON — read by future automation (e.g. an opt-in filter step) and for diffing across sessions
    report = {
        "field": cfg.field,
        "plot": cfg.plot,
        "image_subdir": cfg.image_subdir,
        "n_images": len(per_image),
        "stats": {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "median": median,
            "std": float(scores.std()),
            "p10": float(np.percentile(scores, 10)),
            "p25": float(np.percentile(scores, 25)),
            "p75": float(np.percentile(scores, 75)),
            "p90": float(np.percentile(scores, 90)),
        },
        "n_below_half_median": n_outliers_below_half_median,
        "per_image": per_image,   # full list, already in input order
        "elapsed_s": elapsed,
    }
    out_path = os.path.join(cfg.source_path, cfg.output_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {out_path}\n")
    print("NOTE: this script is diagnostic only — no images were modified or removed.\n")


if __name__ == "__main__":
    main()
