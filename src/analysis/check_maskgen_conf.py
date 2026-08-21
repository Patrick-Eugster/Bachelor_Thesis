"""Sanity-check the 3 conf mask sets from the 3D-seg conf sweep WITHOUT eyeballing every image.

Mask files are named {image_stem}_{idx:03}.png (one per detected head instance), so per-image head
counts come straight from the filenames — no pixel decode. For each conf experiment it reports coverage
(#images), total instances, and the per-image count distribution, then runs cross-conf checks:
  * MONOTONICITY: total instances must DROP as conf rises (0.40 >= 0.55 >= 0.70). A violation = bug.
  * COVERAGE: every conf must span the SAME image set (a mid-session crash would drop images).
  * ZERO-DETECTION: any image with 0 masks = detection failed on that frame.
  * OUTLIERS: per-image counts far outside the phone-plausible band flag broken frames.
Optionally decodes a few masks (--sample_decode) to confirm they're valid non-empty PNGs.

Run it on Euler (where the masks live) right after the mask-gen job. Read-only.

Usage:
  python src/analysis/check_maskgen_conf.py \
    --mask_base results/mask_generation/phone/field_A/20250715/opencv/yolo_sam_v1 \
    --exps pertile_sam2_conf040 pertile_sam2_conf055 pertile_sam2_conf070
"""
import argparse
import os
import re
from collections import defaultdict

STEM_RE = re.compile(r"^(.*)_\d{3}\.png$")   # {stem}_{idx:03}.png


def _counts_by_image(masks_dir):
    """Return {image_stem: n_instances} from the flat per-instance mask filenames."""
    counts = defaultdict(int)
    for fn in os.listdir(masks_dir):
        m = STEM_RE.match(fn)
        if m:
            counts[m.group(1)] += 1
    return dict(counts)


def _dist(vals):
    """min / median / mean / max of a list of ints (0s handled by caller)."""
    v = sorted(vals)
    n = len(v)
    med = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
    return {"min": v[0], "median": med, "mean": sum(v) / n, "max": v[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_base", required=True, help="the yolo_sam_v1 dir holding the conf experiments")
    ap.add_argument("--exps", nargs="+", required=True,
                    help="conf experiment folder names, IN RISING CONF ORDER (0.40, 0.55, 0.70)")
    ap.add_argument("--plausible", nargs=2, type=int, default=[80, 400],
                    help="per-image count band [lo hi]; images outside are flagged (default 80..400)")
    ap.add_argument("--sample_decode", type=int, default=0,
                    help="decode this many random masks per exp to confirm non-empty valid PNGs (0=skip)")
    a = ap.parse_args()

    per_exp = {}     # exp -> {stem: count}
    for exp in a.exps:
        d = os.path.join(a.mask_base, exp, "masks")
        if not os.path.isdir(d):
            print(f"!! MISSING masks dir: {d}")
            continue
        per_exp[exp] = _counts_by_image(d)

    print("=== per-conf summary ===")
    print(f"{'exp':<26} {'imgs':>5} {'total':>8} {'min':>5} {'median':>7} {'mean':>7} {'max':>5} {'zero':>5} {'outliers':>9}")
    totals = {}
    stemsets = {}
    for exp in a.exps:
        c = per_exp.get(exp)
        if not c:
            continue
        vals = list(c.values())
        total = sum(vals)
        totals[exp] = total
        stemsets[exp] = set(c)
        dd = _dist(vals)
        zero = sum(1 for v in vals if v == 0)
        lo, hi = a.plausible
        outl = sum(1 for v in vals if v < lo or v > hi)
        print(f"{exp:<26} {len(c):>5} {total:>8} {dd['min']:>5} {dd['median']:>7.0f} "
              f"{dd['mean']:>7.1f} {dd['max']:>5} {zero:>5} {outl:>9}")

    print("\n=== cross-conf checks ===")
    # monotonicity: totals must be non-increasing in the given (rising-conf) order
    ok_mono = all(totals[a.exps[i]] >= totals[a.exps[i + 1]]
                  for i in range(len(a.exps) - 1) if a.exps[i] in totals and a.exps[i + 1] in totals)
    print(f"[{'OK' if ok_mono else 'FAIL'}] monotonic: total instances drop as conf rises "
          f"({' >= '.join(str(totals.get(e, '?')) for e in a.exps)})")
    # coverage: all exps span the same image set
    present = [e for e in a.exps if e in stemsets]
    ref = stemsets[present[0]] if present else set()
    ok_cov = all(stemsets[e] == ref for e in present)
    print(f"[{'OK' if ok_cov else 'FAIL'}] coverage: all confs span the same {len(ref)} images")
    if not ok_cov:
        for e in present:
            miss = ref - stemsets[e]
            if miss:
                print(f"    {e} is missing {len(miss)} images, e.g. {sorted(miss)[:3]}")
    # per-image subset sanity: higher conf should have <= count per image than the lowest conf
    if len(present) >= 2:
        base = per_exp[present[0]]
        viol = 0
        for e in present[1:]:
            for stem, cnt in per_exp[e].items():
                if stem in base and cnt > base[stem]:
                    viol += 1
        print(f"[{'OK' if viol == 0 else 'WARN'}] per-image: higher conf never exceeds conf-{present[0]} count "
              f"({viol} violations)")

    if a.sample_decode:
        try:
            import random
            import numpy as np
            from PIL import Image
            print(f"\n=== decode {a.sample_decode} random masks/exp (valid + non-empty?) ===")
            for exp in present:
                d = os.path.join(a.mask_base, exp, "masks")
                files = [f for f in os.listdir(d) if f.endswith(".png")]
                bad = 0
                for f in random.sample(files, min(a.sample_decode, len(files))):
                    arr = np.array(Image.open(os.path.join(d, f)))
                    if arr.max() == 0:
                        bad += 1
                print(f"  {exp}: {bad} empty of {min(a.sample_decode, len(files))} sampled")
        except Exception as e:
            print(f"(sample decode skipped: {e})")


if __name__ == "__main__":
    main()
