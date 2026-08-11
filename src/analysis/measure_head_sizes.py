"""Measures the real wheat-head size (in pixels) in the FIP and phone ground truth, and how
large each head is once SAM downscales the image to its fixed 1024-pixel encoder. This replaces
the back-of-envelope "~19 px" estimate in the thesis with measured medians, and gives the FIP-vs-phone
head-size gap that the Discussion leans on.

FIP head size comes from the GT boxes (side = sqrt(w*h)); phone head size comes from the GT instance
masks (side = sqrt(area), a tight measure, plus the instance bounding-box side for a box-comparable
number). Note the two datasets are measured on different bases --- FIP box vs phone mask --- so compare
the phone bbox-side row against FIP, not the phone sqrt(area) row.

Run: python src/analysis/measure_head_sizes.py
"""
import glob
import json
import os
import numpy as np
from PIL import Image

# SAM sees the image at a fixed 1024 px. These are the encode downscales per granularity.
SAM_ENCODE = 1024
PHONE_LONG = 4032          # phone frame long side
TILE_TARGET = 1344         # dynamic SAHI/per-tile target (docs: 4032 -> 1344 px, 12 tiles)
PERHEAD_MARGIN = 0.4       # per-head crop = bbox padded by margin_frac on each side


def pct(a):
    """Median and 10th/90th percentiles of an array, as a compact tuple."""
    a = np.asarray(a, dtype=float)
    return np.median(a), np.percentile(a, 10), np.percentile(a, 90)


def fip_measure():
    """Same as fip_head_sides but also tracks each image's long side for the encode fraction."""
    sides, longs, n_img = [], [], 0
    for txt in sorted(glob.glob("input_plots/fip/plot_*/manual_label/*.txt")):
        stem = os.path.splitext(os.path.basename(txt))[0]
        plot_dir = os.path.dirname(os.path.dirname(txt))
        img = os.path.join(plot_dir, "images", stem + ".png")
        if not os.path.exists(img):
            continue
        W, H = Image.open(img).size
        n_img += 1
        for line in open(txt):
            p = line.split()
            if len(p) < 5:
                continue
            _, _, _, bw, bh = (float(x) for x in p[:5])
            s = np.sqrt((bw * W) * (bh * H))
            sides.append(s)
            longs.append(max(W, H))
    return np.array(sides), np.array(longs), n_img


def phone_measure():
    """Phone head sqrt(area) and instance bbox side in pixels, pooled over the 6 GT instance maps."""
    area_sides, bbox_sides, n_img = [], [], 0
    for sets_dir in sorted(glob.glob("input_plots/phone/field_*/*/manual_label/*_sets")):
        manifest = os.path.join(sets_dir, "manifest.json")
        if not os.path.exists(manifest):
            continue
        m = json.load(open(manifest))
        active = next((s for s in m["sets"] if s["name"] == m["active"]), m["sets"][0])
        png = os.path.join(sets_dir, active["file"] + "_instances.png")
        inst = np.array(Image.open(png))            # uint16 instance map, 0 = background
        n_img += 1
        ids = np.unique(inst)
        ids = ids[ids != 0]
        counts = np.bincount(inst.ravel())
        for i in ids:
            area_sides.append(np.sqrt(counts[i]))
            ys, xs = np.where(inst == i)             # instance bbox
            bbox_sides.append(np.sqrt((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)))
    return np.array(area_sides), np.array(bbox_sides), n_img


def encode_px(side, long_side, tile=TILE_TARGET, margin=PERHEAD_MARGIN):
    """How many pixels a head of the given full-res side spans after SAM's 1024 encode, per granularity."""
    full = side * SAM_ENCODE / long_side
    per_tile = side * SAM_ENCODE / tile
    per_head = np.full_like(np.asarray(side, float), SAM_ENCODE / (1 + 2 * margin))  # crop->1024, constant
    return full, per_tile, per_head


def report(name, sides, long_side, frac_ref):
    """Prints the head-size medians and the per-granularity encode sizes for one measure."""
    med, p10, p90 = pct(sides)
    full, per_tile, per_head = encode_px(sides, long_side)
    print(f"  {name}: median {med:.0f} px  (p10 {p10:.0f}, p90 {p90:.0f}),  "
          f"{100*med/frac_ref:.1f}% of the {frac_ref:.0f}px frame")
    print(f"      at SAM 1024 encode -> full-frame {np.median(full):.0f} px | "
          f"per-tile {np.median(per_tile):.0f} px | per-head ~{np.median(per_head):.0f} px")


def main():
    print("=== FIP (from GT boxes, side = sqrt(w*h)) ===")
    fs, fl, fn = fip_measure()
    print(f"  {fn} labeled images, {len(fs)} heads")
    report("box side", fs, np.median(fl), np.median(fl))

    print("\n=== Phone (from GT instance masks) ===")
    pa, pb, pn = phone_measure()
    print(f"  {pn} labeled images, {len(pa)} heads")
    report("sqrt(area)", pa, PHONE_LONG, PHONE_LONG)
    report("bbox side ", pb, PHONE_LONG, PHONE_LONG)

    print("\n(compare FIP 'box side' against phone 'bbox side' for a like-for-like basis;")
    print(" sqrt(area) is the tighter true-extent measure used for the phone mask argument.)")


if __name__ == "__main__":
    main()
