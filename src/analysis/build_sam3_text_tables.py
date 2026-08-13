"""Regenerates the SAM3 text-prompt appendix tables (phone + FIP) and the key numbers
quoted in the Results, straight from the probe's summary JSONs. Prints LaTeX table bodies
and a short numbers block to stdout; paste the bodies into main.tex.

Data model: for each (dataset, phrase, conf) the probe stored an instance count per image,
in full-frame mode (summary_full.json) and per tile in a 2x2 grid (summary_tiles.json).
We aggregate to a mean-per-image count: full = the stored count; tiled = the 4 tiles summed
per image (NOT de-duplicated, so tiled overcounts heads in tile overlaps), then averaged
over the GT images. Dataset is split (phone/FIP), images are aggregated, mode is a full/tiled
pair, and phrase x conf are the two shown axes.

Run: python src/analysis/build_sam3_text_tables.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

FULL = "results/sam3_text_probe/summary_full.json"
TILES = "results/sam3_text_probe/summary_tiles.json"
CONFS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
# the 13 phrases we swept, in a fixed reading order (specific -> generic -> single word)
PHRASES = ["wheat", "wheat inflorescence", "seed head", "wheat spike", "wheat ear",
           "ear of wheat", "cereal spike", "grain head", "head of wheat",
           "inflorescence", "spike", "ear", "wheat head"]
FIRE_EPS = 0.05   # a phrase "fires" if its mean-per-image count reaches >=0.05 anywhere


def load():
    """Loads the two summary files."""
    return json.load(open(FULL)), json.load(open(TILES))


def full_mean(full, ds, phrase, conf):
    """Mean full-frame instance count over the GT images of one dataset."""
    v = [r["num_instances"] for r in full
         if r["dataset"] == ds and r["phrase"] == phrase and r["conf_thr"] == conf]
    return sum(v) / len(v) if v else 0.0


def tiled_mean(tiles, ds, phrase, conf):
    """Mean tiled count: sum the 4 tiles per image (no merge), then average over images."""
    per = defaultdict(int)
    for r in tiles:
        if r["dataset"] == ds and r["phrase"] == phrase and r["conf_thr"] == conf:
            per[r["stem"]] += r["num_instances"]
    return sum(per.values()) / len(per) if per else 0.0


def fires(full, tiles, ds, phrase):
    """True if the phrase produces any instances at any confidence in either mode."""
    mf = max(full_mean(full, ds, phrase, c) for c in CONFS)
    mt = max(tiled_mean(tiles, ds, phrase, c) for c in CONFS)
    return max(mf, mt) >= FIRE_EPS


def fmt(x):
    """One-decimal cell, blank-ish zeros shown as 0.0 for a clean grid."""
    return f"{x:.1f}"


def table_for(full, tiles, ds):
    """Prints the LaTeX tabular body for one dataset: firing phrases only, each with a
    full and a tiled row across the 9 confidences. Returns the list of dead phrases."""
    firing = [p for p in PHRASES if fires(full, tiles, ds, p)]
    # order firing phrases by their peak tiled count (strongest first)
    firing.sort(key=lambda p: max(tiled_mean(tiles, ds, p, c) for c in CONFS), reverse=True)
    dead = [p for p in PHRASES if p not in firing]

    print(f"% ---- {ds} text-prompt table body ({len(firing)} firing phrases) ----")
    for p in firing:
        fr = [fmt(full_mean(full, ds, p, c)) for c in CONFS]
        tr = [fmt(tiled_mean(tiles, ds, p, c)) for c in CONFS]
        print(f"{p:20s} & full  & " + " & ".join(fr) + r" \\")
        print(f"{'':20s} & tiled & " + " & ".join(tr) + r" \\")
        print(r"\hline")
    print(f"% dead ({len(dead)}): " + ", ".join(dead))
    print()
    return dead


def compact_results_table(full, tiles):
    """Prints the compact Results-chapter table body: the four firing phrases, full and tiled,
    at the two confidences the Results text quotes (0.10 and 0.25)."""
    rows = ["wheat", "wheat inflorescence", "seed head", "wheat spike"]
    print("% ---- compact phone Results table (full/tiled at conf 0.10 and 0.25) ----")
    for p in rows:
        cells = [fmt(full_mean(full, "phone", p, 0.10)), fmt(tiled_mean(tiles, "phone", p, 0.10)),
                 fmt(full_mean(full, "phone", p, 0.25)), fmt(tiled_mean(tiles, "phone", p, 0.25))]
        print(f"{p:20s} & " + " & ".join(cells) + r" \\")
    print()


def gt_counts():
    """Ground-truth head counts per GT image, the reference the probe under-segments against."""
    phone = []
    for sets in sorted(glob.glob("input_plots/phone/*/*/manual_label/*_sets/set0_instances.png")):
        a = np.array(Image.open(sets))
        phone.append(int(len(np.unique(a)) - 1))       # minus the background id 0
    fip = []
    for txt in sorted(glob.glob("input_plots/fip/*/manual_label/*.txt")):
        fip.append(sum(1 for _ in open(txt)))
    return phone, fip


def main():
    """Builds both tables + prints the reference head counts and a few Results numbers."""
    full, tiles = load()
    print("==== COMPACT RESULTS TABLE ====\n")
    compact_results_table(full, tiles)
    print("==== TABLE BODIES ====\n")
    phone_dead = table_for(full, tiles, "phone")
    fip_dead = table_for(full, tiles, "fip")

    print("==== REFERENCE / RESULTS NUMBERS ====")
    phone, fip = gt_counts()
    print(f"phone GT heads/image: {sorted(phone)}  mean {sum(phone)/len(phone):.0f}  "
          f"range {min(phone)}-{max(phone)}")
    print(f"fip   GT heads/image: mean {sum(fip)/len(fip):.0f}  range {min(fip)}-{max(fip)}")
    print(f"phone dead phrases ({len(phone_dead)}): {', '.join(phone_dead)}")
    print(f"fip   dead phrases ({len(fip_dead)}): {', '.join(fip_dead)}")
    # a few headline numbers for the prose
    print(f"phone wheat: full@0.25={full_mean(full,'phone','wheat',0.25):.1f} "
          f"tiled@0.25={tiled_mean(tiles,'phone','wheat',0.25):.1f} "
          f"tiled@0.10={tiled_mean(tiles,'phone','wheat',0.10):.1f}")
    print(f"fip   wheat: full@0.25={full_mean(full,'fip','wheat',0.25):.1f} "
          f"tiled@0.25={tiled_mean(tiles,'fip','wheat',0.25):.1f}")


if __name__ == "__main__":
    main()
