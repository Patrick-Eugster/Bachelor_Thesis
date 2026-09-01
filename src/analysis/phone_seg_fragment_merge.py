#!/usr/bin/env python
"""phone_seg_fragment_merge.py — GO/NO-GO gate for the "fragmentation" hypothesis on phone 3D seg.

The diagnosed failure (see docs/analysis_results/PHONE_SEG_A0715_EVAL_TABLE.md) is that the seg cuts one
real wheat head into SEVERAL 3D ids (nPred ~1.7-2x nGT), so almost no single piece clears the IoU-0.5 bar
against a whole GT head -> instance F1 stuck ~0.05. This script tests, WITHOUT any Euler run, whether two
cheap 2D post-processes on the already-pulled 2DSeg label map recover instance F1:

  1. SIZE FILTER          — drop predicted ids below a pixel-area threshold (kills tiny sliver false positives).
  2. FRAGMENT REABSORB    — CONSERVATIVE merge: absorb a SMALL piece into an ADJACENT MUCH-LARGER piece only.
                            Gated by a max fragment area AND a size ratio, so two comparably-sized touching
                            heads are NEVER merged (phone heads touch a lot — a blind touch-merge would wrongly
                            fuse real neighbours). This reabsorbs slivers back into their parent head.
  3. TOUCH MERGE (ceiling)— blind union-find over every touching pair. OVER-MERGES real neighbours by design;
                            OFF by default, printed only to bracket the theoretical upper bound. NOT a real config.

It is a 2D approximation of the real fix (which would merge 3D ids by multi-view/Gaussian-cluster adjacency and
needs all_obj_labels.pth, not pulled). But as a CEILING GATE it is valid: if collapsing touching fragments on
the final projected map does NOT lift instance F1, the 3D version won't either -> don't spend an Euler slot on it.

SAFETY: READ-ONLY on results/ and input_plots/. All transforms are in-memory on a COPY of the label map; the
on-disk 2DSeg .pt files are never written. Writes ONLY the NEW file
docs/analysis_results/phone_seg_fragment_merge_eval.json, and only when --write is passed. Default = dry-run
(compute + print, write nothing). It NEVER touches phone_seg_cpu_eval.json / phone_seg_instance_eval.json.

    python src/analysis/phone_seg_fragment_merge.py            # dry-run, both winner cells (A + D), writes nothing
    python src/analysis/phone_seg_fragment_merge.py --ceiling  # also print the aggressive touch-merge upper bound
    python src/analysis/phone_seg_fragment_merge.py --write     # also write the json
"""
import os
import sys
import json
import argparse
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# reuse the EXACT scoring + loading machinery so numbers are directly comparable to phone_seg_instance_eval.
from phone_seg_cpu_eval import RUNS, S  # noqa: E402
from phone_seg_instance_eval import (  # noqa: E402
    load_gt_instances, compute_iou_matrix, metrics_at, count_merges_splits,
    preds_from_labelmap, nn_resize_labels, VARIANT_SUBDIR,
)

OUT = "docs/analysis_results/phone_seg_fragment_merge_eval.json"

# which runs to test: the current winners (conf070 per_head SAM2 + IoU 0.6) on both fields. Matched to the
# baseline numbers in the eval table (A inst F1 0.051 / pix IoU 0.417; D inst F1 0.081).
TARGET_NAMES = ["ocv15k_perhead_sam2_conf070_iou06", "D0627_ocv15k_perhead_sam2_conf070_iou06"]

# sweeps (kept small — this is a gate, not a grid search)
SIZE_THRESHOLDS = [0, 50, 100, 200, 400]        # min pixel area to KEEP (0 = no filter = the current number)
REABSORB_PARAMS = [(200, 4.0), (400, 4.0), (400, 8.0), (800, 8.0)]  # (max fragment area, min neighbour/frag ratio)


def load_lab_and_gt(name, mp, exp, stem, sfm, sess):
    """Load one run's predicted 2DSeg label map + its GT instance map (read-only), nearest-resizing the
    prediction to the GT shape if needed. Returns (lab int32, gt_map, gt_ids, gt_areas) or a status string."""
    base = f"results/reconstruction/phone/{sess}"
    pt = f"{base}/{mp}/segmentation_3d/{exp}/2DSeg/{stem}.pt"
    if not os.path.exists(pt):
        return "missing_pred"
    sub = VARIANT_SUBDIR.get(sfm, "")
    label_dir = os.path.join("input_plots", "phone", sess, sub, "manual_label") if sub \
        else os.path.join("input_plots", "phone", sess, "manual_label")
    gt_map, gt_ids, gt_areas = load_gt_instances(label_dir, stem)
    if gt_map is None:
        return "no_instance_gt"
    lab = torch.load(pt, weights_only=True)
    lab = lab.numpy() if hasattr(lab, "numpy") else np.array(lab)
    lab = lab.astype(np.int32)
    if lab.shape != gt_map.shape:
        lab = nn_resize_labels(lab, gt_map.shape)
    return lab, gt_map, gt_ids, gt_areas


def score_labelmap(lab, gt_map, gt_ids, gt_areas):
    """Instance metrics (IoU>=0.5, plus F1@.25) for one label map against the GT — same definitions as
    phone_seg_instance_eval so the before/after is directly comparable."""
    preds = preds_from_labelmap(lab)
    n_pred, n_gt = len(preds), len(gt_ids)
    iou, inter = compute_iou_matrix(gt_map, gt_ids, gt_areas, preds)
    m50, _ = metrics_at(iou, 0.5, n_pred, n_gt)
    m25, _ = metrics_at(iou, 0.25, n_pred, n_gt)
    ms = count_merges_splits(inter, preds, gt_areas, gt_ids, 0.5, 0.5)
    return {"n_pred": n_pred, "precision": round(m50["precision"], 4), "recall": round(m50["recall"], 4),
            "f1": round(m50["f1"], 4), "iou_matched": round(m50["mean_iou_matched"], 4),
            "f1_iou25": round(m25["f1"], 4), "split_gts": ms["split_gts"]}


# ---------------------------------------------------------------------------------------------------------
# the three in-memory transforms (all operate on a COPY; on-disk data is never touched)
# ---------------------------------------------------------------------------------------------------------
def size_filter(lab, min_area):
    """Drop every predicted id whose pixel area is below min_area (set those pixels to background)."""
    if min_area <= 0:
        return lab
    out = lab.copy()
    ids, counts = np.unique(out, return_counts=True)
    small = ids[(ids != 0) & (counts < min_area)]
    if small.size:
        out[np.isin(out, small)] = 0
    return out


def border_pairs(lab):
    """Length of the shared border between every pair of adjacent nonzero ids (4-connectivity).
    Returns {(i,j): border_pixels} with i<j. Vectorized over the 2 axis-shifts."""
    pc = Counter()
    for a, b in ((lab[1:, :], lab[:-1, :]), (lab[:, 1:], lab[:, :-1])):
        m = (a != b) & (a != 0) & (b != 0)
        if not m.any():
            continue
        aa, bb = a[m], b[m]
        lo = np.minimum(aa, bb)
        hi = np.maximum(aa, bb)
        for i, j in zip(lo.tolist(), hi.tolist()):
            pc[(i, j)] += 1
    return pc


def reabsorb_fragments(lab, max_frag_area, ratio):
    """CONSERVATIVE merge: a piece with area < max_frag_area is absorbed into the adjacent id it shares the
    longest border with, but ONLY if that neighbour is >= ratio x its area (i.e. it's a sliver of a bigger
    head, not a real neighbouring head). Two comparably-sized touching heads are left untouched."""
    out = lab.copy()
    areas = np.bincount(out.ravel())
    pc = border_pairs(out)
    # neighbours[i] = list of (neighbour_id, shared_border_len)
    neighbours = defaultdict(list)
    for (i, j), blen in pc.items():
        neighbours[i].append((j, blen))
        neighbours[j].append((i, blen))
    # fragments = small ids, absorbed smallest-first; use ORIGINAL areas for the decision (one pass, good
    # enough for a ceiling gate — we are not chaining absorptions).
    frag_ids = [i for i in range(1, len(areas)) if 0 < areas[i] < max_frag_area]
    frag_ids.sort(key=lambda i: areas[i])
    remap = {}
    for i in frag_ids:
        cands = [(blen, j) for (j, blen) in neighbours.get(i, []) if areas[j] >= ratio * areas[i]]
        if not cands:
            continue
        _, j = max(cands)            # neighbour with the longest shared border among the much-larger ones
        remap[i] = j
    if remap:
        # resolve any short chains (i->j where j itself was absorbed) to a final target
        def final(x):
            seen = set()
            while x in remap and x not in seen:
                seen.add(x)
                x = remap[x]
            return x
        lut = np.arange(len(areas))
        for i in remap:
            lut[i] = final(i)
        out = lut[out]
    return out


def touch_merge_ceiling(lab, min_border):
    """AGGRESSIVE upper-bound ONLY: union-find over every pair of ids sharing >= min_border pixels, relabel
    each component to one id. This OVER-MERGES real touching heads by design — reported to bracket the ceiling,
    never used as a real config."""
    out = lab.copy()
    pc = border_pairs(out)
    ids = np.unique(out)
    parent = {int(i): int(i) for i in ids if i != 0}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), blen in pc.items():
        if blen >= min_border and i in parent and j in parent:
            parent[find(i)] = find(j)
    maxid = int(out.max())
    lut = np.arange(maxid + 1)
    for i in list(parent):
        if i <= maxid:
            lut[i] = find(i)
    return lut[out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ceiling", action="store_true",
                    help="also print the aggressive touch-merge upper bound (over-merges real neighbours)")
    ap.add_argument("--write", action="store_true",
                    help="write the json (default: dry-run, print only, write nothing)")
    args = ap.parse_args()

    targets = [r for r in RUNS if r[0] in TARGET_NAMES]
    out_rows = []
    for run in targets:
        name, mp, exp, gtp, stem, tag, sfm, iters = run[:8]
        sess = run[8] if len(run) > 8 else S
        loaded = load_lab_and_gt(name, mp, exp, stem, sfm, sess)
        if isinstance(loaded, str):
            print(f"{name}: {loaded} — skipped")
            continue
        lab0, gt_map, gt_ids, gt_areas = loaded

        print(f"\n=== {name}  ({sess}, nGT {len(gt_ids)}) ===")
        print(f"{'transform':<28}{'nPred':>7}{'P':>8}{'R':>8}{'F1':>8}{'F1@25':>8}{'IoUm':>8}{'spl':>6}")
        base = score_labelmap(lab0, gt_map, gt_ids, gt_areas)
        print(f"{'baseline (no post-proc)':<28}{base['n_pred']:>7}{base['precision']:>8.3f}"
              f"{base['recall']:>8.3f}{base['f1']:>8.3f}{base['f1_iou25']:>8.3f}{base['iou_matched']:>8.3f}"
              f"{base['split_gts']:>6}")
        rec = {"name": name, "sess": sess, "n_gt": int(len(gt_ids)), "baseline": base, "size_filter": [],
               "reabsorb": [], "ceiling": None}

        # 1. size filter sweep
        for mn in SIZE_THRESHOLDS:
            if mn == 0:
                continue
            s = score_labelmap(size_filter(lab0, mn), gt_map, gt_ids, gt_areas)
            s["min_area"] = mn
            rec["size_filter"].append(s)
            print(f"{'size>=' + str(mn):<28}{s['n_pred']:>7}{s['precision']:>8.3f}{s['recall']:>8.3f}"
                  f"{s['f1']:>8.3f}{s['f1_iou25']:>8.3f}{s['iou_matched']:>8.3f}{s['split_gts']:>6}")

        # 2. conservative fragment reabsorption sweep
        for mfa, ratio in REABSORB_PARAMS:
            s = score_labelmap(reabsorb_fragments(lab0, mfa, ratio), gt_map, gt_ids, gt_areas)
            s["max_frag_area"], s["ratio"] = mfa, ratio
            rec["reabsorb"].append(s)
            print(f"{f'reabsorb<{mfa},x{ratio:g}':<28}{s['n_pred']:>7}{s['precision']:>8.3f}{s['recall']:>8.3f}"
                  f"{s['f1']:>8.3f}{s['f1_iou25']:>8.3f}{s['iou_matched']:>8.3f}{s['split_gts']:>6}")

        # 3. aggressive touch-merge ceiling (optional)
        if args.ceiling:
            s = score_labelmap(touch_merge_ceiling(lab0, 5), gt_map, gt_ids, gt_areas)
            rec["ceiling"] = s
            print(f"{'[CEILING] touch-merge':<28}{s['n_pred']:>7}{s['precision']:>8.3f}{s['recall']:>8.3f}"
                  f"{s['f1']:>8.3f}{s['f1_iou25']:>8.3f}{s['iou_matched']:>8.3f}{s['split_gts']:>6}"
                  f"   <- OVER-MERGES real heads, upper bound only")
        out_rows.append(rec)

    if not args.write:
        print(f"\n[DRY RUN] nothing written. Re-run with --write to save {OUT}. "
              f"(results/ + 2DSeg .pt + existing eval jsons are never touched either way.)")
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"note": "2D post-process ceiling gate for the fragmentation hypothesis; read-only on results/",
               "runs": out_rows}, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}   (NEW file; results/, 2DSeg .pt, and the other eval jsons untouched)")


if __name__ == "__main__":
    main()
