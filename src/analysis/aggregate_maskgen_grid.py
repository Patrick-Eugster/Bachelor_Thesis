"""Aggregate the phone mask-gen grid eval JSONs into one ranked comparison table.

Reads every results/mask_generation/phone/evaluation/<method>/masks_instance/<exp>/eval_masks_instance.json,
pools the per-image records into P/R/F1 + mean matched IoU + boundary IoU/F + merge/split counts, prints the
cells sorted by instance F1, and writes grid_summary.csv. Run after scripts/run_maskgen_grid.sh.

No confidence / AP here — the mask eval is confidence-free on purpose (SAM has no per-mask score; ranking by
the detector's box score would measure the detector, not the mask). See docs/mask_generation/MASK_EVAL_METRICS.md.
"""
import os
import glob
import json
import csv

import numpy as np

ROOT = "results/mask_generation/phone/evaluation"


def pool(json_path):
    """Pool one eval JSON's per-image records into a single row of grid metrics (or None if empty)."""
    d = json.load(open(json_path))
    imgs = d.get("images", [])
    if not imgs:
        return None
    tp = fp = fn = mg = sp = ngt = npred = 0
    mi_pairs, bi, bf = [], [], []
    for im in imgs:
        a, ms = im["at_threshold"], im["merge_split"]
        tp += a["tp"]; fp += a["fp"]; fn += a["fn"]
        ngt += im["n_gt"]; npred += im["n_pred"]
        mi_pairs.append((a["mean_iou_matched"], a["tp"]))          # weight mean IoU by #matches
        mg += ms.get("merge_preds", 0); sp += ms.get("split_gts", 0)
        b = im.get("boundary")
        if b and not np.isnan(b["boundary_iou"]):
            bi.append(b["boundary_iou"]); bf.append(b["boundary_f"])
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    wn = sum(n for _, n in mi_pairs)
    miou = sum(m * n for m, n in mi_pairs) / wn if wn else 0.0
    return {
        "method": d["method"],
        "exp": os.path.basename(os.path.dirname(json_path)),
        "n_img": len(imgs), "GT": ngt, "pred": npred,
        "F1": F1, "recall": R, "precision": P, "mean_iou": miou,
        "boundary_iou": float(np.mean(bi)) if bi else float("nan"),
        "boundary_f": float(np.mean(bf)) if bf else float("nan"),
        "merges": mg, "splits": sp,
    }


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "*", "masks_instance", "*", "eval_masks_instance.json"))):
        r = pool(f)
        if r:
            rows.append(r)
    if not rows:
        print("No eval JSONs found under", ROOT)
        return
    rows.sort(key=lambda r: -r["F1"])

    hdr = (f"{'method':14s} {'exp':14s} {'img':>3s} {'F1':>6s} {'recall':>6s} {'prec':>6s} "
           f"{'mIoU':>6s} {'bIoU':>6s} {'bF':>6s} {'merg':>5s} {'splt':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['method']:14s} {r['exp']:14s} {r['n_img']:>3d} {r['F1']:>6.3f} {r['recall']:>6.3f} "
              f"{r['precision']:>6.3f} {r['mean_iou']:>6.3f} {r['boundary_iou']:>6.3f} {r['boundary_f']:>6.3f} "
              f"{r['merges']:>5d} {r['splits']:>5d}")

    out = os.path.join(ROOT, "grid_summary.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV -> {out}   ({len(rows)} cells)")


if __name__ == "__main__":
    main()
