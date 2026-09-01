"""score_sam_masks.py — direct 2D quality score for the SAM1-vs-SAM2-vs-SAM3 mask A/B.

We can't use eval_seg_2d here (it's wired to the 3D-segmentation output tree). Instead this reads the
raw per-head SAM masks a mask-gen experiment wrote (results/.../<method>/<exp>/masks/) and, for every
image that has a manual GT mask (input_plots/.../manual_label/<stem>_gt_mask.png, a binary 0/255
foreground), UNIONS all that image's instance masks into one binary foreground and compares it to GT.

Reported per experiment (one row per SAM backend): pixel IoU / precision / recall / F1 on the GT
images, plus the head count. Binary-foreground IoU under-measures instance quality (a merge of two
heads still looks fine in the union), so this is the SPEED+SANITY screen on clean FIP heads — the real
instance-quality battle is on phone (needs the CVAT GT we're building). Throughput (sec/image) comes
from the mask-gen run's own SAM summary in its log, not from here.

Usage:
    python src/analysis/score_sam_masks.py --field fip --plot plot_461 \
        --exp sambench_sam1 sambench_sam2 [sambench_sam3] [--method yolo_sam_v1]
"""

import os
import re
import glob
import json
import argparse

import numpy as np
import cv2


def _binary(mask, thr=127):
    """uint8 image -> bool foreground (> thr)."""
    return mask > thr


def _union_pred(masks_dir, stem, gt_shape):
    """One image's binary foreground union, resized to GT shape. Prefers a single <stem>_union.png
    (written by save_union_mask — the light A/B path); else ORs the per-head <stem>_NNN.png files.
    Returns (pred_bool, n_instances) — n is -1 when only the union PNG exists (per-head count unknown)."""
    H, W = gt_shape
    union_path = os.path.join(masks_dir, f"{stem}_union.png")
    if os.path.exists(union_path):
        m = cv2.imread(union_path, cv2.IMREAD_GRAYSCALE)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        return _binary(m), -1   # per-head count not available from a union PNG
    files = sorted(glob.glob(os.path.join(masks_dir, f"{stem}_*.png")))
    files = [f for f in files if re.match(rf"^{re.escape(stem)}_\d+\.png$", os.path.basename(f))]
    pred = np.zeros((H, W), dtype=bool)
    n = 0
    for f in files:
        m = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        pred |= _binary(m)
        n += 1
    return pred, n


def _metrics(gt, pred):
    """Binary-foreground IoU / precision / recall / F1 / pixel-accuracy for one image."""
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    tp = int(inter)
    fp = int(pred.sum() - inter)
    fn = int(gt.sum() - inter)
    iou = tp / union if union else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = float((gt == pred).mean())
    return {"IoU": iou, "precision": prec, "recall": rec, "F1": f1, "pixel_acc": acc}


def score_experiment(field, plot, method, exp, gt_dir):
    """Score one mask-gen experiment against every available GT mask. Returns a dict:
    {per_image: {stem: metrics+n}, mean: metrics_over_gt_images, n_gt_images}."""
    masks_dir = os.path.join("results", "mask_generation", field, plot, method, exp, "masks")
    if not os.path.isdir(masks_dir):
        raise SystemExit(f"masks dir not found: {masks_dir} (did the {exp} run finish with save_masks=true?)")

    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*_gt_mask.png")))
    if not gt_files:
        raise SystemExit(f"no *_gt_mask.png in {gt_dir}")

    per_image = {}
    for gt_path in gt_files:
        stem = os.path.basename(gt_path).replace("_gt_mask.png", "")
        gt = _binary(cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE))
        pred, n = _union_pred(masks_dir, stem, gt.shape)
        row = _metrics(gt, pred)
        row["n_heads"] = n
        per_image[stem] = row

    keys = ["IoU", "precision", "recall", "F1", "pixel_acc"]
    mean = {k: float(np.mean([per_image[s][k] for s in per_image])) for k in keys}
    # per-backend peak VRAM/RAM + speed, written by run_sam_phase (sam_perf.json next to masks/)
    perf = {}
    perf_path = os.path.join(os.path.dirname(masks_dir), "sam_perf.json")
    if os.path.exists(perf_path):
        perf = json.load(open(perf_path))
    return {"per_image": per_image, "mean": mean, "n_gt_images": len(gt_files), "perf": perf}


def main():
    ap = argparse.ArgumentParser(description="Score raw SAM masks vs manual GT (binary foreground IoU).")
    ap.add_argument("--field", default="fip")
    ap.add_argument("--plot", required=True, help="fip plot (e.g. plot_461) or phone session")
    ap.add_argument("--method", default="yolo_sam_v1", help="result subtree (matches the mask-gen method)")
    ap.add_argument("--exp", nargs="+", required=True, help="one or more experiment names to compare")
    ap.add_argument("--gt_dir", default=None, help="override GT dir (default input_plots/<field>/<plot>/manual_label)")
    ap.add_argument("--out", default=None, help="write JSON here (default results/analysis/sam_backend_ab/<plot>.json)")
    args = ap.parse_args()

    gt_dir = args.gt_dir or os.path.join("input_plots", args.field, args.plot, "manual_label")

    results = {}
    for exp in args.exp:
        results[exp] = score_experiment(args.field, args.plot, args.method, exp, gt_dir)

    # --- table ---
    n_gt = next(iter(results.values()))["n_gt_images"]
    print("\n" + "=" * 74)
    print(f"  SAM BACKEND A/B — {args.field}/{args.plot}  (binary foreground vs {n_gt} GT mask(s))")
    print("=" * 74)
    print(f"  {'experiment':<20}{'IoU':>7}{'prec':>7}{'recall':>7}{'F1':>7}"
          f"{'VRAM':>8}{'RAM':>8}{'s/img':>7}")
    print("-" * 74)
    for exp in args.exp:
        m = results[exp]["mean"]
        p = results[exp].get("perf", {})
        vram = f"{p['peak_vram_alloc_gb']:.1f}G" if p.get("peak_vram_alloc_gb") else "-"
        ram = f"{p['peak_ram_rss_gb']:.1f}G" if p.get("peak_ram_rss_gb") else "-"
        spi = f"{p['avg_sec_per_image']:.2f}" if p.get("avg_sec_per_image") else "-"
        print(f"  {exp:<20}{m['IoU']:>7.3f}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['F1']:>7.3f}"
              f"{vram:>8}{ram:>8}{spi:>7}")
    print("-" * 74)
    print("  VRAM=peak allocated (torch), RAM=peak RSS, s/img=avg SAM time per image")
    # per-image head counts (useful: a big count gap flags merges/splits between backends)
    stems = sorted(next(iter(results.values()))["per_image"].keys())
    print("  head counts per GT image (‘?’ = union-only run, per-head count not saved):")
    for stem in stems:
        counts = "  ".join(
            f"{exp}={('?' if results[exp]['per_image'][stem]['n_heads'] < 0 else results[exp]['per_image'][stem]['n_heads'])}"
            for exp in args.exp)
        print(f"    {stem:<38} {counts}")
    print("=" * 74)
    print("  NOTE: throughput (sec/image) is in each run's SAM PLOT SUMMARY (mask-gen log), not here.")
    print("=" * 74 + "\n")

    out = args.out or os.path.join("docs", "analysis_results", "sam_backend_ab", f"{args.plot}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"field": args.field, "plot": args.plot, "method": args.method,
                   "gt_dir": gt_dir, "results": results}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
