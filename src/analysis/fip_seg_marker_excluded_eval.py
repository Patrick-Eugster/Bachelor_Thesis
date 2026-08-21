"""Second, marker-excluded 2D seg eval for the FIP detector bridge, computed locally from saved data and
WITHOUT touching any existing results file. For each of the 21 runs (7 plots x 3 detector mask sets) it
rebuilds the binary prediction from the per-variant 2DSeg label map (pred = any head id > 0) and the
binary GT from the manual mask, then excludes a disk of radius R around each calibration-marker center
taken from the reprocessed marker_projections.csv (the same centers eyeballed in marker_csv_check). It
reports precision/recall/F1/IoU with the markers included (a self-check that must match the stored
metrics_2d.json) and with them excluded. Results are written to a NEW json + printed; nothing under
results/ is overwritten. Run from repo root.

Caveat: the CSV lists only markers Agisoft pinned for that camera, so a visible-but-unpinned marker is
NOT excluded -> this is an approximate cleanup, not a complete one."""
import csv
import glob
import json
import os
import numpy as np
import cv2
import torch
from PIL import Image

PLOTS = [f"plot_{n}" for n in range(461, 468)]
DETS = ["seg_yv5_640", "seg_yv5_1280", "seg_yolo11"]
SEG = "results/reconstruction/fip/{p}/vanilla_3dgs/fipseg15k_pp/segmentation_3d/{d}"
MARKER_CSV = "demoanlage2025_v0/FIP_single_row_exp_2024_reprocessed/{p}/marker_projections.csv"
R = 275
OUT = "docs/analysis_results/fip_seg_marker_excluded_eval.json"


def gt_stem(p):
    return os.path.splitext(os.path.basename(glob.glob(f"input_plots/fip/{p}/manual_label/*.txt")[0]))[0]


def marker_disk_mask(p, stem, shape):
    """Boolean mask of the R-radius disks around each CSV marker center for this camera (True = exclude)."""
    m = np.zeros(shape, np.uint8)
    for r in csv.DictReader(open(MARKER_CSV.format(p=p))):
        if r["Camera"] == stem:
            cv2.circle(m, (int(round(float(r["X"]))), int(round(float(r["Y"])))), R, 1, -1)
    return m.astype(bool)


def scores(gt, pred, valid):
    """Precision/recall/F1/IoU over the valid (non-excluded) pixels."""
    g, p = gt & valid, pred & valid
    tp = int((g & p).sum()); fp = int((~g & p).sum()); fn = int((g & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return dict(precision=prec, recall=rec, f1=f1, iou=iou)


def main():
    rows = []
    max_selfcheck_err = 0.0
    for p in PLOTS:
        stem = gt_stem(p)
        gt = np.array(Image.open(f"input_plots/fip/{p}/manual_label/{stem}_gt_mask.png").convert("L")) >= 128
        excl = marker_disk_mask(p, stem, gt.shape)
        valid_all = np.ones(gt.shape, bool)
        valid_ex = ~excl
        for d in DETS:
            lb = torch.load(f"{SEG.format(p=p, d=d)}/2DSeg/{stem}.pt", weights_only=True)
            lb = lb.numpy() if hasattr(lb, "numpy") else np.array(lb)
            if lb.shape != gt.shape:
                lb = cv2.resize(lb.astype(np.int32), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred = lb > 0
            raw = scores(gt, pred, valid_all)
            exj = scores(gt, pred, valid_ex)
            stored = json.load(open(f"{SEG.format(p=p, d=d)}/eval_2d/metrics_2d.json"))[0]["precision"]
            max_selfcheck_err = max(max_selfcheck_err, abs(raw["precision"] - stored))
            rows.append(dict(plot=p, detector=d, stored_precision=stored, raw=raw, marker_excluded=exj))

    # per-detector means
    means = {}
    for d in DETS:
        sub = [r for r in rows if r["detector"] == d]
        means[d] = {kind: {k: float(np.mean([r[kind][k] for r in sub])) for k in ("iou", "precision", "recall", "f1")}
                    for kind in ("raw", "marker_excluded")}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(radius_px=R, self_check_max_precision_err=max_selfcheck_err, per_run=rows, means=means),
              open(OUT, "w"), indent=2)

    print(f"self-check: max |recompute - stored| precision = {max_selfcheck_err:.4f} (should be ~0)\n")
    print(f"{'detector':<14} {'':<10}{'IoU':>7}{'P':>7}{'R':>7}{'F1':>7}")
    for d in DETS:
        for kind in ("raw", "marker_excluded"):
            m = means[d][kind]
            print(f"{d:<14} {kind:<14}{m['iou']:>7.3f}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}")
    print(f"\nplot_466 / seg_yv5_1280 (the contaminated cell):")
    r466 = [r for r in rows if r["plot"] == "plot_466" and r["detector"] == "seg_yv5_1280"][0]
    print(f"  raw            P {r466['raw']['precision']:.3f}  IoU {r466['raw']['iou']:.3f}")
    print(f"  marker-excluded P {r466['marker_excluded']['precision']:.3f}  IoU {r466['marker_excluded']['iou']:.3f}")
    print(f"\nwrote {OUT}  (nothing under results/ was modified)")


if __name__ == "__main__":
    main()
