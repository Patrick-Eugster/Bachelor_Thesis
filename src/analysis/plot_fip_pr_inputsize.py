"""Overlays the precision-recall curves of the FIP input-size sweep (YOLOv5 at 640/1280/1920/2560 px)
on one axis, so the AP scalars in the thesis input-size table become a visible curve. The AP and the
matching are reproduced exactly from src/mask_generation/evaluation/eval_yolo_boxes.py (global pooling
of all predictions across the seven labeled plots, greedy IoU-0.5 match, COCO 101-point AP), so the
per-size AP printed here should equal the table. The fixed 0.35 operating point is marked on each curve.
Output goes to docs/analysis_results/ for eyeballing before anything is added to the thesis."""
import os
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SIZES = [640, 1280, 1920, 2560]
PLOTS = [f"plot_{n}" for n in range(461, 468)]
CONF_OP = 0.35   # the fixed operating point used everywhere in the pipeline
IOU_THR = 0.5
RUN = "fip_imgsz_{s}"
CONF_DIR = "results/mask_generation/fip/{plot}/yolo_sam_v1/" + RUN + "/bboxes_with_conf/{stem}.pt"
OUT = "docs/analysis_results/fip_pr_inputsize.png"

# multi-hue sequential palette (viridis): distinct hues AND ordered lightness, light = small
# input, dark = large input, so the curves stay easy to tell apart while still reading in order.
COLORS = {s: c for s, c in zip(SIZES, plt.cm.viridis(np.linspace(0.85, 0.12, len(SIZES))))}


def load_gt_boxes(label_path, img_w, img_h):
    """Reads YOLO-format GT (class cx cy w h, normalized) and returns xyxy pixel boxes.
    Copied verbatim from eval_yolo_boxes.py so the match is identical."""
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = (float(x) for x in parts[:5])
            boxes.append([(cx - w / 2) * img_w, (cy - h / 2) * img_h,
                          (cx + w / 2) * img_w, (cy + h / 2) * img_h])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def compute_iou_matrix(pred_boxes, gt_boxes):
    """Vectorized pairwise IoU, shape (N_pred, N_gt). Copied from eval_yolo_boxes.py."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    p = pred_boxes[:, np.newaxis, :]
    g = gt_boxes[np.newaxis, :, :]
    ix1 = np.maximum(p[:, :, 0], g[:, :, 0]); iy1 = np.maximum(p[:, :, 1], g[:, :, 1])
    ix2 = np.minimum(p[:, :, 2], g[:, :, 2]); iy2 = np.minimum(p[:, :, 3], g[:, :, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    pa = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    ga = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = pa[:, np.newaxis] + ga[np.newaxis, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def compute_ap(all_pred_entries, all_gt_boxes_list, iou_threshold):
    """Global-pool AP, precisions, recalls, confs. Copied verbatim from eval_yolo_boxes.py."""
    n_gt_total = sum(len(gt) for gt in all_gt_boxes_list)
    if n_gt_total == 0 or len(all_pred_entries) == 0:
        return 0.0, [], [], []
    sorted_preds = sorted(all_pred_entries, key=lambda x: -x[0])
    matched_gt = [set() for _ in all_gt_boxes_list]
    tp_list, fp_list = [], []
    for _conf, x1, y1, x2, y2, img_idx in sorted_preds:
        gt_boxes = all_gt_boxes_list[img_idx]
        if len(gt_boxes) == 0:
            tp_list.append(0); fp_list.append(1); continue
        iou_row = compute_iou_matrix(np.array([[x1, y1, x2, y2]], dtype=np.float32), gt_boxes)[0]
        best_gt_idx, best_iou = -1, -1.0
        for gt_idx in range(len(gt_boxes)):
            if gt_idx not in matched_gt[img_idx] and iou_row[gt_idx] > best_iou:
                best_iou = iou_row[gt_idx]; best_gt_idx = gt_idx
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            tp_list.append(1); fp_list.append(0); matched_gt[img_idx].add(best_gt_idx)
        else:
            tp_list.append(0); fp_list.append(1)
    tp_cum = np.cumsum(tp_list); fp_cum = np.cumsum(fp_list)
    recalls = tp_cum / n_gt_total
    precisions = tp_cum / (tp_cum + fp_cum)
    ap = 0.0
    for r_thr in np.linspace(0, 1, 101):
        prec_at_r = precisions[recalls >= r_thr]
        ap += float(np.max(prec_at_r)) if len(prec_at_r) > 0 else 0.0
    ap /= 101
    confs = [float(e[0]) for e in sorted_preds]
    return float(ap), precisions.tolist(), recalls.tolist(), confs


def pool_size(s):
    """Pools the seven labeled plots' predictions + GT for one input size, then returns the AP curve."""
    entries, gts = [], []
    for i, plot in enumerate(PLOTS):
        lbl = glob.glob(f"input_plots/fip/{plot}/manual_label/*.txt")[0]
        stem = os.path.splitext(os.path.basename(lbl))[0]
        img = f"input_plots/fip/{plot}/images/{stem}.png"
        w, h = Image.open(img).size
        gts.append(load_gt_boxes(lbl, w, h))
        t = torch.load(CONF_DIR.format(plot=plot, s=s, stem=stem), weights_only=True).numpy()
        for x1, y1, x2, y2, c in t:
            entries.append((float(c), x1, y1, x2, y2, i))
    return compute_ap(entries, gts, IOU_THR)


def op_point(precisions, recalls, confs, thr):
    """Returns the (recall, precision) on the curve at the last prediction with conf >= thr."""
    idx = [k for k, c in enumerate(confs) if c >= thr]
    if not idx:
        return None
    j = idx[-1]
    return recalls[j], precisions[j]


def main():
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for s in SIZES:
        ap, prec, rec, confs = pool_size(s)
        ax.plot(rec, prec, color=COLORS[s], lw=2, label=f"{s} px  (AP {ap:.3f})")
        op = op_point(prec, rec, confs, CONF_OP)
        if op:
            ax.plot(op[0], op[1], "o", color=COLORS[s], ms=7, mec="black", mew=0.7, zorder=5)
        print(f"{s:>4} px : AP {ap:.3f}   op@{CONF_OP} -> recall {op[0]:.3f}, precision {op[1]:.3f}")
    ax.plot([], [], "o", color="white", mec="black", mew=0.7, label=f"operating point (conf {CONF_OP})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xticks(np.linspace(0, 1, 11)); ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(True, which="major", alpha=0.3)
    ax.set_title("Average precision of the input-size sweep (FIP, boxes matched at IoU 0.5)")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
