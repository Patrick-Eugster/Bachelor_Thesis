"""Assemble the E2 full-res YOLOv5 SAM2 cells next to the existing phone mask-gen grid, so the new
"YOLOv5 @4032" block can be read against "YOLOv5 @1280", SAHI, and YOLO11 at SAM2. Baselines come from
grid_summary.csv (micro F1 = 2PR/(P+R), P=TP/pred, R=TP/GT); the three full-res cells are recomputed the
SAME micro way from their eval_masks_instance.json, so every number in the table is comparable. Read-only."""
import argparse
import csv
import json
import os

# the three full-res cells produced by e2_yolov5_fullres_grid{,_sam1}.sh (eval_experiment -> granularity).
# The SAM version is selected by --sam so the same assembler emits the SAM1 or the SAM2 block.
def _fullres_exps(sam):
    """maps granularity -> the e2 full-res eval_experiment name for this SAM version."""
    return {"full_frame": f"e2_ff_{sam}", "per_tile": f"e2_pt_{sam}", "per_head": f"e2_ph_{sam}"}


def _exp_gran(sam):
    """grid_summary exp naming -> granularity, for this SAM version's baseline rows."""
    return {f"gt_ff_{sam}": "full_frame", f"gt_tile_{sam}": "per_tile", f"gt_head_{sam}": "per_head"}


EVAL_ROOT = "results/mask_generation/phone/evaluation/yolo_sam_v1/masks_instance"
GRID_CSV = "results/mask_generation/phone/evaluation/grid_summary.csv"
METHOD_LABEL = {"yolo_sam_v1": "YOLOv5 @1280", "sahi_yolo_sam": "SAHI", "yolo11_sam": "YOLO11"}
GRAN_ORDER = ["full_frame", "per_tile", "per_head"]


def _micro_from_json(path):
    """Micro F1/P/R + tp-weighted mean matched IoU from an eval_masks_instance.json (grid_summary's way)."""
    imgs = json.load(open(path)).get("images", [])
    tp = sum(i.get("at_threshold", {}).get("tp", 0) for i in imgs)
    pred = sum(i.get("n_pred", 0) for i in imgs)
    gt = sum(i.get("n_gt", 0) for i in imgs)
    p = tp / pred if pred else 0.0
    r = tp / gt if gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    iou = (sum(i.get("at_threshold", {}).get("mean_iou_matched", 0.0) * i.get("at_threshold", {}).get("tp", 0)
               for i in imgs) / tp) if tp else 0.0
    return {"f1": f1, "precision": p, "recall": r, "mean_iou": iou, "pred": pred, "gt": gt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam", default="sam2", choices=["sam1", "sam2"],
                    help="which SAM version's full-res block + baselines to assemble")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sam = a.sam
    out = a.out or f"docs/analysis_results/e2_fullres_grid/E2_fullres_yolov5_{sam}.md"
    fullres_exps = _fullres_exps(sam)
    exp_gran = _exp_gran(sam)

    # baselines from grid_summary (this SAM version only)
    base = {}   # (method, gran) -> dict
    for row in csv.DictReader(open(GRID_CSV)):
        if row["exp"] in exp_gran and row["method"] in METHOD_LABEL:
            base[(row["method"], exp_gran[row["exp"]])] = {
                "f1": float(row["F1"]), "precision": float(row["precision"]),
                "recall": float(row["recall"]), "mean_iou": float(row["mean_iou"])}

    # the three full-res cells, recomputed micro
    full = {}
    for gran, exp in fullres_exps.items():
        jp = os.path.join(EVAL_ROOT, exp, "eval_masks_instance.json")
        full[gran] = _micro_from_json(jp) if os.path.exists(jp) else None

    lines = [f"# E2 — full-res YOLOv5 in the phone mask-gen grid ({sam.upper()})",
             "",
             "All cells scored on the 6 pinhole GT images, Hungarian match at IoU 0.5, conf 0.35, micro-",
             "averaged F1 (= 2PR/(P+R), P=TP/pred, R=TP/GT) — identical to grid_summary.csv, so the new",
             "YOLOv5 @4032 rows are directly comparable to the existing detectors.",
             "",
             "| detector | granularity | F1 | precision | recall | matched-mask IoU |",
             "|---|---|---|---|---|---|"]

    def row(label, d):
        if d is None:
            return f"| {label} | — | (not run) | | | |"
        return (f"| {label} | {gran.replace('_',' ')} | {d['f1']:.3f} | "
                f"{d['precision']:.3f} | {d['recall']:.3f} | {d['mean_iou']:.3f} |")

    # order: for each granularity, print the 4 detectors together so the full-res block sits with the rest
    for gran in GRAN_ORDER:
        for m in ("yolo_sam_v1", "sahi_yolo_sam", "yolo11_sam"):
            d = base.get((m, gran))
            if d:
                lines.append(f"| {METHOD_LABEL[m]} | {gran.replace('_',' ')} | {d['f1']:.3f} | "
                             f"{d['precision']:.3f} | {d['recall']:.3f} | {d['mean_iou']:.3f} |")
        d = full.get(gran)
        lines.append(row("**YOLOv5 @4032**", d) if d else f"| **YOLOv5 @4032** | {gran} | (not run) | | | |")
        lines.append("|  |  |  |  |  |  |")

    # quick verdict at per_tile / per_head
    def cmp_line(gran):
        f = full.get(gran)
        if not f:
            return None
        y1280 = base.get(("yolo_sam_v1", gran), {}).get("f1")
        sahi = base.get(("sahi_yolo_sam", gran), {}).get("f1")
        y11 = base.get(("yolo11_sam", gran), {}).get("f1")
        return (f"- **{gran.replace('_',' ')}:** YOLOv5 @4032 F1 = {f['f1']:.3f}  "
                f"(vs YOLOv5 @1280 {y1280:.3f}, SAHI {sahi:.3f}, YOLO11 {y11:.3f})")

    lines += ["", "## Reading it"]
    for gran in GRAN_ORDER:
        c = cmp_line(gran)
        if c:
            lines.append(c)
    lines += ["",
              "If YOLOv5 @4032 reaches or beats SAHI/YOLO11 at per_tile/per_head, the resolution — not the",
              "model or the tiling — was the main lever, and SAHI's tiling is redundant for phone. If it",
              "still trails, tiling buys something a single full-res pass does not (anchor/receptive-field),",
              "and SAHI stays justified; the fix is then to report YOLOv5 at its fair resolution, not 1280."]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
