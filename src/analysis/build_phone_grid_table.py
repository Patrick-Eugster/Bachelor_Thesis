"""Emit the phone mask-gen F1 grid (tab:maskgen-phone-grid) straight from the scored data, so no
number is hand-copied. The three original detectors (YOLOv5 @1280, SAHI, YOLO11) come from
grid_summary.csv, whose F1 is already the micro F1 (= 2PR/(P+R), P=TP/pred, R=TP/GT); the full-res
YOLOv5 @4032 block is recomputed the SAME micro way from the e2_{ff,pt,ph}_{sam1,2,3} eval JSONs,
so every cell is comparable. A missing cell (e.g. SAM3 @4032 before its Euler run lands) prints as
'---' with a warning on stderr, so the table can be built now and refilled when the JSON arrives.
The best present cell is bolded. Caption is authored by hand in main.tex; --latex prints the tabular
body only.

    python src/analysis/build_phone_grid_table.py            # human-readable check
    python src/analysis/build_phone_grid_table.py --latex    # LaTeX tabular for the thesis
"""
import argparse
import csv
import json
import os
import sys

GRID_CSV = "results/mask_generation/phone/evaluation/grid_summary.csv"
E2_EVAL = "results/mask_generation/phone/evaluation/yolo_sam_v1/masks_instance/{exp}/eval_masks_instance.json"

# display granularity -> (grid_summary code, e2 code); grid uses tile/head, e2 uses pt/ph
GRAN = [("full frame", "ff", "ff"), ("per tile", "tile", "pt"), ("per head", "head", "ph")]
SAMS = ["sam1", "sam2", "sam3"]

# detector rows in table order. 'grid' -> grid_summary method; 'e2' -> full-res JSONs.
# YOLOv5 @1280 and @4032 sit adjacent so the resolution jump reads directly.
DETECTORS = [
    ("YOLOv5 @1280", "grid", "yolo_sam_v1"),
    ("YOLOv5 @4032", "e2",   None),
    ("SAHI",         "grid", "sahi_yolo_sam"),
    ("YOLO11",       "grid", "yolo11_sam"),
]


def _grid_f1():
    """(method, exp) -> F1 float, straight from grid_summary.csv."""
    return {(r["method"], r["exp"]): float(r["F1"]) for r in csv.DictReader(open(GRID_CSV))}


def _e2_micro_f1(exp):
    """Micro F1 (grid_summary's way) from an e2 eval JSON, or None if the JSON is missing."""
    path = E2_EVAL.format(exp=exp)
    if not os.path.exists(path):
        return None
    imgs = json.load(open(path)).get("images", [])
    tp = sum(i.get("at_threshold", {}).get("tp", 0) for i in imgs)
    pred = sum(i.get("n_pred", 0) for i in imgs)
    gt = sum(i.get("n_gt", 0) for i in imgs)
    p = tp / pred if pred else 0.0
    r = tp / gt if gt else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def build():
    """Collect every (detector, granularity, sam) F1 into a dict, plus the yolo11-seg value.
    Returns (cells, seg_f1, top_rounded, missing_exps)."""
    grid = _grid_f1()
    missing = []
    cells = {}
    for di, (_, source, method) in enumerate(DETECTORS):
        for gi, (_, gcode, ecode) in enumerate(GRAN):
            for si, sam in enumerate(SAMS):
                if source == "grid":
                    v = grid.get((method, f"gt_{gcode}_{sam}"))
                else:
                    v = _e2_micro_f1(f"e2_{ecode}_{sam}")
                    if v is None:
                        missing.append(f"e2_{ecode}_{sam}")
                cells[(di, gi, si)] = v
    seg = grid.get(("yolo11_seg", "gt_eval"))
    present = [v for v in cells.values() if v is not None]
    top = round(max(present), 3) if present else None
    return cells, seg, top, missing


def _fmt(v, top):
    """Format one cell: '---' if missing, bolded if it equals the top rounded F1."""
    if v is None:
        return "---"
    s = f"{v:.3f}"
    return f"\\textbf{{{s}}}" if top is not None and round(v, 3) == top else s


def emit_latex(cells, seg, top):
    """The tab:maskgen-phone-grid tabular, numbers only. Caption is hand-authored in main.tex."""
    out = ["% phone mask-gen F1 grid --- numbers auto-generated; caption authored in main.tex",
           "\\begin{table}[H]", "\\centering", "\\begin{tabular}{l l ccc}", "\\hline",
           "Detector & Granularity & SAM1 & SAM2 & SAM3 \\\\", "\\hline"]
    for di, (label, _, _) in enumerate(DETECTORS):
        for gi, (gdisp, _, _) in enumerate(GRAN):
            row_label = label if gi == 0 else ""
            vals = " & ".join(_fmt(cells[(di, gi, si)], top) for si in range(len(SAMS)))
            out.append(f"{row_label} & {gdisp} & {vals} \\\\")
        out.append("\\hline")
    seg_s = _fmt(seg, top) if seg is not None else "---"
    out.append(f"\\multicolumn{{2}}{{l}}{{yolo11-seg (direct)}} & \\multicolumn{{3}}{{c}}{{{seg_s}}} \\\\")
    out += ["\\hline", "\\end{tabular}", "\\caption{}  % caption authored in main.tex",
            "\\label{tab:maskgen-phone-grid}", "\\end{table}"]
    return "\n".join(out)


def emit_text(cells, seg, top):
    """Plain readable grid for eyeballing."""
    out = [f"{'Detector':<14}{'Granularity':<12}{'SAM1':>7}{'SAM2':>7}{'SAM3':>7}"]
    for di, (label, _, _) in enumerate(DETECTORS):
        for gi, (gdisp, _, _) in enumerate(GRAN):
            row_label = label if gi == 0 else ""
            vals = "".join(f"{('---' if cells[(di,gi,si)] is None else f'{cells[(di,gi,si)]:.3f}'):>7}"
                           for si in range(len(SAMS)))
            out.append(f"{row_label:<14}{gdisp:<12}{vals}")
    out.append(f"{'yolo11-seg':<14}{'(direct)':<12}{('---' if seg is None else f'{seg:.3f}'):>7}")
    out.append(f"(bold/top rounded F1 = {top})")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="print the LaTeX tabular (numbers only)")
    a = ap.parse_args()
    cells, seg, top, missing = build()
    if missing:
        print(f"WARNING: {len(missing)} cell(s) not scored yet, printed as '---': {sorted(set(missing))}",
              file=sys.stderr)
    print(emit_latex(cells, seg, top) if a.latex else emit_text(cells, seg, top))


if __name__ == "__main__":
    main()
