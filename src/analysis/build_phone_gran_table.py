"""Emit the phone mask-gen granularity table (tab:maskgen-phone-gran) from the scored data, so no
number is hand-copied. Each row (full frame / per tile / per head) is the simple mean over the four
detector setups x three SAM versions at that granularity --- twelve cells. The three baseline detectors
(YOLOv5 @1280, SAHI, YOLO11) come from grid_summary.csv; the fourth, YOLOv5 @4032, is pooled the SAME
way (aggregate_maskgen_grid.pool) from the e2_{ff,pt,ph}_{sam1,2,3} eval JSONs, so it averages in on
equal footing. YOLOv5 is counted at both its resolutions on purpose --- this table averages over every
configuration that was run, not one per model. --latex prints the tabular body only; the caption is
authored by hand in main.tex.

    python src/analysis/build_phone_gran_table.py            # human-readable check
    python src/analysis/build_phone_gran_table.py --latex    # LaTeX tabular for the thesis
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from aggregate_maskgen_grid import pool  # identical pooling so the @4032 cells match the baselines

GRID_CSV = "results/mask_generation/phone/evaluation/grid_summary.csv"
E2_EVAL = "results/mask_generation/phone/evaluation/yolo_sam_v1/masks_instance/{exp}/eval_masks_instance.json"

# granularity display -> (grid_summary exp code, e2 exp code); grid uses tile/head, e2 uses pt/ph
GRAN = [("full frame", "ff", "ff"), ("per tile", "tile", "pt"), ("per head", "head", "ph")]
SAMS = ["sam1", "sam2", "sam3"]
BASE_METHODS = ["yolo_sam_v1", "sahi_yolo_sam", "yolo11_sam"]  # the three @1280/native baselines


def _baseline_cells():
    """(method, exp) -> metric dict, straight from grid_summary.csv."""
    return {(r["method"], r["exp"]): r for r in csv.DictReader(open(GRID_CSV))}


def collect():
    """For each granularity, gather the twelve cell metric dicts (4 detectors x 3 SAM). Returns
    (per-gran list of cells, missing_exps)."""
    base = _baseline_cells()
    out, missing = {}, []
    for disp, gcode, ecode in GRAN:
        cells = []
        for m in BASE_METHODS:
            for s in SAMS:
                cells.append(base[(m, f"gt_{gcode}_{s}")])
        for s in SAMS:
            path = E2_EVAL.format(exp=f"e2_{ecode}_{s}")
            if not os.path.exists(path):
                missing.append(f"e2_{ecode}_{s}")
                continue
            cells.append(pool(path))
        out[disp] = cells
    return out, missing


def _avg(cells, key):
    """Simple mean of one metric over the cells (merges rounded for display)."""
    vals = [float(c[key]) for c in cells]
    return sum(vals) / len(vals)


def emit_latex(per_gran):
    """The tab:maskgen-phone-gran body, numbers only. Caption is hand-authored in main.tex."""
    out = ["% phone mask-gen granularity table --- numbers auto-generated; caption authored in main.tex",
           "\\begin{table}[H]", "\\centering", "\\begin{tabular}{l cccccc}", "\\hline",
           "Granularity & F1 $\\uparrow$ & precision $\\uparrow$ & recall $\\uparrow$ & IoU$_m$ $\\uparrow$ & boundary IoU $\\uparrow$ & merges $\\downarrow$ \\\\",
           "\\hline"]
    for disp, _, _ in GRAN:
        c = per_gran[disp]
        out.append(f"{disp} & {_avg(c,'F1'):.3f} & {_avg(c,'precision'):.3f} & {_avg(c,'recall'):.3f} & "
                   f"{_avg(c,'mean_iou'):.3f} & {_avg(c,'boundary_iou'):.3f} & {_avg(c,'merges'):.0f} \\\\")
    out += ["\\hline", "\\end{tabular}", "\\caption{}  % caption authored in main.tex",
            "\\label{tab:maskgen-phone-gran}", "\\end{table}"]
    return "\n".join(out)


def emit_text(per_gran):
    """Plain readable table for eyeballing (with the cell count per row)."""
    out = [f"{'Granularity':11} {'F1':>6} {'prec':>6} {'recall':>6} {'IoU_m':>6} {'bIoU':>6} {'merges':>7}  n"]
    for disp, _, _ in GRAN:
        c = per_gran[disp]
        out.append(f"{disp:11} {_avg(c,'F1'):>6.3f} {_avg(c,'precision'):>6.3f} {_avg(c,'recall'):>6.3f} "
                   f"{_avg(c,'mean_iou'):>6.3f} {_avg(c,'boundary_iou'):>6.3f} {_avg(c,'merges'):>7.0f}  {len(c)}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="print the LaTeX tabular (numbers only)")
    a = ap.parse_args()
    per_gran, missing = collect()
    if missing:
        print(f"WARNING: {len(missing)} @4032 cell(s) not scored yet, skipped: {sorted(set(missing))}",
              file=sys.stderr)
    print(emit_latex(per_gran) if a.latex else emit_text(per_gran))


if __name__ == "__main__":
    main()
