"""Emit the full per-cell phone mask-gen appendix table (tab:maskgen-phone-full) straight from the
scored data, so no number is hand-copied. The 28 baseline cells (YOLOv5 @1280, SAHI, YOLO11, and the
direct yolo11-seg) come from grid_summary.csv; the 9 full-res YOLOv5 @4032 cells are pooled the SAME
way (aggregate_maskgen_grid.pool) from their e2_{ff,pt,ph}_{sam1,2,3} eval JSONs, so every row is
comparable. Rows are ranked by F1. A missing e2 cell prints as '---' with a stderr warning. --latex
prints the tabular body only; the caption is authored by hand in main.tex.

    python src/analysis/build_phone_full_table.py            # human-readable check
    python src/analysis/build_phone_full_table.py --latex    # LaTeX tabular for the appendix
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from aggregate_maskgen_grid import pool  # identical pooling so @4032 rows match the baselines

GRID_CSV = "results/mask_generation/phone/evaluation/grid_summary.csv"
E2_EVAL = "results/mask_generation/phone/evaluation/yolo_sam_v1/masks_instance/{exp}/eval_masks_instance.json"

DET_NAME = {"yolo_sam_v1": "YOLOv5 @1280", "sahi_yolo_sam": "SAHI",
            "yolo11_sam": "YOLO11", "yolo11_seg": "yolo11-seg"}
GRAN_ABBR = {"full frame": "ff", "per tile": "pt", "per head": "ph"}
SAM_ABBR = {"SAM1": "S1", "SAM2": "S2", "SAM3": "S3"}
# full-res block: (granularity, e2 exp code)
E2_GRAN = [("full frame", "ff"), ("per tile", "pt"), ("per head", "ph")]
E2_SAMS = [("sam1", "SAM1"), ("sam2", "SAM2"), ("sam3", "SAM3")]


def _gran(exp):
    """Experiment name -> granularity label (or None for the direct segmenter)."""
    if "_ff_" in exp:
        return "full frame"
    if "_tile_" in exp:
        return "per tile"
    if "_head_" in exp:
        return "per head"
    return None


def _sam(exp):
    """SAM version off the experiment-name suffix, or None."""
    for tag, name in (("sam1", "SAM1"), ("sam2", "SAM2"), ("sam3", "SAM3")):
        if exp.endswith(tag):
            return name
    return None


def _row_from(d, det, gran, sam):
    """One appendix row-dict from a pooled/CSV metric dict."""
    return {"det": det, "gran": gran, "sam": sam,
            "F1": float(d["F1"]), "recall": float(d["recall"]), "precision": float(d["precision"]),
            "mean_iou": float(d["mean_iou"]),
            "boundary_iou": float(d["boundary_iou"]), "boundary_iou_dyn": float(d["boundary_iou_dyn"]),
            "boundary_f": float(d["boundary_f"]), "boundary_f_dyn": float(d["boundary_f_dyn"]),
            "merges": int(float(d["merges"])), "splits": int(float(d["splits"]))}


def collect():
    """Every appendix row: 28 baselines from the CSV + 9 @4032 cells pooled from JSON. Returns
    (rows, missing_exps)."""
    rows, missing = [], []
    for r in csv.DictReader(open(GRID_CSV)):
        det = DET_NAME[r["method"]]
        rows.append(_row_from(r, det, _gran(r["exp"]), _sam(r["exp"])))
    for gran, gcode in E2_GRAN:
        for stag, sname in E2_SAMS:
            path = E2_EVAL.format(exp=f"e2_{gcode}_{stag}")
            if not os.path.exists(path):
                missing.append(f"e2_{gcode}_{stag}")
                continue
            rows.append(_row_from(pool(path), "YOLOv5 @4032", gran, sname))
    rows.sort(key=lambda r: -r["F1"])
    return rows, missing


def emit_latex(rows):
    """The tab:maskgen-phone-full body, numbers only. Caption is hand-authored in main.tex."""
    top = round(max(r["F1"] for r in rows), 3)
    out = ["% phone mask-gen full per-cell table --- numbers auto-generated; caption authored in main.tex",
           "\\begin{table}[H]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{4pt}",
           "\\begin{tabular}{l l l c c c c cc cc c c}", "\\hline",
           " & & & & & & & \\multicolumn{2}{c}{bIoU $\\uparrow$} & \\multicolumn{2}{c}{bF $\\uparrow$} & & \\\\",
           "\\cline{8-9}\\cline{10-11}",
           "Detector & Gran & S & F1 $\\uparrow$ & R $\\uparrow$ & P $\\uparrow$ & IoU$_m$ $\\uparrow$ "
           "& fix & dyn & fix & dyn & mrg $\\downarrow$ & spl $\\downarrow$ \\\\", "\\hline"]
    for r in rows:
        g = GRAN_ABBR.get(r["gran"], "---")
        s = SAM_ABBR.get(r["sam"], "---")
        f1 = f"\\textbf{{{r['F1']:.3f}}}" if round(r["F1"], 3) == top else f"{r['F1']:.3f}"
        out.append(f"{r['det']} & {g} & {s} & {f1} & {r['recall']:.3f} & {r['precision']:.3f} & "
                   f"{r['mean_iou']:.3f} & {r['boundary_iou']:.3f} & {r['boundary_iou_dyn']:.3f} & "
                   f"{r['boundary_f']:.3f} & {r['boundary_f_dyn']:.3f} & {r['merges']} & {r['splits']} \\\\")
    out += ["\\hline", "\\end{tabular}", "\\caption{}  % caption authored in main.tex",
            "\\label{tab:maskgen-phone-full}", "\\end{table}"]
    return "\n".join(out)


def emit_text(rows):
    """Plain readable ranking for eyeballing."""
    out = [f"{'Detector':<13}{'gr':>3}{'S':>3}{'F1':>7}{'R':>7}{'P':>7}{'IoU':>7}"
           f"{'bIoU':>7}{'bIoUd':>7}{'mrg':>5}{'spl':>5}"]
    for r in rows:
        out.append(f"{r['det']:<13}{GRAN_ABBR.get(r['gran'],'--'):>3}{SAM_ABBR.get(r['sam'],'--'):>3}"
                   f"{r['F1']:>7.3f}{r['recall']:>7.3f}{r['precision']:>7.3f}{r['mean_iou']:>7.3f}"
                   f"{r['boundary_iou']:>7.3f}{r['boundary_iou_dyn']:>7.3f}{r['merges']:>5d}{r['splits']:>5d}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="print the LaTeX tabular (numbers only)")
    a = ap.parse_args()
    rows, missing = collect()
    if missing:
        print(f"WARNING: {len(missing)} @4032 cell(s) not scored yet, skipped: {sorted(set(missing))}",
              file=sys.stderr)
    print(emit_latex(rows) if a.latex else emit_text(rows))


if __name__ == "__main__":
    main()
