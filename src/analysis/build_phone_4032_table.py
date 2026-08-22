"""Emit the YOLOv5 @4032 detail table (tab:maskgen-phone-4032) for the phone mask-gen grid: F1,
precision, and recall for each SAM version (SAM1/2/3) at each granularity (full frame / per tile /
per head), for the winning detector only. This is the table the Results prose leans on when it
compares, say, SAM2 at per tile versus per head, where the main F1 grid shows F1 alone. Every cell is
pooled the same way as the rest of the grid (aggregate_maskgen_grid.pool over the six GT images) from
the e2_{ff,pt,ph}_{sam1,2,3} eval JSONs, so the F1 column matches the YOLOv5 @4032 block of the main
grid exactly. --latex prints the tabular body only; the caption is authored by hand in main.tex.

    python src/analysis/build_phone_4032_table.py            # human-readable check
    python src/analysis/build_phone_4032_table.py --latex    # LaTeX tabular for the thesis
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from aggregate_maskgen_grid import pool  # identical pooling so the cells match the main grid

E2_EVAL = "results/mask_generation/phone/evaluation/yolo_sam_v1/masks_instance/e2_{g}_{s}/eval_masks_instance.json"
GRAN = [("full frame", "ff"), ("per tile", "pt"), ("per head", "ph")]
SAMS = [("SAM1", "sam1"), ("SAM2", "sam2"), ("SAM3", "sam3")]


def collect():
    """(gran display, sam display) -> pooled metric dict for every YOLOv5 @4032 cell that was scored."""
    out, missing = {}, []
    for gdisp, gcode in GRAN:
        for sdisp, scode in SAMS:
            path = E2_EVAL.format(g=gcode, s=scode)
            if not os.path.exists(path):
                missing.append(f"e2_{gcode}_{scode}")
                continue
            out[(gdisp, sdisp)] = pool(path)
    return out, missing


def emit_latex(cells):
    """The tab:maskgen-phone-4032 body, numbers only. Caption authored in main.tex.
    The best F1 in each granularity block is bolded."""
    out = ["% phone mask-gen YOLOv5 @4032 detail table --- numbers auto-generated; caption in main.tex",
           "\\begin{table}[H]", "\\centering", "\\begin{tabular}{l l ccc}", "\\hline",
           "Granularity & SAM & F1 $\\uparrow$ & precision $\\uparrow$ & recall $\\uparrow$ \\\\", "\\hline"]
    for gi, (gdisp, _) in enumerate(GRAN):
        best = max((float(cells[(gdisp, s)]["F1"]) for s, _ in SAMS if (gdisp, s) in cells), default=None)
        for si, (sdisp, _) in enumerate(SAMS):
            if (gdisp, sdisp) not in cells:
                continue
            c = cells[(gdisp, sdisp)]
            f1 = float(c["F1"])
            f1s = f"\\textbf{{{f1:.3f}}}" if best is not None and abs(f1 - best) < 1e-9 else f"{f1:.3f}"
            gcol = gdisp if si == 0 else ""
            out.append(f"{gcol} & {sdisp} & {f1s} & {float(c['precision']):.3f} & {float(c['recall']):.3f} \\\\")
        if gi < len(GRAN) - 1:
            out.append("\\hline")
    out += ["\\hline", "\\end{tabular}", "\\caption{}  % caption authored in main.tex",
            "\\label{tab:maskgen-phone-4032}", "\\end{table}"]
    return "\n".join(out)


def emit_text(cells):
    """Plain readable table for eyeballing."""
    out = [f"{'Granularity':11} {'SAM':4} {'F1':>6} {'prec':>6} {'recall':>6}"]
    for gdisp, _ in GRAN:
        for sdisp, _ in SAMS:
            if (gdisp, sdisp) not in cells:
                continue
            c = cells[(gdisp, sdisp)]
            out.append(f"{gdisp:11} {sdisp:4} {float(c['F1']):>6.3f} "
                       f"{float(c['precision']):>6.3f} {float(c['recall']):>6.3f}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="print the LaTeX tabular (numbers only)")
    a = ap.parse_args()
    cells, missing = collect()
    if missing:
        print(f"WARNING: {len(missing)} cell(s) not scored yet, skipped: {sorted(set(missing))}",
              file=sys.stderr)
    print(emit_latex(cells) if a.latex else emit_text(cells))


if __name__ == "__main__":
    main()
