"""Emit the compact phone confidence-sweep appendix table (tab:maskgen-phone-conf) straight from the
conf-sweep JSONs, so no number is hand-copied. Rows = the six swept cells (SAM1/SAM2 x full-frame/per-tile/
per-head) on the winning detector (YOLOv5 @4032); columns = best F1, the confidence it peaks at, and the
estimated AP, for both the IoU 0.5 headline bar and the looser IoU 0.3. --latex prints the tabular body
only; the caption is authored by hand in main.tex.

    python src/analysis/build_phone_conf_table.py            # human-readable check
    python src/analysis/build_phone_conf_table.py --latex    # LaTeX tabular for the appendix
"""
import argparse
import json
import os

BASE = "results/mask_generation/phone/evaluation/conf_sweep"
# display order: granularity blocks, SAM1 then SAM2 within each
ROWS = [
    ("full frame", "sam1_full_frame", "SAM1"), ("full frame", "sam2_full_frame", "SAM2"),
    ("per tile",   "sam1_per_tile",   "SAM1"), ("per tile",   "sam2_per_tile",   "SAM2"),
    ("per head",   "sam1_per_head",   "SAM1"), ("per head",   "sam2_per_head",   "SAM2"),
]


def _cell(name):
    """(bestF1, conf, AP) at IoU 0.5 and 0.3 for one swept cell."""
    c = json.load(open(f"{BASE}/{name}.json"))["curves"]
    out = {}
    for iou in ("0.5", "0.3"):
        cc = c[iou]
        out[iou] = (cc["best_f1"], cc["best_f1_conf"], cc["ap_estimate"])
    return out


def _load():
    """All rows as (granularity, sam, metrics-dict); flags any missing cell."""
    rows, missing = [], []
    for gran, name, sam in ROWS:
        p = f"{BASE}/{name}.json"
        if not os.path.exists(p):
            missing.append(name); continue
        rows.append((gran, sam, _cell(name)))
    return rows, missing


def emit_latex(rows):
    """The tab:maskgen-phone-conf body, numbers only. Caption is hand-authored in main.tex."""
    out = ["% phone conf-sweep table --- numbers auto-generated; caption authored in main.tex",
           "\\begin{table}[H]", "\\centering", "\\begin{tabular}{l l ccc ccc}", "\\hline",
           " & & \\multicolumn{3}{c}{IoU $\\ge$ 0.5} & \\multicolumn{3}{c}{IoU $\\ge$ 0.3} \\\\",
           "\\cline{3-5}\\cline{6-8}",
           "Granularity & SAM & F1 $\\uparrow$ & conf & AP $\\uparrow$ & F1 $\\uparrow$ & conf & AP $\\uparrow$ \\\\",
           "\\hline"]
    prev = None
    for gran, sam, m in rows:
        label = gran if gran != prev else ""
        prev = gran
        f5, c5, a5 = m["0.5"]; f3, c3, a3 = m["0.3"]
        out.append(f"{label} & {sam} & {f5:.3f} & {c5:.2f} & {a5:.3f} & {f3:.3f} & {c3:.2f} & {a3:.3f} \\\\")
        if sam == "SAM2":
            out.append("\\hline")
    out += ["\\end{tabular}", "\\caption{}  % caption authored in main.tex",
            "\\label{tab:maskgen-phone-conf}", "\\end{table}"]
    return "\n".join(out)


def emit_text(rows):
    """Plain readable table for eyeballing."""
    out = [f"{'gran':11}{'SAM':5}{'F1@.5':>7}{'conf':>6}{'AP@.5':>7}   {'F1@.3':>7}{'conf':>6}{'AP@.3':>7}"]
    for gran, sam, m in rows:
        f5, c5, a5 = m["0.5"]; f3, c3, a3 = m["0.3"]
        out.append(f"{gran:11}{sam:5}{f5:>7.3f}{c5:>6.2f}{a5:>7.3f}   {f3:>7.3f}{c3:>6.2f}{a3:>7.3f}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="print the LaTeX tabular (numbers only)")
    a = ap.parse_args()
    rows, missing = _load()
    if missing:
        import sys
        print(f"WARNING: missing conf-sweep cells: {missing}", file=sys.stderr)
    print(emit_latex(rows) if a.latex else emit_text(rows))


if __name__ == "__main__":
    main()
