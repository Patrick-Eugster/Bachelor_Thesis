"""Load the manual wheat-head-count ground truth from the FIP reprocessed xlsx.

The supervisor's `wheat_head_counts.xlsx` (in
demoanlage2025_v0/FIP_single_row_exp_2024_reprocessed/) has one row per plot with the manual head
count for each of the 6 wheat rows: columns `lot, plot, row1_wheat_head_count ... row6_wheat_head_count,
comments`. The plot-level GT total is the sum of the 6 row counts.

This is the GT to compare against the pipeline's predicted total head count
(`segmentation_3d/<exp>/seg_summary.json` -> `wheat_heads_found`, written by run_3d_seg.py).

Usage (print the table):
    python src/segmentation_3d/headcount_gt.py --xlsx demoanlage2025_v0/FIP_single_row_exp_2024_reprocessed/wheat_head_counts.xlsx
"""

import argparse
import openpyxl


def load_headcount_gt(xlsx_path):
    """Read the head-count xlsx and return {plot_id(int): {"total": int, "rows": [6 ints], "comments": str|None}}.
    Plot total = sum of the row1..row6 head-count columns."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else "" for h in rows[0]]

    # locate the columns we need by name so we don't hardcode positions
    plot_i = header.index("plot")
    row_cols = [i for i, h in enumerate(header) if h.startswith("row") and h.endswith("wheat_head_count")]
    comment_i = header.index("comments") if "comments" in header else None

    out = {}
    for r in rows[1:]:
        if r[plot_i] is None:
            continue
        plot_id = int(r[plot_i])
        row_counts = [int(r[i]) for i in row_cols if r[i] is not None]
        out[plot_id] = {
            "total": sum(row_counts),
            "rows": row_counts,
            "comments": r[comment_i] if comment_i is not None else None,
        }
    return out


def main():
    """CLI: print the per-plot GT head counts (row breakdown + total)."""
    ap = argparse.ArgumentParser(description="Print manual wheat-head-count GT from the FIP xlsx.")
    ap.add_argument("--xlsx", required=True, help="path to wheat_head_counts.xlsx")
    args = ap.parse_args()

    gt = load_headcount_gt(args.xlsx)
    print(f"{'plot':>6} {'total':>6}   rows")
    for plot_id in sorted(gt):
        g = gt[plot_id]
        note = f"   # {g['comments']}" if g["comments"] else ""
        print(f"{plot_id:>6} {g['total']:>6}   {g['rows']}{note}")


if __name__ == "__main__":
    main()
