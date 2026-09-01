"""Emit the FIP head-count tables: manual whole-plot head count vs the number of distinct
3D head identities the segmentation assigns (wheat_heads_found), for the three detector mask
sets on the fixed fipseg15k_pp model (same runs as the detector quality table).

  true count = sum of the six per-row manual counts per plot (docs/data/wheat_head_counts*.xlsx)
  assigned   = seg_summary.json "wheat_heads_found" (distinct 3D heads over the whole plot)

Prints LaTeX tabular bodies ONLY (captions hand-authored in main.tex). Read-only.
  --which results  -> compact: one row per mask set, mean assigned + mean |count error|
  --which appendix -> full: one row per plot (true, 640, 1280, yolo11) + a mean-|err%| row
"""
import argparse
import glob
import json
import os

ROOT = "results/reconstruction/fip"
PLOTS = list(range(461, 468))
DET = [("YOLOv5 @640", "seg_yv5_640"), ("YOLOv5 @1280", "seg_yv5_1280"), ("YOLO11 @3008", "seg_yolo11")]


def true_counts():
    """whole-plot manual head count per plot = sum of the six row-counts in the xlsx."""
    import openpyxl
    path = sorted(glob.glob("docs/data/wheat_head_counts*.xlsx"))[0]
    ws = openpyxl.load_workbook(path, data_only=True).active
    rows = list(ws.iter_rows(values_only=True))
    hdr = rows[0]
    p_i = hdr.index("plot")
    row_cols = [i for i, h in enumerate(hdr) if h and str(h).startswith("row")]
    out = {}
    for r in rows[1:]:
        if r[p_i] is None:
            continue
        out[int(r[p_i])] = sum(int(r[i]) for i in row_cols if r[i] is not None)
    return out


def assigned(plot, seg):
    """distinct 3D heads the segmentation found on the whole plot (pipeline's own count)."""
    f = os.path.join(ROOT, f"plot_{plot}", "vanilla_3dgs", "fipseg15k_pp",
                     "segmentation_3d", seg, "seg_summary.json")
    return json.load(open(f))["wheat_heads_found"]


def results_table():
    """compact body: per mask set, mean assigned heads and mean absolute count error."""
    true = true_counts()
    for lbl, seg in DET:
        a = [assigned(p, seg) for p in PLOTS]
        mean_a = sum(a) / len(PLOTS)
        mean_err = sum(abs(assigned(p, seg) - true[p]) / true[p] for p in PLOTS) / len(PLOTS)
        print(f"{lbl} & {mean_a:.0f} & {mean_err * 100:.1f}\\% \\\\")


def appendix_table():
    """full body: per plot the true count and the three assigned counts, then a mean-|err%| row."""
    true = true_counts()
    errs = {seg: [] for _, seg in DET}
    for p in PLOTS:
        cells = []
        for _, seg in DET:
            a = assigned(p, seg)
            errs[seg].append(abs(a - true[p]) / true[p])
            cells.append(str(a))
        print(f"{p} & {true[p]} & " + " & ".join(cells) + r" \\")
    print(r"\hline")
    print(r"mean $|$err$|$ & --- & "
          + " & ".join(f"{sum(errs[seg]) / len(PLOTS) * 100:.1f}\\%" for _, seg in DET) + r" \\")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["results", "appendix"], required=True)
    args = ap.parse_args()
    (results_table if args.which == "results" else appendix_table)()
