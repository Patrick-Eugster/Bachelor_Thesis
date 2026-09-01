"""Emit the per-plot FIP 3D-seg quality tables straight from the seg_yv5_1280 eval_2d
metrics, so both the Results IoU-only table and the full appendix table are reproducible.

Two axes of the recon-settings A/B (all on the SAME yv5_1280 masks, 15k gsplat):
  fipseg15k_ppoff   = baseline  (pp OFF, absgrad OFF)
  fipseg15k_pp      = +pp       (pp ON,  absgrad OFF)
  fipseg15k_absgrad = +absgrad  (pp ON,  absgrad ON)

Prints LaTeX tabular bodies ONLY (no caption/label/wrapper -- those are hand-authored
in main.tex). Read-only; writes nothing.
  --which results  -> Results table: IoU per plot, 3 config columns
  --which appendix -> Appendix table: all 6 metrics per plot, grouped by config
"""
import argparse
import json
import os

ROOT = "results/reconstruction/fip"
PLOTS = [f"plot_46{i}" for i in range(1, 8)]
EXPS = [("baseline", "fipseg15k_ppoff"), ("+pp", "fipseg15k_pp"), ("+absgrad", "fipseg15k_absgrad")]
MET = ["iou", "precision", "recall", "f1", "mse", "ssim"]
# detector axis: three mask sets segged on the SAME model (fipseg15k_pp = pp on, absgrad off),
# each written to its own seg-name folder.
DET = [("640", "seg_yv5_640"), ("1280", "seg_yv5_1280"), ("yolo11", "seg_yolo11")]


def metrics(exp, plot, seg="seg_yv5_1280"):
    """the eval_2d metric dict for one plot's seg run (json is a 1-elem list)."""
    f = os.path.join(ROOT, plot, "vanilla_3dgs", exp, "segmentation_3d", seg,
                     "eval_2d", "metrics_2d.json")
    return json.load(open(f))[0]


def results_table():
    """Results table body: one row per plot, IoU under baseline / +pp / +absgrad, then a mean row."""
    acc = {e: 0.0 for _, e in EXPS}
    for p in PLOTS:
        cells = []
        for _, e in EXPS:
            v = metrics(e, p)["iou"]
            acc[e] += v
            cells.append(f"{v:.3f}")
        print(f"{p[-3:]} & " + " & ".join(cells) + r" \\")
    print(r"\hline")
    print(r"mean & " + " & ".join(f"{acc[e] / len(PLOTS):.3f}" for _, e in EXPS) + r" \\")


def appendix_table():
    """Appendix table body: grouped by config, one row per plot over all 6 metrics, mean per group."""
    for lbl, e in EXPS:
        print(r"\multicolumn{7}{l}{\textit{" + lbl + r"}} \\")
        acc = [0.0] * len(MET)
        for p in PLOTS:
            m = metrics(e, p)
            vals = [m[k] for k in MET]
            acc = [a + v for a, v in zip(acc, vals)]
            print(f"{p[-3:]} & " + " & ".join(f"{v:.3f}" for v in vals) + r" \\")
        print("mean & " + " & ".join(f"{a / len(PLOTS):.3f}" for a in acc) + r" \\")
        print(r"\hline")


def detector_table():
    """Appendix detector table body: IoU per plot under the three mask sets, on fixed model fipseg15k_pp."""
    acc = {s: 0.0 for _, s in DET}
    for p in PLOTS:
        cells = []
        for _, s in DET:
            v = metrics("fipseg15k_pp", p, seg=s)["iou"]
            acc[s] += v
            cells.append(f"{v:.3f}")
        print(f"{p[-3:]} & " + " & ".join(cells) + r" \\")
    print(r"\hline")
    print(r"mean & " + " & ".join(f"{acc[s] / len(PLOTS):.3f}" for _, s in DET) + r" \\")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["results", "appendix", "detector"], required=True)
    args = ap.parse_args()
    {"results": results_table, "appendix": appendix_table, "detector": detector_table}[args.which]()
