"""Two-panel confidence figure for the phone mask-gen conf/AP section. Left panel: the precision-recall
curve (one point per swept YOLO confidence) with the estimated AP in the legend; right panel: F1 against
the confidence threshold, with the fixed 0.35 operating point marked. Both panels show the winning detector
(YOLOv5 @4032) at SAM2, per-tile and per-head, on the 6 phone GT images. Numbers come from the conf-sweep
JSONs (each mask carries its own box confidence, greedy-matched by confidence --- the standard AP build).

    python src/analysis/plot_maskgen_conf_pr.py                # IoU 0.5 -> maskgen_phone_conf_pr.png (main)
    python src/analysis/plot_maskgen_conf_pr.py --iou 0.3      # IoU 0.3 -> maskgen_phone_conf_pr_iou03.png (appendix)
"""
import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = "results/mask_generation/phone/evaluation/conf_sweep"
OPPT = 0.35  # the fixed operating point used by the grid
# (json cell, legend label, color) --- colors match the matched-IoU histogram (per-head = winner blue)
CELLS = [
    ("sam2_per_head", "per head", "#4C72B0"),
    ("sam2_per_tile", "per tile", "#55A868"),
    ("sam2_full_frame", "full frame", "#C44E52"),
]


def _curve(cell, iou):
    """Returns (rows, ap) for one cell at the requested IoU bar."""
    d = json.load(open(f"{BASE}/{cell}.json"))["curves"][iou]
    return d["rows"], d["ap_estimate"]


def _op_row(rows):
    """The swept row nearest the fixed 0.35 operating point."""
    return min(rows, key=lambda r: abs(r["conf"] - OPPT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou", default="0.5", choices=["0.5", "0.3"])
    a = ap.parse_args()
    suffix = "" if a.iou == "0.5" else "_iou03"
    out = f"thesis/figures/maskgen_phone_conf_pr{suffix}.png"

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for cell, label, color in CELLS:
        rows, apv = _curve(cell, a.iou)
        rec = [r["recall"] for r in rows]
        prec = [r["precision"] for r in rows]
        conf = [r["conf"] for r in rows]
        f1 = [r["f1"] for r in rows]
        # left: PR curve, dot at the 0.35 operating point, X at the F1 peak
        axl.plot(rec, prec, "-", color=color, linewidth=1.8, label=f"{label}  (AP {apv:.2f})")
        op = _op_row(rows)
        axl.plot(op["recall"], op["precision"], "o", color=color, markersize=5, zorder=5)
        best = max(rows, key=lambda r: r["f1"])
        axl.plot(best["recall"], best["precision"], "X", color=color, markersize=8,
                 markeredgecolor="white", markeredgewidth=0.6, zorder=6)
        # right: F1 vs confidence, X at each curve's F1 peak (the F1-optimal conf) — same marker as left
        axr.plot(conf, f1, "-", color=color, linewidth=1.8, label=label)
        axr.plot(best["conf"], best["f1"], "X", color=color, markersize=8,
                 markeredgecolor="white", markeredgewidth=0.6, zorder=6)
    axr.axvline(OPPT, color="0.4", linestyle="--", linewidth=1.0)
    axr.text(OPPT + 0.01, 0.02, "0.35", color="0.4", fontsize=8.5, transform=axr.get_xaxis_transform())

    axl.set_xlabel("recall"); axl.set_ylabel("precision")
    axl.set_xlim(0, None); axl.set_ylim(0, 1.0)
    # left legend: the three curves plus grey marker keys for the dot (0.35) and X (F1 peak)
    h, l = axl.get_legend_handles_labels()
    h += [Line2D([], [], marker="o", color="0.3", linestyle="None", markersize=5),
          Line2D([], [], marker="X", color="0.3", linestyle="None", markersize=8,
                 markeredgecolor="white", markeredgewidth=0.6)]
    l += ["0.35 conf", "F1 peak conf"]
    axl.legend(h, l, frameon=False, fontsize=9, loc="lower right")
    axr.set_xlabel("confidence threshold"); axr.set_ylabel("F1")
    axr.set_xlim(0, 1.0); axr.set_ylim(0, None)
    # right legend: the three curves plus grey keys for the 0.35 line and the X (F1 peak) — mirrors left
    hr, lr = axr.get_legend_handles_labels()
    hr += [Line2D([], [], color="0.4", linestyle="--", linewidth=1.0),
           Line2D([], [], marker="X", color="0.3", linestyle="None", markersize=8,
                  markeredgecolor="white", markeredgewidth=0.6)]
    lr += ["0.35 conf", "F1 peak conf"]
    axr.legend(hr, lr, frameon=False, fontsize=9, loc="upper right")
    for ax in (axl, axr):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200)
    for cell, label, _ in CELLS:
        rows, apv = _curve(cell, a.iou)
        best = max(rows, key=lambda r: r["f1"])
        print(f"{label:9} AP={apv:.3f}  bestF1={best['f1']:.3f}@{best['conf']:.2f}  "
              f"@0.35 F1={_op_row(rows)['f1']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
