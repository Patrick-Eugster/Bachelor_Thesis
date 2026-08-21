"""Plots the precision / recall / F1 (left axis) and false-positive COUNT (right axis)
vs confidence for one conf-sweep cell, so the 'flat-F1 / falling-FP / rising-precision'
shoulder is visible — the operating-point argument for 3D-seg seeding (precision matters,
per-image recall does not because each head is seen in ~90 views). Reads the JSON written
by sweep_conf_mask_ap.py. Defaults to the production cell SAM2+per_tile at IoU>=0.5."""
import json, os, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/mask_generation/phone/evaluation/conf_sweep")
    ap.add_argument("--cell", default="sam2_per_tile")
    ap.add_argument("--iou", default="0.5")
    ap.add_argument("--out", default="docs/analysis_results/conf_sweep")
    a = ap.parse_args()

    j = json.load(open(os.path.join(a.indir, a.cell + ".json")))
    rows = j["curves"][a.iou]["rows"]
    rows = [r for r in rows if r["recall"] > 0]        # drop the degenerate conf=0.95 (all-zero)
    conf = [r["conf"] for r in rows]
    prec = [r["precision"] for r in rows]
    rec = [r["recall"] for r in rows]
    f1 = [r["f1"] for r in rows]
    fp = [r["fp"] for r in rows]
    bf_conf = j["curves"][a.iou]["best_f1_conf"]

    fig, axL = plt.subplots(figsize=(8, 5))
    axL.plot(conf, prec, "-o", ms=3, color="#1f77b4", label="precision")
    axL.plot(conf, rec, "-o", ms=3, color="#ff7f0e", label="recall")
    axL.plot(conf, f1, "-o", ms=3, color="#2ca02c", label="F1")
    axL.set_xlabel("YOLO conf threshold")
    axL.set_ylabel("precision / recall / F1")
    axL.set_ylim(0, 1)
    axL.set_xlim(0, 1)
    axL.xaxis.set_major_locator(MultipleLocator(0.1))    # labelled every 0.1
    axL.xaxis.set_minor_locator(MultipleLocator(0.05))   # faint line every 0.05
    axL.grid(which="major", alpha=0.3)
    axL.grid(which="minor", alpha=0.12)

    axR = axL.twinx()                                   # FP count on the right axis
    axR.plot(conf, fp, "--s", ms=3, color="#d62728", label="FP count (junk masks)")
    axR.set_ylabel("false-positive mask count", color="#d62728")
    axR.tick_params(axis="y", labelcolor="#d62728")

    # mark the F1-optimum conf — named in the legend so its meaning is explicit
    axL.axvline(bf_conf, color="gray", ls=":", lw=1.2, label=f"best-F1 conf ({bf_conf})")

    lines = axL.get_lines() + axR.get_lines()
    axL.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
    plt.title(f"{a.cell}  |  IoU >= {a.iou}  (6 GT imgs, pooled)")
    fig.tight_layout()

    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"tradeoff_{a.cell}_iou{a.iou}.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


if __name__ == "__main__":
    main()
