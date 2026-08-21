"""Plot + aggregate the conf-sweep JSONs written by sweep_conf_mask_ap.py (one per {backend}_{mode} cell).

Reads every <indir>/*.json, and for a chosen IoU threshold produces:
  - per cell:  F1-vs-conf and a PR curve  -> <outdir>/<cell>_iou{thr}.png
  - overlay :  F1-vs-conf for ALL cells on one axis -> <outdir>/_overlay_f1_iou{thr}.png
  - a summary table (best-F1 conf, best F1, AP estimate per cell) printed + saved as _summary_iou{thr}.md

This is the Issue-1 readout: the best conf per SAM mode/backend and which mode wins. Matplotlib only.

  python src/analysis/plot_conf_sweep.py \
      --indir results/mask_generation/phone/evaluation/conf_sweep \
      --outdir docs/analysis_results/conf_sweep --iou 0.5
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402


def _style_conf_axis(ax):
    """Label the conf axis every 0.1 and draw a faint minor gridline every 0.05, so the
    reader can read off any dot's conf even though we can't label all ~31 sweep steps."""
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.grid(which="major", alpha=0.3)
    ax.grid(which="minor", alpha=0.12)


def _load_cells(indir):
    """Load every *.json in indir -> list of result dicts, sorted by (mode, backend) for stable colors."""
    cells = []
    for p in sorted(glob.glob(os.path.join(indir, "*.json"))):
        d = json.load(open(p))
        d["_name"] = f"{d.get('backend','?')}_{d.get('mode','?')}"
        cells.append(d)
    return cells


def _curve(cell, iou_key):
    """Return (confs, precision, recall, f1) lists for one cell at the given IoU key, or None if absent."""
    cur = cell.get("curves", {}).get(iou_key)
    if not cur:
        return None
    rows = cur["rows"]
    return ([r["conf"] for r in rows], [r["precision"] for r in rows],
            [r["recall"] for r in rows], [r["f1"] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/mask_generation/phone/evaluation/conf_sweep")
    ap.add_argument("--outdir", default="docs/analysis_results/conf_sweep")
    ap.add_argument("--iou", type=float, default=0.5, help="which IoU-threshold curve to plot")
    args = ap.parse_args()
    iou_key = str(args.iou)
    os.makedirs(args.outdir, exist_ok=True)

    cells = _load_cells(args.indir)
    if not cells:
        raise SystemExit(f"no *.json in {args.indir}")
    print(f"{len(cells)} cell(s) from {args.indir} | IoU>={args.iou}")

    # per-cell F1-vs-conf + PR
    for cell in cells:
        c = _curve(cell, iou_key)
        if c is None:
            print(f"  {cell['_name']}: no IoU={args.iou} curve, skipping"); continue
        confs, prec, rec, f1 = c
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        a1.plot(confs, f1, "-o", ms=3, label="F1")
        a1.plot(confs, prec, "-", alpha=.6, label="precision")
        a1.plot(confs, rec, "-", alpha=.6, label="recall")
        bi = max(range(len(f1)), key=lambda i: f1[i])
        a1.axvline(confs[bi], color="k", ls=":", lw=1, label=f"best-F1 conf ({confs[bi]})")
        a1.set_title(f"{cell['_name']}  best F1={f1[bi]:.3f}@{confs[bi]}")
        a1.set_xlabel("YOLO conf threshold"); a1.set_ylabel("score"); a1.set_ylim(0, 1); a1.legend(fontsize=8)
        _style_conf_axis(a1)
        a2.plot(rec, prec, "-o", ms=3)
        # label only the meaningful conf anchors: the two ends, the F1-opt, and the 3 3D-seg
        # test candidates — enough to orient the curve without labelling all ~31 dots.
        labelled = False
        for tc in (0.05, 0.26, 0.40, 0.55, 0.70, 0.90):
            j = min(range(len(confs)), key=lambda i: abs(confs[i] - tc))
            if abs(confs[j] - tc) < 0.02:                 # nearest dot (0.55 isn't on the grid -> 0.54)
                a2.annotate(f"{confs[j]:.2f}", (rec[j], prec[j]), textcoords="offset points",
                            xytext=(5, 4), fontsize=7, color="#333")
                a2.plot(rec[j], prec[j], "o", ms=5, color="#d62728",
                        label="YOLO conf threshold" if not labelled else None)
                labelled = True
        a2.set_title(f"PR (IoU>={args.iou})"); a2.set_xlabel("recall"); a2.set_ylabel("precision")
        a2.set_xlim(0, 1); a2.set_ylim(0, 1); a2.legend(fontsize=8, loc="lower left")
        a2.xaxis.set_major_locator(MultipleLocator(0.1)); a2.xaxis.set_minor_locator(MultipleLocator(0.05))
        a2.yaxis.set_major_locator(MultipleLocator(0.1)); a2.yaxis.set_minor_locator(MultipleLocator(0.05))
        a2.grid(which="major", alpha=0.3); a2.grid(which="minor", alpha=0.12)
        fig.tight_layout()
        out = os.path.join(args.outdir, f"{cell['_name']}_iou{args.iou}.png")
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"  wrote {out}")

    # overlay F1-vs-conf
    fig, axf = plt.subplots(figsize=(7.5, 5))
    for cell in cells:
        c = _curve(cell, iou_key)
        if c is None:
            continue
        confs, _, _, f1 = c
        axf.plot(confs, f1, "-o", ms=2.5, label=cell["_name"])
    axf.set_title(f"F1 vs conf — all cells (IoU>={args.iou})")
    axf.set_xlabel("YOLO conf threshold"); axf.set_ylabel("F1"); axf.set_ylim(0, 1); axf.legend(fontsize=8)
    _style_conf_axis(axf)
    fig.tight_layout()
    ov = os.path.join(args.outdir, f"_overlay_f1_iou{args.iou}.png")
    fig.savefig(ov, dpi=130); plt.close(fig)
    print(f"  wrote {ov}")

    # summary table
    lines = [f"# Conf-sweep summary (IoU>={args.iou})  —  {len(cells)} cells\n",
             "| cell | best F1 | best conf | AP est | n_img | total_gt |",
             "|---|---|---|---|---|---|"]
    rowsum = []
    for cell in cells:
        cur = cell.get("curves", {}).get(iou_key)
        if not cur:
            continue
        rowsum.append((cur["best_f1"], cell["_name"], cur["best_f1_conf"], cur["ap_estimate"],
                       cell.get("n_images"), cell.get("total_gt")))
    for bf, name, bc, apx, ni, ng in sorted(rowsum, reverse=True):   # best F1 first
        lines.append(f"| {name} | {bf:.3f} | {bc} | {apx:.3f} | {ni} | {ng} |")
    md = "\n".join(lines) + "\n"
    print("\n" + md)
    sp = os.path.join(args.outdir, f"_summary_iou{args.iou}.md")
    open(sp, "w").write(md)
    print(f"  wrote {sp}")


if __name__ == "__main__":
    main()
