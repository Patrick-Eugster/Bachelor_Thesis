"""
compare_yolo_sahi.py — compare YOLO vs SAHI box counts per image for ONE session, and
plot the detection-confidence distribution so the conf threshold can be tuned.

It reads the saved boxes from each detector's result tree:
  results/mask_generation/<dataset>/<session>/<method>/<exp>/bboxes_with_conf/*.pt   (5 cols: x1,y1,x2,y2,conf)
  ... falling back to .../bboxes/*.pt (4 cols, no conf) if the conf files aren't there.

Per image it counts boxes (>= --count_threshold when confidences are available) and matches YOLO
to SAHI by image stem. Outputs to results/analysis/yolo_sahi_<session>/:
  - counts.csv              per-image yolo/sahi counts (+ diff)
  - count_compare.png       per-image comparison (scatter vs identity + sorted lines)
  - conf_hist.png           confidence histograms (yolo vs sahi) — only if conf is available
  - count_vs_threshold.png  total kept boxes as the conf threshold sweeps 0..1 — the tuning curve

NOTE on SAHI confidences: SAHI runs its tiles AT the keep threshold by design, so to see the full
sub-threshold distribution you must regenerate SAHI with a LOW conf_threshold_good_box. Because SAHI's
NMM merge can reshape boxes when low-conf boxes are present, a SAHI count obtained by filtering a
low-threshold run at t is only APPROXIMATELY equal to a production SAHI run AT t (YOLO has no such
caveat — NMS suppresses, never reshapes). Treat the SAHI histogram as indicative for tuning.
"""

import os
import glob
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_boxes(folder):
    """Read every *.pt in a bboxes folder → {stem: Nx? float array}. Handles empty tensors and
    both the 4-col (xyxy) and 5-col (xyxy+conf) layouts."""
    out = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.pt"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        t = torch.load(f)
        a = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
        if a.ndim != 2 or a.shape[0] == 0:
            out[stem] = np.zeros((0, a.shape[1] if a.ndim == 2 else 4), dtype=np.float32)
        else:
            out[stem] = a.astype(np.float32)
    return out


def _resolve_dir(dataset, session, method, exp):
    """Prefer bboxes_with_conf (has confidence) over bboxes (count-only). Returns (folder, has_conf)."""
    base = os.path.join("results/mask_generation", dataset, session, method, exp)
    conf_dir = os.path.join(base, "bboxes_with_conf")
    if os.path.isdir(conf_dir) and glob.glob(os.path.join(conf_dir, "*.pt")):
        return conf_dir, True
    return os.path.join(base, "bboxes"), False


def _count_at(boxes, has_conf, thr):
    """Box count for one image at a confidence threshold (just the row count when no conf is saved)."""
    if boxes.shape[0] == 0:
        return 0
    if has_conf and boxes.shape[1] >= 5:
        return int((boxes[:, 4] >= thr).sum())
    return int(boxes.shape[0])


def main():
    ap = argparse.ArgumentParser(description="Compare YOLO vs SAHI box counts + confidence for one session.")
    ap.add_argument("--dataset", default="phone")
    ap.add_argument("--session", required=True, help="e.g. field_A/20250715")
    ap.add_argument("--yolo_exp", default="initial")
    ap.add_argument("--sahi_exp", default="initial")
    ap.add_argument("--count_threshold", type=float, default=0.35,
                    help="conf threshold used for the per-image count (only applies if conf is saved)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    yolo_dir, yolo_conf = _resolve_dir(args.dataset, args.session, "yolo_sam_v1", args.yolo_exp)
    sahi_dir, sahi_conf = _resolve_dir(args.dataset, args.session, "sahi_yolo_sam", args.sahi_exp)
    print(f"YOLO boxes: {yolo_dir}   (conf available: {yolo_conf})")
    print(f"SAHI boxes: {sahi_dir}   (conf available: {sahi_conf})")

    yolo = _load_boxes(yolo_dir)
    sahi = _load_boxes(sahi_dir)
    stems = sorted(set(yolo) | set(sahi))
    if not stems:
        print("No boxes found for either detector — check session/exp names.")
        return

    out = args.out or os.path.join("results/analysis", "yolo_sahi_" + args.session.replace("/", "_"))
    os.makedirs(out, exist_ok=True)
    t = args.count_threshold

    # --- per-image counts + CSV ---
    rows, y_counts, s_counts = [], [], []
    for stem in stems:
        yc = _count_at(yolo.get(stem, np.zeros((0, 4))), yolo_conf, t)
        sc = _count_at(sahi.get(stem, np.zeros((0, 4))), sahi_conf, t)
        rows.append((stem, yc, sc, sc - yc))
        y_counts.append(yc)
        s_counts.append(sc)
    y_counts, s_counts = np.array(y_counts), np.array(s_counts)

    csv_path = os.path.join(out, "counts.csv")
    with open(csv_path, "w") as f:
        f.write("image,yolo_count,sahi_count,sahi_minus_yolo\n")
        for stem, yc, sc, d in rows:
            f.write(f"{stem},{yc},{sc},{d}\n")

    print(f"\nimages: {len(stems)}   threshold for counts: {t}")
    print(f"YOLO total heads: {y_counts.sum()}   mean/img: {y_counts.mean():.1f}")
    print(f"SAHI total heads: {s_counts.sum()}   mean/img: {s_counts.mean():.1f}")
    print(f"SAHI−YOLO per image: mean {(s_counts-y_counts).mean():+.1f}  (SAHI finds more where positive)")

    # --- count comparison figure (scatter vs identity + sorted lines) ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    lim = max(y_counts.max(), s_counts.max()) * 1.05 + 1
    ax[0].scatter(y_counts, s_counts, s=18, alpha=0.6)
    ax[0].plot([0, lim], [0, lim], "k--", lw=1, label="equal")
    ax[0].set_xlabel("YOLO boxes / image"); ax[0].set_ylabel("SAHI boxes / image")
    ax[0].set_title(f"per-image count (thr={t})"); ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
    ax[0].legend()
    order = np.argsort(y_counts)
    ax[1].plot(y_counts[order], label=f"YOLO (Σ={y_counts.sum()})")
    ax[1].plot(s_counts[order], label=f"SAHI (Σ={s_counts.sum()})", alpha=0.8)
    ax[1].set_xlabel("image (sorted by YOLO count)"); ax[1].set_ylabel("boxes / image")
    ax[1].set_title("per-image counts, sorted"); ax[1].legend()
    fig.suptitle(f"{args.session}  —  YOLO vs SAHI box counts")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "count_compare.png"), dpi=110)
    plt.close(fig)

    # --- confidence histogram of the KEPT boxes (needs conf) ---
    # We histogram the confidence of every box we actually keep (conf >= count_threshold) so you can
    # see where they cluster: if most kept boxes are well above the threshold, you can safely raise it.
    if yolo_conf or sahi_conf:
        def _kept_conf(boxes_dict, has_conf):
            if not has_conf:
                return np.array([])
            c = np.concatenate([b[:, 4] for b in boxes_dict.values() if b.shape[0] and b.shape[1] >= 5]) \
                if any(b.shape[0] for b in boxes_dict.values()) else np.array([])
            return c[c >= t]  # only the boxes we keep at the current threshold

        yconf = _kept_conf(yolo, yolo_conf)
        sconf = _kept_conf(sahi, sahi_conf)

        def _pct(c, name):
            if not c.size:
                return
            p = np.percentile(c, [10, 25, 50, 75, 90])
            print(f"{name} kept-box conf: n={c.size}  mean={c.mean():.3f}  "
                  f"p10={p[0]:.2f} p25={p[1]:.2f} median={p[2]:.2f} p75={p[3]:.2f} p90={p[4]:.2f}")
            for cut in (0.4, 0.5, 0.6, 0.7):
                print(f"    raise threshold to {cut:.1f} → keep {(c >= cut).mean()*100:5.1f}% of current {name} boxes")
        print()
        _pct(yconf, "YOLO"); _pct(sconf, "SAHI")

        fig, ax = plt.subplots(figsize=(11, 5))
        bins = np.linspace(t, 1, 40)
        if yconf.size:
            ax.hist(yconf, bins=bins, alpha=0.5, label=f"YOLO (n={yconf.size}, median {np.median(yconf):.2f})")
        if sconf.size:
            ax.hist(sconf, bins=bins, alpha=0.5, label=f"SAHI (n={sconf.size}, median {np.median(sconf):.2f})")
        ax.axvline(t, color="r", ls="--", lw=1, label=f"current threshold {t}")
        ax.set_xlabel("confidence of kept boxes"); ax.set_ylabel("# boxes")
        ax.set_title(f"{args.session} — confidence distribution of the boxes we keep (thr={t})")
        ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(out, "conf_hist.png"), dpi=110); plt.close(fig)
        print("\nwrote: counts.csv, count_compare.png, conf_hist.png")
    else:
        print("wrote: counts.csv, count_compare.png  (no confidence saved → re-run with save_bboxes_conf=true for the histogram)")
    print(f"output dir: {out}")


if __name__ == "__main__":
    main()
