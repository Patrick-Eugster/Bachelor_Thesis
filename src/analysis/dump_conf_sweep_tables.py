"""Dumps the full conf-sweep numbers (every cell, every IoU, every conf) into one
markdown file so we can eyeball the precision/recall/FP tradeoff before deciding on an
operating point for the thesis. Reads the per-cell JSONs written by sweep_conf_mask_ap.py
(scored on Euler) and writes FULL_TABLES.md. Purely a formatter — no scoring here."""
import json, glob, os, argparse


def _fmt_cell(name, curve):
    """One markdown table for a single {backend,mode,iou} cell, all conf rows."""
    out = [f"### {name}",
           f"best_f1 = {curve['best_f1']:.3f} @ conf {curve['best_f1_conf']}   |   AP_est = {curve['ap_estimate']:.3f}",
           "",
           "| conf | prec | recall | f1 | tp | fp | fn |",
           "|---|---|---|---|---|---|---|"]
    for r in curve["rows"]:
        out.append(f"| {r['conf']:.2f} | {r['precision']:.3f} | {r['recall']:.3f} | "
                   f"{r['f1']:.3f} | {r['tp']} | {r['fp']} | {r['fn']} |")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/mask_generation/phone/evaluation/conf_sweep")
    ap.add_argument("--out", default="docs/analysis_results/conf_sweep/FULL_TABLES.md")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.indir, "*.json")))
    if not files:
        raise SystemExit(f"no JSONs in {a.indir}")
    cells = {os.path.basename(f)[:-5]: json.load(open(f)) for f in files}
    # nicest reading order: fast/accurate granularity first, full_frame last
    order = ["sam2_per_tile", "sam1_per_tile", "sam2_per_head",
             "sam1_per_head", "sam2_full_frame", "sam1_full_frame"]
    order = [c for c in order if c in cells] + [c for c in cells if c not in order]
    ious = cells[order[0]]["iou_thresholds"]

    doc = ["# Phone mask-gen conf sweep — full tables",
           "",
           "Mask-level instance-seg scores over the 6 hand-labeled GT images (3196 GT heads).",
           "Each SAM mask carries its own YOLO box confidence; masks are greedy-matched to GT",
           "instance masks by mask-IoU. Cells = {SAM1,SAM2} x {per_tile, per_head, full_frame}.",
           "per_tile is the fast production mode (~7-10 s/img).",
           "",
           "**Counts (tp/fp/fn):** `tp` = mask that correctly hit a GT head (IoU >= bar);",
           "`fp` = junk mask that matched no head (or a duplicate on an already-taken head) —",
           "these pollute 3D-seg matching; `fn` = GT head no mask found. Always tp+fn = 3196.",
           "precision = tp/(tp+fp), recall = tp/(tp+fn).",
           "",
           "**Aggregation:** all 6 GT images are ADDED into one pile (micro-average), NOT",
           "averaged per-image — so the dense 551-head image weighs more than a sparse one.",
           "That's why total_gt = 3196 (sum of the 6 images' head counts).",
           "",
           "## Reading it for 3D-seg seeding (why NOT the F1 peak)",
           "- Each head is seen in ~90 views, so **per-image recall barely matters** — a head",
           "  only needs ONE clean mask in ONE view to seed its 3D id. Missing it in a given",
           "  image is cheap (another view catches it).",
           "- A false-positive mask is NOT cheap: it seeds a spurious 3D object or drags a real",
           "  head's cross-view match-IoU below threshold. So **precision >> recall** here.",
           "- Therefore prefer a conf ABOVE the F1-optimum, on the shoulder where F1 is still",
           "  nearly flat but the `fp` column has dropped sharply. For SAM2+per_tile @ IoU>=0.5",
           "  that shoulder is ~conf 0.42-0.48 (prec ~0.81-0.83, fp ~260-330 vs 634 at the",
           "  F1-peak 0.26) — do NOT keep pushing past ~0.50 where tp/recall bleed fast.",
           "- Don't chase the low-conf edge either: it maximizes recall we don't need while",
           "  carrying the most junk.",
           ""]
    for iou in ious:
        doc.append(f"\n---\n\n## IoU >= {iou}\n")
        for c in order:
            doc.append(_fmt_cell(c, cells[c]["curves"][str(iou)]))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(doc))
    print(f"wrote {a.out}  ({len(order)} cells x {len(ious)} IoU)")


if __name__ == "__main__":
    main()
