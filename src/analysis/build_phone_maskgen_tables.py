"""Regenerates the phone mask-generation result tables + the exact numbers quoted in the
thesis Results text, from the aggregated grid_summary.csv. This is the single source for
Table A (F1 matrix), Table B (granularity effect), and the appendix 28-row table, so the
thesis numbers can always be re-derived instead of being copied by hand.

Run: python src/analysis/build_phone_maskgen_tables.py
Input: results/mask_generation/phone/evaluation/grid_summary.csv (gitignored eval output).
"""
import csv
import glob
import json
import os
import collections

CSV = "results/mask_generation/phone/evaluation/grid_summary.csv"
DET_NAME = {"yolo_sam_v1": "YOLOv5", "sahi_yolo_sam": "SAHI",
            "yolo11_sam": "YOLO11", "yolo11_seg": "yolo11-seg"}
GRAN_ORDER = ["full frame", "per tile", "per head"]
SAM_ORDER = ["SAM1", "SAM2", "SAM3"]


def granularity(exp):
    """Maps an experiment name to its SAM granularity, or None for the direct segmenter."""
    if "_ff_" in exp:
        return "full frame"
    if "_tile_" in exp:
        return "per tile"
    if "_head_" in exp:
        return "per head"
    return None


def sam_version(exp):
    """Reads the SAM version off the experiment-name suffix (sam1/sam2/sam3)."""
    for tag, name in (("sam1", "SAM1"), ("sam2", "SAM2"), ("sam3", "SAM3")):
        if exp.endswith(tag):
            return name
    return None


def load():
    """Loads the grid rows and tags each with detector / granularity / SAM version."""
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        r["det"] = DET_NAME[r["method"]]
        r["gran"] = granularity(r["exp"])
        r["sam"] = sam_version(r["exp"])
        for k in ("F1", "recall", "precision", "mean_iou", "boundary_iou", "boundary_f",
                  "boundary_iou_dyn", "boundary_f_dyn"):
            r[k] = float(r[k])
        r["merges"] = int(float(r["merges"]))
        r["splits"] = int(float(r["splits"]))
    return rows


def f1_matrix(rows):
    """Table A: per-cell F1 as a detector x granularity grid with SAM1/2/3 columns."""
    print("=== Table A: F1 matrix (3 decimals) ===")
    for det in ["YOLOv5", "SAHI", "YOLO11"]:
        for gran in GRAN_ORDER:
            cells = {r["sam"]: r["F1"] for r in rows if r["det"] == det and r["gran"] == gran}
            vals = " & ".join(f"{cells[s]:.3f}" for s in SAM_ORDER)
            label = det if gran == "full frame" else ""
            print(f"{label:7}& {gran:11}& {vals} \\\\")
    seg = next(r["F1"] for r in rows if r["det"] == "yolo11-seg")
    print(f"yolo11-seg (direct): {seg:.3f}")


def granularity_table(rows):
    """Table B: F1 / matched-mask IoU / boundary IoU / merges, averaged over the 9 cells
    (3 detectors x 3 SAM versions) at each granularity."""
    print("\n=== Table B: granularity effect (averaged over 9 cells each) ===")
    print(f"{'gran':11} F1     IoU_m  bIoU   merges")
    for gran in GRAN_ORDER:
        g = [r for r in rows if r["gran"] == gran]
        n = len(g)
        f1 = sum(r["F1"] for r in g) / n
        iou = sum(r["mean_iou"] for r in g) / n
        bio = sum(r["boundary_iou"] for r in g) / n
        mrg = sum(r["merges"] for r in g) / n
        print(f"{gran:11} {f1:.3f}  {iou:.3f}  {bio:.3f}  {mrg:.0f}   (n={n})")


def appendix_table(rows):
    """Full 28-row appendix table, ranked by F1. Granularity abbreviated ff/pt/ph."""
    abbr = {"full frame": "ff", "per tile": "pt", "per head": "ph"}
    print("\n=== Appendix: full 28-row table (ranked by F1) ===")
    for r in sorted(rows, key=lambda r: -r["F1"]):
        g = abbr.get(r["gran"], "---")
        s = {"SAM1": "S1", "SAM2": "S2", "SAM3": "S3"}.get(r["sam"], "---")
        print(f"{r['det']} & {g} & {s} & {r['F1']:.3f} & {r['recall']:.3f} & "
              f"{r['precision']:.3f} & {r['mean_iou']:.3f} & "
              f"{r['boundary_iou']:.3f} & {r['boundary_iou_dyn']:.3f} & "
              f"{r['boundary_f']:.3f} & {r['boundary_f_dyn']:.3f} & {r['merges']} & {r['splits']} \\\\")


def prose_numbers(rows):
    """Prints the specific cell numbers quoted in the Results prose, so each can be checked."""
    print("\n=== Prose numbers (as quoted in Results) ===")
    def cell(det, gran, sam):
        return next(r for r in rows if r["det"] == det and r["gran"] == gran and r["sam"] == sam)

    print("Detector best cells (all peak at per head):")
    for det in ["YOLO11", "SAHI", "YOLOv5"]:
        best = max((r for r in rows if r["det"] == det and r["gran"]), key=lambda r: r["F1"])
        print(f"  {det:7} best = {best['gran']}/{best['sam']}  F1 {best['F1']:.3f}")
    y = cell("YOLO11", "per head", "SAM2"); s = cell("SAHI", "per head", "SAM2")
    print(f"  per-head SAM2 precision: YOLO11 {y['precision']:.3f} vs SAHI {s['precision']:.3f}")
    print(f"  per-head SAM2 recall:    YOLO11 {y['recall']:.3f} vs SAHI {s['recall']:.3f}")
    print(f"  yolo11-seg F1: {next(r['F1'] for r in rows if r['det']=='yolo11-seg'):.3f}")

    print("SAM flip:")
    print(f"  YOLO11 full frame: SAM1 {cell('YOLO11','full frame','SAM1')['F1']:.3f} vs SAM2 {cell('YOLO11','full frame','SAM2')['F1']:.3f}")
    print(f"  YOLO11 per head:   SAM2 {cell('YOLO11','per head','SAM2')['F1']:.3f} vs SAM1 {cell('YOLO11','per head','SAM1')['F1']:.3f}")
    s2 = cell("YOLO11", "per head", "SAM2"); s3 = cell("YOLO11", "per head", "SAM3")
    print(f"  per-head tie: SAM3 {s3['F1']:.3f} vs SAM2 {s2['F1']:.3f}; "
          f"IoU_m {s3['mean_iou']:.3f} vs {s2['mean_iou']:.3f}; bIoU {s3['boundary_iou']:.3f} vs {s2['boundary_iou']:.3f}")
    print(f"Best cell (YOLO11 per head): SAM2 F1 {s2['F1']:.3f}, SAM3 F1 {s3['F1']:.3f}, "
          f"SAM2 P {s2['precision']:.3f}, IoU_m {s2['mean_iou']:.3f}, recall {s2['recall']:.3f}")
    print(f"Max F1 over all cells: {max(r['F1'] for r in rows):.3f}")


def threshold_table():
    """F1 (and recall) at IoU 0.25 / 0.5 / 0.75 for the winning detector (YOLOv5 @4032)'s best cell
    at each granularity, from each cell's f1_curve_mean. Best SAM per granularity is SAM3 (see the
    full-res grid). Shows how the scores hold up under a looser and a stricter match."""
    print("\n=== Threshold sensitivity: YOLOv5 @4032 best cell per granularity ===")
    rows = [("full frame", "e2_ff_sam3"), ("per tile", "e2_pt_sam3"), ("per head", "e2_ph_sam3")]
    print(f"{'granularity':11}  F1@.25  F1@.5  F1@.75   R@.25  R@.5  R@.75")
    for label, exp in rows:
        f = f"results/mask_generation/phone/evaluation/yolo_sam_v1/masks_instance/{exp}/eval_masks_instance.json"
        if not os.path.exists(f):
            print(f"{label:11}  MISSING ({f})"); continue
        curve = {pt["iou"]: pt for pt in json.load(open(f))["aggregate_curve_hist"]["f1_curve_mean"]}
        f1 = [curve[t]["f1"] for t in (0.25, 0.5, 0.75)]
        rc = [curve[t]["recall"] for t in (0.25, 0.5, 0.75)]
        print(f"{label:11}  {f1[0]:.3f}  {f1[1]:.3f}  {f1[2]:.3f}    {rc[0]:.3f} {rc[1]:.3f} {rc[2]:.3f}")


# YOLOv5 @4032 SAM3 has no sam_perf.json (that full-res run saved only the eval, not timing), so its three
# values are read from the Euler log slurm_logs/e2_sam3_fullres_11416384.out (the 6-GT-image "Average Time
# Per Image" per granularity, same basis as every other cell). Listed here as constants with that provenance.
SAM3_4032 = {"full frame": 19.22, "per tile": 13.35, "per head": 150.61}


def runtime_table():
    """SAM-phase seconds per image, averaged over the four detector configurations (YOLOv5@1280,
    YOLOv5@4032, SAHI, YOLO11), as a granularity x SAM-version grid. Reads each config's sam_perf.json
    (avg_sec_per_image): gt_* are the three original-resolution configs (one dir per detector), e2_* is the
    full-res YOLOv5@4032 config. The @4032 SAM3 cell has no sam_perf.json so it comes from SAM3_4032 above."""
    print("\n=== Runtime: SAM seconds per image (granularity x SAM version, 4 detector configs) ===")
    times = collections.defaultdict(list)
    for f in glob.glob("results/mask_generation/phone/**/sam_perf.json", recursive=True):
        d = json.load(open(f))
        exp = os.path.basename(os.path.dirname(f))
        # gt_* = the three original-resolution detector configs; e2_* = the full-res YOLOv5@4032 config
        # (its granularity is tagged _pt_/_ph_ rather than _tile_/_head_).
        if exp.startswith("gt_"):
            gran = "full frame" if "_ff_" in exp else "per tile" if "_tile_" in exp else "per head" if "_head_" in exp else None
        elif exp.startswith("e2_"):
            gran = "full frame" if "_ff_" in exp else "per tile" if "_pt_" in exp else "per head" if "_ph_" in exp else None
        else:
            continue
        if gran is None or "avg_sec_per_image" not in d:
            continue
        sam = {"sam1": "SAM1", "sam2": "SAM2", "sam3": "SAM3"}.get(d.get("backend"))
        if sam:
            times[(gran, sam)].append(d["avg_sec_per_image"])
    # add the @4032 SAM3 cell (log-sourced, no sam_perf.json) so SAM3 also averages over four configs
    for gran, v in SAM3_4032.items():
        times[(gran, "SAM3")].append(v)
    print(f"{'gran':11} SAM1  SAM2  SAM3   (n configs)")
    for gran in GRAN_ORDER:
        cells, ns = [], []
        for s in SAM_ORDER:
            v = times.get((gran, s), [])
            cells.append(f"{sum(v)/len(v):4.0f}" if v else "  --")
            ns.append(len(v))
        print(f"{gran:11} " + "  ".join(cells) + f"    (n={ns})")


def main():
    if not os.path.exists(CSV):
        raise SystemExit(f"missing {CSV} --- run the grid + aggregate_maskgen_grid.py first")
    rows = load()
    assert len(rows) == 28, f"expected 28 cells, got {len(rows)}"
    f1_matrix(rows)
    granularity_table(rows)
    appendix_table(rows)
    prose_numbers(rows)
    threshold_table()
    runtime_table()


if __name__ == "__main__":
    main()
