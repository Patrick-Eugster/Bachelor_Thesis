"""
Compare the two Euler benchmark runs on all 7 FIP plots:
  - test_diffgs_full  (diff-gaussian engine + use_principal_point)
  - test_gsplat_full  (gsplat engine + use_principal_point) + Test A renderer_compare.json

Reads, per plot:
  - results.json            -> PSNR / SSIM / LPIPS (render quality)  [both runs]
  - renderer_compare.json   -> engine-vs-engine alpha IoU + RGB PSNR (the tan-residual)  [gsplat run]
  - segmentation_3d/run_1/results.csv -> #heads + total mask matches (seg behaviour)  [both runs]

Prints three comparison tables + averages, and writes docs/analysis_results/gsplat_vs_diffgs.json.
Read-only. Run: python src/analysis/compare_gsplat_runs.py
"""

import os
import csv
import json
import numpy as np

REPO = "/workspace"
ROOT = os.path.join(REPO, "results", "reconstruction", "fip")
OUT_JSON = os.path.join(REPO, "docs", "analysis_results", "gsplat_vs_diffgs.json")
PLOTS = [f"plot_46{i}" for i in range(1, 8)]
DIFFGS, GSPLAT = "test_diffgs_full", "test_gsplat_full"


def read_quality(exp_dir):
    """Return (PSNR, SSIM, LPIPS) from results.json's ours_30000 block, or None."""
    p = os.path.join(exp_dir, "results.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p)).get("ours_30000", {})
    return d.get("PSNR"), d.get("SSIM"), d.get("LPIPS")


def read_testA(gsplat_dir):
    """Return (alpha_iou, rgb_psnr, alpha_disagree_px) from renderer_compare.json test split, or None."""
    p = os.path.join(gsplat_dir, "renderer_compare.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p)).get("test", {}).get("mean", {})
    return m.get("alpha_iou"), m.get("rgb_psnr"), m.get("alpha_disagree_px")


def read_seg(exp_dir):
    """Return (n_heads_3d, total_matches) from segmentation_3d/run_1/results.csv, or None.
    n_heads_3d = number of assigned 3D head IDs (data rows); total_matches = sum of num_matches."""
    p = os.path.join(exp_dir, "segmentation_3d", "run_1", "results.csv")
    if not os.path.exists(p):
        return None
    n, matches = 0, 0
    with open(p) as f:
        for row in csv.DictReader(f):
            n += 1
            try:
                matches += int(row.get("num_matches", 0))
            except ValueError:
                pass
    return n, matches


def read_seg2d(exp_dir):
    """Return GT-based 2D seg metrics from segmentation_3d/run_1/eval_2d/metrics_2d.json, or None.
    Returns (iou, f1, pred_head_count, gt_head_count, count_error_ratio) of the GT view."""
    p = os.path.join(exp_dir, "segmentation_3d", "run_1", "eval_2d", "metrics_2d.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if not d:
        return None
    m = d[0]   # one GT view per plot
    return m.get("iou"), m.get("f1"), m.get("pred_head_count"), m.get("gt_head_count"), m.get("count_error_ratio")


def main():
    rows = []
    for plot in PLOTS:
        base = os.path.join(ROOT, plot, "vanilla_3dgs")
        rows.append({
            "plot": plot,
            "diffgs_quality": read_quality(os.path.join(base, DIFFGS)),
            "gsplat_quality": read_quality(os.path.join(base, GSPLAT)),
            "testA": read_testA(os.path.join(base, GSPLAT)),
            "diffgs_seg": read_seg(os.path.join(base, DIFFGS)),
            "gsplat_seg": read_seg(os.path.join(base, GSPLAT)),
            "diffgs_seg2d": read_seg2d(os.path.join(base, DIFFGS)),
            "gsplat_seg2d": read_seg2d(os.path.join(base, GSPLAT)),
        })

    # ── Table 1: render quality ──
    print("\n" + "=" * 92)
    print("QUALITY  (PSNR / SSIM / LPIPS, test split, 30k)        diffgs            gsplat        ΔPSNR")
    print("-" * 92)
    dP, gP, dS, gS, dL, gL = ([] for _ in range(6))
    for r in rows:
        d, g = r["diffgs_quality"], r["gsplat_quality"]
        if not (d and g):
            print(f"{r['plot']:<10} (missing)"); continue
        dP.append(d[0]); gP.append(g[0]); dS.append(d[1]); gS.append(g[1]); dL.append(d[2]); gL.append(g[2])
        print(f"{r['plot']:<10} {d[0]:6.2f}/{d[1]:.3f}/{d[2]:.3f}   {g[0]:6.2f}/{g[1]:.3f}/{g[2]:.3f}   {g[0]-d[0]:+5.2f}")
    print("-" * 92)
    if dP:
        print(f"{'AVG':<10} {np.mean(dP):6.2f}/{np.mean(dS):.3f}/{np.mean(dL):.3f}   "
              f"{np.mean(gP):6.2f}/{np.mean(gS):.3f}/{np.mean(gL):.3f}   {np.mean(gP)-np.mean(dP):+5.2f}")

    # ── Table 2: Test A (engine agreement / tan-residual) ──
    print("\n" + "=" * 92)
    print("TEST A  gsplat-vs-diffgs render agreement      alpha_IoU      RGB_PSNR(dB)   disagree_px")
    print("-" * 92)
    ious, psnrs = [], []
    for r in rows:
        t = r["testA"]
        if not t:
            print(f"{r['plot']:<10} (missing)"); continue
        ious.append(t[0]); psnrs.append(t[1])
        print(f"{r['plot']:<10}                              {t[0]:.6f}     {t[1]:8.2f}     {t[2]:.0f}")
    print("-" * 92)
    if ious:
        print(f"{'AVG':<10}                              {np.mean(ious):.6f}     {np.mean(psnrs):8.2f}")

    # ── Table 3: segmentation accuracy vs GT (eval_2d, one labelled view per plot) ──
    print("\n" + "=" * 100)
    print("SEG vs GT        diffgs: IoU / F1  pred/gt        gsplat: IoU / F1  pred/gt       ΔIoU")
    print("-" * 100)
    dI, gI, dF, gF = ([] for _ in range(4))
    for r in rows:
        d, g = r["diffgs_seg2d"], r["gsplat_seg2d"]
        if not (d and g):
            print(f"{r['plot']:<10} (missing seg2d)"); continue
        dI.append(d[0]); gI.append(g[0]); dF.append(d[1]); gF.append(g[1])
        print(f"{r['plot']:<10}   {d[0]:.3f} / {d[1]:.3f}  {d[2]:4d}/{d[3]:<4d}     "
              f"{g[0]:.3f} / {g[1]:.3f}  {g[2]:4d}/{g[3]:<4d}    {g[0]-d[0]:+.3f}")
    print("-" * 100)
    if dI:
        print(f"{'AVG':<10}   {np.mean(dI):.3f} / {np.mean(dF):.3f}            "
              f"{np.mean(gI):.3f} / {np.mean(gF):.3f}           {np.mean(gI)-np.mean(dI):+.3f}")

    # ── Table 4: total 3D heads assigned (segmentation_3d/run_1/results.csv) ──
    print("\n" + "=" * 100)
    print("3D HEADS         diffgs: #heads / matches        gsplat: #heads / matches      Δheads")
    print("-" * 100)
    for r in rows:
        d, g = r["diffgs_seg"], r["gsplat_seg"]
        if not (d and g):
            print(f"{r['plot']:<10} (missing seg)"); continue
        print(f"{r['plot']:<10}      {d[0]:5d} / {d[1]:6d}               {g[0]:5d} / {g[1]:6d}           {g[0]-d[0]:+d}")
    print("=" * 100 + "\n")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(rows, open(OUT_JSON, "w"), indent=2)
    print(f"Wrote -> {OUT_JSON}\n")


if __name__ == "__main__":
    main()
