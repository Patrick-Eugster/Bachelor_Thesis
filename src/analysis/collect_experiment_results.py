"""Harvest every experiment run into one master table — the recovery/tracking layer for EXPERIMENTS.md.

Walks the results/ tree and, for each run, reads the SAVED config.yaml as the source of truth (NOT the
folder name — a folder can be mislabeled, e.g. test_absgrad_v2/ has absgrad:false), plus results.json
(recon metrics), seg_summary.json (3D head count) and eval_2d/metrics_2d.json (2D seg metrics). Emits
docs/analysis_results/experiments/RESULTS.csv (+ a short RESULTS.md) so no experiment value is ever lost
to a wiped results/ tree or a misleading folder name.

Read-only. Run after any batch:  python src/analysis/collect_experiment_results.py
"""
import argparse
import csv
import glob
import json
import os
import shutil
import statistics

import yaml

# columns in a fixed order so the CSV is stable across runs
COLS = [
    "kind", "dataset", "plot", "field", "date", "arm", "experiment", "iteration",
    "renderer", "git_commit", "git_dirty",
    "absgrad", "use_principal_point", "use_agisoft_sfm", "iterations_cfg", "resolution",
    "sh_degree", "densify_grad_threshold",
    "PSNR", "SSIM", "LPIPS",
    "PSNR_roi", "SSIM_roi", "LPIPS_roi", "sharpness_ratio_roi",
    "PSNR_markers", "SSIM_markers", "LPIPS_markers",
    "seg_name", "wheat_heads_found",
    "eval2d_iou", "eval2d_f1", "eval2d_precision", "eval2d_recall", "eval2d_n_imgs",
    "path",
]


def _load_yaml(path):
    """Load a config.yaml (OmegaConf dump + our appended runtime keys). {} on any problem."""
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _cfg_get(cfg, *keys, default=None):
    """Fetch a key that may sit at top level OR inside the reconstruction: block (Hydra nesting)."""
    for k in keys:
        if k in cfg:
            return cfg[k]
    rec = cfg.get("reconstruction", {}) if isinstance(cfg.get("reconstruction"), dict) else {}
    for k in keys:
        if k in rec:
            return rec[k]
    return default


def _parse_path(exp_dir, results_root):
    """Pull dataset / plot / field / date / arm / experiment_name out of the result path.
    FIP:   results/reconstruction/fip/plot_467/vanilla_3dgs/<exp>
    Phone: results/reconstruction/phone/field_A/20250715/[agisoft/]vanilla_3dgs/<exp>"""
    rel = os.path.relpath(exp_dir, os.path.join(results_root, "reconstruction"))
    parts = rel.split(os.sep)
    info = {"dataset": parts[0] if parts else "", "plot": "", "field": "", "date": "",
            "arm": "agisoft" if "agisoft" in parts else "colmap", "experiment": os.path.basename(exp_dir)}
    if "vanilla_3dgs" in parts:
        vi = parts.index("vanilla_3dgs")
        mid = [p for p in parts[1:vi] if p != "agisoft"]   # the plot / field+date between dataset and tree
        info["experiment"] = parts[vi + 1] if len(parts) > vi + 1 else info["experiment"]
        if info["dataset"] == "fip":
            info["plot"] = mid[0] if mid else ""
        else:
            info["field"] = mid[0] if len(mid) > 0 else ""
            info["date"] = mid[1] if len(mid) > 1 else ""
            info["plot"] = "/".join(mid)
    return info


def _recon_row(exp_dir, results_root):
    """Build one row per (experiment × iteration) from config.yaml + results.json. [] if no results."""
    cfg = _load_yaml(os.path.join(exp_dir, "config.yaml"))
    res_path = os.path.join(exp_dir, "results.json")
    if not os.path.isfile(res_path):
        return []
    try:
        results = json.load(open(res_path))
    except Exception:
        return []
    base = _parse_path(exp_dir, results_root)
    base.update({
        "kind": "recon",
        "renderer": cfg.get("wheat_renderer"),
        "git_commit": cfg.get("git_commit"),
        "git_dirty": cfg.get("git_dirty"),
        "absgrad": _cfg_get(cfg, "absgrad"),
        "use_principal_point": _cfg_get(cfg, "use_principal_point"),
        "use_agisoft_sfm": cfg.get("use_agisoft_sfm"),
        "iterations_cfg": _cfg_get(cfg, "iterations"),
        "resolution": _cfg_get(cfg, "resolution"),
        "sh_degree": _cfg_get(cfg, "sh_degree"),
        "densify_grad_threshold": _cfg_get(cfg, "densify_grad_threshold"),
        "path": exp_dir,
    })
    rows = []
    for it_key, m in results.items():           # ours_15000, ours_30000, ...
        if not isinstance(m, dict):
            continue
        row = dict(base)
        row["iteration"] = it_key.replace("ours_", "")
        row["PSNR"], row["SSIM"], row["LPIPS"] = m.get("PSNR"), m.get("SSIM"), m.get("LPIPS")
        roi = m.get("roi", {}) if isinstance(m.get("roi"), dict) else {}
        row["PSNR_roi"], row["SSIM_roi"], row["LPIPS_roi"] = roi.get("PSNR"), roi.get("SSIM"), roi.get("LPIPS")
        row["sharpness_ratio_roi"] = roi.get("sharpness_ratio")
        mk = m.get("markers", {}) if isinstance(m.get("markers"), dict) else {}
        row["PSNR_markers"], row["SSIM_markers"], row["LPIPS_markers"] = mk.get("PSNR"), mk.get("SSIM"), mk.get("LPIPS")
        rows.append(row)
    return rows


def _seg_rows(results_root):
    """One row per 3D-seg run (seg_summary.json), with the parent recon context + eval_2d aggregate."""
    rows = []
    for summ in glob.glob(os.path.join(results_root, "reconstruction", "**", "seg_summary.json"), recursive=True):
        seg_dir = os.path.dirname(summ)                       # .../vanilla_3dgs/<exp>/segmentation_3d/<seg>
        exp_dir = os.path.dirname(os.path.dirname(seg_dir))   # .../vanilla_3dgs/<exp>
        cfg = _load_yaml(os.path.join(exp_dir, "config.yaml"))
        base = _parse_path(exp_dir, results_root)
        try:
            heads = json.load(open(summ)).get("wheat_heads_found")
        except Exception:
            heads = None
        row = {c: None for c in COLS}
        row.update({
            "kind": "seg", **{k: base[k] for k in ("dataset", "plot", "field", "date", "arm", "experiment")},
            "renderer": cfg.get("wheat_renderer"), "git_commit": cfg.get("git_commit"),
            "use_agisoft_sfm": cfg.get("use_agisoft_sfm"),
            "seg_name": os.path.basename(seg_dir), "wheat_heads_found": heads, "path": seg_dir,
        })
        # eval_2d aggregate (mean over the per-camera list), if present
        ev = os.path.join(seg_dir, "eval_2d", "metrics_2d.json")
        if os.path.isfile(ev):
            try:
                per = json.load(open(ev))
                per = [p for p in per if isinstance(p, dict)]
                if per:
                    row["eval2d_iou"] = round(statistics.mean(p["iou"] for p in per if "iou" in p), 4)
                    row["eval2d_f1"] = round(statistics.mean(p["f1"] for p in per if "f1" in p), 4)
                    row["eval2d_precision"] = round(statistics.mean(p["precision"] for p in per if "precision" in p), 4)
                    row["eval2d_recall"] = round(statistics.mean(p["recall"] for p in per if "recall" in p), 4)
                    row["eval2d_n_imgs"] = len(per)
            except Exception:
                pass
        rows.append(row)
    return rows


def _copy_if(src, dst_dir):
    """Copy src into dst_dir if it exists (make the dir on demand). Returns True on copy."""
    if src and os.path.isfile(src):
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
        return True
    return False


def _copy_sample_pngs(src_dir, dst_dir, n, max_px=640):
    """Copy the first n sorted images as small downscaled JPEG THUMBNAILS (visual proof only — full-res
    renders would bloat the git-tracked archive to GBs). ~50-100 KB each instead of several MB."""
    if not os.path.isdir(src_dir):
        return 0
    from PIL import Image
    imgs = sorted(f for f in os.listdir(src_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))[:n]
    if imgs:
        os.makedirs(dst_dir, exist_ok=True)
    for p in imgs:
        try:
            im = Image.open(os.path.join(src_dir, p)).convert("RGB")
            im.thumbnail((max_px, max_px))
            im.save(os.path.join(dst_dir, os.path.splitext(p)[0] + ".jpg"), quality=85)
        except Exception:
            pass
    return len(imgs)


def _input_split_check(info, results_root):
    """Best-effort path to the arm's split_check.json (lives in input_plots/, not results/)."""
    base = os.path.join("input_plots", info["dataset"])
    if info["dataset"] == "fip":
        base = os.path.join(base, info["plot"])
    else:
        base = os.path.join(base, info["field"], info["date"])
    if info["arm"] == "agisoft":
        base = os.path.join(base, "agisoft")
    return os.path.join(base, "logs", "split_check.json")


def _archive_dst(archive_root, info):
    """The archive folder for a run: archive/<dataset>/<plot|field/date>/<arm>/<experiment>/."""
    loc = info["plot"] or f"{info['field']}/{info['date']}"
    return os.path.join(archive_root, info["dataset"], loc, info["arm"], info["experiment"])


def archive_run(exp_dir, info, archive_root, n_png=0):
    """Copy a recon run's small, durable PROOF files out of the gitignored results/ tree into docs/,
    so the evidence survives a wipe. JSON/YAML only by default (tiny); render+gt PNG thumbnails only if
    n_png>0 (opt-in — full sets would bloat the git-tracked archive)."""
    dst = _archive_dst(archive_root, info)
    os.makedirs(dst, exist_ok=True)
    for name in ("config.yaml", "results.json", "run_report.json", "run_report.txt"):
        _copy_if(os.path.join(exp_dir, name), dst)
    _copy_if(_input_split_check(info, "results"), dst)   # the split proof (from input_plots/)
    if n_png > 0:
        test_dir = os.path.join(exp_dir, "test")
        if os.path.isdir(test_dir):
            iters = sorted(d for d in os.listdir(test_dir) if d.startswith("ours_"))
            if iters:
                top = os.path.join(test_dir, iters[-1])
                _copy_sample_pngs(os.path.join(top, "renders"), os.path.join(dst, "sample_renders"), n_png)
                _copy_sample_pngs(os.path.join(top, "gt"), os.path.join(dst, "sample_gt"), n_png)
    return dst


def archive_seg(seg_dir, info, archive_root):
    """Copy a 3D-seg run's proof (seg_summary.json + eval_2d/metrics_2d.json) into the run's archive,
    under segmentation_3d/<seg_name>/. JSON only."""
    dst = os.path.join(_archive_dst(archive_root, info), "segmentation_3d", os.path.basename(seg_dir))
    os.makedirs(dst, exist_ok=True)
    _copy_if(os.path.join(seg_dir, "seg_summary.json"), dst)
    _copy_if(os.path.join(seg_dir, "eval_2d", "metrics_2d.json"), dst)
    return dst


def main():
    ap = argparse.ArgumentParser(description="Harvest all experiment runs into one master table.")
    ap.add_argument("--results-root", default="results", help="root of the results/ tree")
    ap.add_argument("--out-dir", default="docs/analysis_results/experiments", help="where RESULTS.{csv,md} go")
    ap.add_argument("--archive", action="store_true",
                    help="also copy each run's proof JSON/YAML (config/results/report/split_check + seg_summary/"
                         "metrics_2d) into <out-dir>/archive/ so they survive the gitignored results/ tree being wiped")
    ap.add_argument("--archive-pngs", type=int, default=0,
                    help="opt-in: also archive N render+gt thumbnail JPEGs per run (0 = JSON only, the default)")
    args = ap.parse_args()

    # recon rows: every folder holding a results.json under a vanilla_3dgs/ tree
    recon_rows = []
    for res in glob.glob(os.path.join(args.results_root, "reconstruction", "**", "results.json"), recursive=True):
        exp_dir = os.path.dirname(res)
        if "segmentation_3d" in exp_dir:      # skip stray results.json inside seg subfolders
            continue
        recon_rows.extend(_recon_row(exp_dir, args.results_root))
    seg_rows = _seg_rows(args.results_root)
    rows = recon_rows + seg_rows

    # stable sort so diffs are readable
    rows.sort(key=lambda r: (r.get("kind") or "", r.get("dataset") or "", r.get("plot") or "",
                             r.get("arm") or "", r.get("experiment") or "", str(r.get("iteration") or "")))

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "RESULTS.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in COLS})

    # a short human-readable recon summary (the numbers you paste back into EXPERIMENTS.md)
    md_path = os.path.join(args.out_dir, "RESULTS.md")
    with open(md_path, "w") as f:
        f.write("# Experiment results (auto-harvested)\n\n")
        f.write(f"Source of truth = each run's `config.yaml`. {len(recon_rows)} recon rows, "
                f"{len(seg_rows)} seg rows. Regenerate: `python src/analysis/collect_experiment_results.py`\n\n")
        f.write("## Reconstruction (whole-image)\n\n")
        f.write("| dataset | plot/field/date | arm | experiment | iter | renderer | PSNR | SSIM | LPIPS | commit |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in recon_rows:
            loc = r["plot"] or f"{r['field']}/{r['date']}"
            psnr = f"{r['PSNR']:.2f}" if isinstance(r["PSNR"], (int, float)) else "—"
            ssim = f"{r['SSIM']:.3f}" if isinstance(r["SSIM"], (int, float)) else "—"
            lp = f"{r['LPIPS']:.3f}" if isinstance(r["LPIPS"], (int, float)) else "—"
            f.write(f"| {r['dataset']} | {loc} | {r['arm']} | {r['experiment']} | {r['iteration']} "
                    f"| {r['renderer'] or '?'} | {psnr} | {ssim} | {lp} | {r['git_commit'] or '?'} |\n")
        if seg_rows:
            f.write("\n## 3D segmentation\n\n")
            f.write("| dataset | plot/field/date | arm | experiment | seg | heads | eval2d IoU | F1 | commit |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for r in seg_rows:
                loc = r["plot"] or f"{r['field']}/{r['date']}"
                f.write(f"| {r['dataset']} | {loc} | {r['arm']} | {r['experiment']} | {r['seg_name']} "
                        f"| {r['wheat_heads_found']} | {r['eval2d_iou']} | {r['eval2d_f1']} | {r['git_commit'] or '?'} |\n")

    print(f"harvested {len(recon_rows)} recon + {len(seg_rows)} seg rows")
    print(f"  → {csv_path}")
    print(f"  → {md_path}")

    # optional: copy each run's proof files into a durable, git-tracked archive
    if args.archive:
        archive_root = os.path.join(args.out_dir, "archive")
        seen = set()
        n = 0
        for res in glob.glob(os.path.join(args.results_root, "reconstruction", "**", "results.json"), recursive=True):
            exp_dir = os.path.dirname(res)
            if "segmentation_3d" in exp_dir or exp_dir in seen:
                continue
            seen.add(exp_dir)
            archive_run(exp_dir, _parse_path(exp_dir, args.results_root), archive_root, args.archive_pngs)
            n += 1
        # seg runs: their JSONs live under a parent recon exp that may or may not have a results.json
        s = 0
        for summ in glob.glob(os.path.join(args.results_root, "reconstruction", "**", "seg_summary.json"), recursive=True):
            seg_dir = os.path.dirname(summ)
            exp_dir = os.path.dirname(os.path.dirname(seg_dir))
            archive_seg(seg_dir, _parse_path(exp_dir, args.results_root), archive_root)
            s += 1
        png_note = f", {args.archive_pngs} thumb(s)/run" if args.archive_pngs else " (JSON only)"
        print(f"archived {n} recon + {s} seg runs{png_note} → {archive_root}/  (commit the docs/ submodule to preserve)")


if __name__ == "__main__":
    main()
