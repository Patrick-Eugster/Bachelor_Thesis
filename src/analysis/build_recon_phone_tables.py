"""Collects the phone 3DGS reconstruction metrics into per-region averages and
emits LaTeX tables for the thesis. Reads results.json for the four evaluation
sessions (A/0627, D/0627, A/0715, D/0715) and averages PSNR/SSIM/LPIPS plus the
Laplacian sharpness ratio over them, for every region (whole/inner/roi/markers).

Two experiments:
  camera  : 4 camera sources at 15k default  (pinhole/opencv/radial/agisoft)
  dense   : OPENCV under 2 densifications x 2 training lengths

Run:  python src/analysis/build_recon_phone_tables.py
"""
import json
from pathlib import Path

RESULTS = Path("results/reconstruction/phone")
SESSIONS = [("field_A", "20250627"), ("field_D", "20250627"),
            ("field_A", "20250715"), ("field_D", "20250715")]
REGIONS = ["whole", "inner", "roi", "markers"]
METRICS = ["PSNR", "SSIM", "LPIPS", "sharpness_ratio"]

# camera sources: (label, variant-subdir, exp-folder). pinhole = default variant (no subdir).
CAMERAS = [("PINHOLE", "", "baseline"), ("OPENCV", "opencv", "baseline"),
           ("RADIAL", "radial", "baseline"), ("Agisoft", "agisoft", "baseline")]
# densification grid on OPENCV: (label, exp-folder, iter-key)
DENSE = [("15k, default", "baseline", "ours_15000"),
         ("15k, AbsGS", "absgrad", "ours_15000"),
         ("30k, default", "dense17k_noabsgrad", "ours_30000"),
         ("30k, AbsGS", "dense17k", "ours_30000")]


def load_scene(field, date, variant, exp):
    """Reads one results.json and returns the single ours_* iteration block."""
    sub = f"{variant}/" if variant else ""
    p = RESULTS / field / date / f"{sub}vanilla_3dgs/{exp}/results.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return next(iter(d.values()))  # the sole ours_XXXXX block


def region_metric(block, region, metric):
    """Pulls one metric for one region from a results block. 'whole' lives at the
    top level; the other regions are nested sub-dicts."""
    src = block if region == "whole" else block.get(region)
    if src is None:
        return None
    return src.get(metric)


def average(cells):
    """Mean of the non-None values, or None if nothing was found."""
    vals = [c for c in cells if c is not None]
    return sum(vals) / len(vals) if vals else None


def collect(rows):
    """rows = list of (label, variant, exp, iterkey). Returns
    {label: {region: {metric: avg}}} averaged over the four sessions."""
    out = {}
    for label, variant, exp, _ in rows:
        out[label] = {}
        blocks = [load_scene(f, d, variant, exp) for f, d in SESSIONS]
        for region in REGIONS:
            out[label][region] = {}
            for metric in METRICS:
                cells = [region_metric(b, region, metric) if b else None for b in blocks]
                out[label][region][metric] = average(cells)
    return out


def fmt(v, metric):
    """Formats a metric value with the right precision, or a dash if missing."""
    if v is None:
        return "--"
    if metric == "PSNR":
        return f"{v:.2f}"
    return f"{v:.3f}"


def print_plain(title, data):
    """Console dump so we can eyeball every number before it goes in the thesis."""
    print(f"\n### {title}")
    for label in data:
        print(f"  {label}")
        for region in REGIONS:
            cells = "  ".join(f"{m}={fmt(data[label][region][m], m)}" for m in METRICS)
            print(f"    {region:8s} {cells}")


REGION_TITLE = {"whole": "Whole image", "inner": "Inner crop",
                "roi": "Plot ROI", "markers": "Coded markers"}
REGION_LABEL = {"whole": "whole", "inner": "inner", "roi": "roi", "markers": "markers"}
# main-text row order: 4 camera sources (15k default) then the 3 remaining densification arms.
CAM_ROWS = ["PINHOLE", "OPENCV", "RADIAL", "Agisoft"]
DENSE_ROWS = ["15k, default", "15k, AbsGS", "30k, default", "30k, AbsGS"]
# pretty row labels for the densification arms (internal keys stay short)
DENSE_DISPLAY = {"15k, default": "15{,}000, default", "15k, AbsGS": "15{,}000, AbsGS",
                 "30k, default": "30{,}000, default", "30k, AbsGS": "30{,}000, AbsGS"}
# which column each metric goes in, and whether higher (True) or lower (False) is better.
# sharpness deliberately gets no "best" bold: a higher render/GT ratio is not simply better
# on phone (it trades against PSNR), so bolding it would mislead.
COL = [("PSNR", "PSNR $\\uparrow$", True), ("SSIM", "SSIM $\\uparrow$", True),
       ("LPIPS", "LPIPS $\\downarrow$", False), ("sharpness_ratio", "sharpness", None)]


def best_value(data, region, metric, higher):
    """Best value of one metric down a region table (for bolding). Returns None if
    the column should not be bolded (sharpness)."""
    if higher is None:
        return None
    vals = [data[a][region][metric] for a in data if data[a][region][metric] is not None]
    return max(vals) if higher else min(vals)


def latex_cell(v, metric, best):
    """One formatted table cell, bolded if it is the column best."""
    s = fmt(v, metric)
    if best is not None and v is not None and abs(v - best) < 1e-9:
        return f"\\textbf{{{s}}}"
    return s


# Captions are written by hand in the thesis (main.tex), never generated here, so a
# machine-written caption can never leak into the prose. We emit an empty caption as a
# placeholder; the real one lives in the thesis next to the pasted table.
def caption_placeholder():
    """Empty caption line. The real caption is authored in main.tex, not generated."""
    return "\\caption{}  % caption authored in main.tex"


def emit_split_table(region, data, row_order, display, label):
    """LaTeX for one self-contained region table: the given arms (one experiment),
    all metrics, best-per-column bolded (sharpness never bolded)."""
    bests = {m: best_value({a: data[a] for a in row_order}, region, m, hi)
             for m, _, hi in COL}
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{l cccc}")
    lines.append("\\hline")
    lines.append(" & " + " & ".join(h for _, h, _ in COL) + " \\\\")
    lines.append("\\hline")
    for a in row_order:
        cells = " & ".join(latex_cell(data[a][region][m], m, bests[m]) for m, _, _ in COL)
        lines.append(f"{display.get(a, a)} & {cells} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(caption_placeholder())
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def emit_combined_table(data, row_order, display, label):
    """One table stacking all four regions as labelled blocks, separated by rules.
    Best-per-column is bolded within each region block (cross-region bests are
    meaningless: markers always beat wheat on PSNR)."""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{l cccc}")
    lines.append("\\hline")
    lines.append(" & " + " & ".join(h for _, h, _ in COL) + " \\\\")
    for region in REGIONS:
        bests = {m: best_value({a: data[a] for a in row_order}, region, m, hi)
                 for m, _, hi in COL}
        lines.append("\\hline")
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textbf{{{REGION_TITLE[region]}}}}} \\\\")
        for a in row_order:
            cells = " & ".join(latex_cell(data[a][region][m], m, bests[m]) for m, _, _ in COL)
            lines.append(f"{display.get(a, a)} & {cells} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(caption_placeholder())
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# captions stay lean: Setup already defines the metrics, regions, camera models, and
# sharpness ratio, so the caption only states the table's identity and what bold means.
def emit_camera_combined(cam):
    """All four regions of the camera-parameter comparison in one table."""
    return emit_combined_table(cam, CAM_ROWS, {}, "tab:recon-phone-cam")


def emit_dense_combined(dense):
    """All four regions of the densification/training-length comparison in one table."""
    return emit_combined_table(dense, DENSE_ROWS, DENSE_DISPLAY, "tab:recon-phone-dense")


def emit_camera_table(region, cam):
    """Camera-parameter comparison for one region (4 sources, 15k default)."""
    return emit_split_table(region, cam, CAM_ROWS, {},
                            f"tab:recon-phone-cam-{REGION_LABEL[region]}")


def emit_dense_table(region, dense):
    """Densification/training-length comparison for one region (OPENCV, 4 configs)."""
    return emit_split_table(region, dense, DENSE_ROWS, DENSE_DISPLAY,
                            f"tab:recon-phone-dense-{REGION_LABEL[region]}")


SESSION_SHORT = {("field_A", "20250627"): "A/0627", ("field_D", "20250627"): "D/0627",
                 ("field_A", "20250715"): "A/0715", ("field_D", "20250715"): "D/0715"}
# full row spec for the appendix: (display-label, variant, exp, iterkey)
CAM_SPEC = [("PINHOLE", "", "baseline", ""), ("OPENCV", "opencv", "baseline", ""),
            ("RADIAL", "radial", "baseline", ""), ("Agisoft", "agisoft", "baseline", "")]
DENSE_SPEC = [("15{,}000, AbsGS", "opencv", "absgrad", "ours_15000"),
              ("30{,}000, default", "opencv", "dense17k_noabsgrad", "ours_30000"),
              ("30{,}000, AbsGS", "opencv", "dense17k", "ours_30000")]


def emit_appendix_region_table(region):
    """LaTeX for the full per-session breakdown of one region: every arm shown for
    each of the four sessions, all metrics, nothing averaged."""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{ll cccc}")
    lines.append("\\hline")
    lines.append(" & & " + " & ".join(h for _, h, _ in COL) + " \\\\")

    def block(title, spec):
        lines.append("\\hline")
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for label, variant, exp, _ in spec:
            for i, (f, d) in enumerate(SESSIONS):
                b = load_scene(f, d, variant, exp)
                arm = label if i == 0 else ""
                cells = " & ".join(fmt(region_metric(b, region, m) if b else None, m)
                                   for m, _, _ in COL)
                lines.append(f"{arm} & {SESSION_SHORT[(f, d)]} & {cells} \\\\")

    block("Camera parameters (15{,}000, default)", CAM_SPEC)
    block("Densification and training length (\\texttt{OPENCV})", DENSE_SPEC)
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(caption_placeholder())
    lines.append(f"\\label{{tab:recon-phone-app-{REGION_LABEL[region]}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    cam = collect([(l, v, e, "") for l, v, e in CAMERAS])
    dense = collect([(l, "opencv", e, k) for l, e, k in DENSE])
    if "--latex" in sys.argv:
        print("% ===== Camera-parameter region tables (main text) =====")
        for region in REGIONS:
            print(emit_camera_table(region, cam))
            print()
        print("% ===== Densification / training-length region tables (main text) =====")
        for region in REGIONS:
            print(emit_dense_table(region, dense))
            print()
    elif "--combined" in sys.argv:
        print("% ===== Camera parameters (combined, all regions) =====")
        print(emit_camera_combined(cam))
        print()
        print("% ===== Densification / training length (combined, all regions) =====")
        print(emit_dense_combined(dense))
        print()
    elif "--appendix" in sys.argv:
        for region in REGIONS:
            print(emit_appendix_region_table(region))
            print()
    else:
        print_plain("CAMERA sources (15k default), avg over 4 sessions", cam)
        print_plain("DENSIFICATION grid (OPENCV), avg over 4 sessions", dense)
