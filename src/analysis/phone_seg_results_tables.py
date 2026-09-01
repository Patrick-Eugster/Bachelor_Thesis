"""Emit the phone 3D-seg Results LaTeX table BODIES (rows only, no caption, no \\begin{tabular}).

Two tables, both read-only from the scored JSONs:
  A = mask-generation + match-threshold sweep on field_A/20250715 (full cull, opencv 15k)
  B = two-session winner + reconstruction levers (field_A/20250715 and field_D/20250627)

Pixel metrics come from docs/analysis_results/phone_seg_cpu_eval.json, instance metrics from
phone_seg_instance_eval.json. Captions are hand-authored in main.tex. Run from repo root.
"""
import json

PX = {r["name"]: r for r in json.load(open("docs/analysis_results/phone_seg_cpu_eval.json"))["runs"] if r.get("status") == "ok"}
IN = {r["name"]: r for r in json.load(open("docs/analysis_results/phone_seg_instance_eval.json"))["runs"] if r.get("status") == "ok"}


def p(name):
    """pixel row (IoU, P, R, F1) or None."""
    return PX.get(name)


def i(name):
    """instance row (f1, f1_iou25, n_pred) or None."""
    return IN.get(name)


def f(x, d=3):
    """format a float, or a dash if missing."""
    return f"{x:.{d}f}" if x is not None else "---"


# ---------------- Table 0: tilt-fix + cull ladder (A/0715, SAM1 per_tile @0.35) ----------------
# The frustum-only PRE-tilt-fix run is scored under the label "ocv15k" (folder ocv15k_yolov5_pertile).
TABLE_0 = [
    ("frustum cull only (pre-tilt-fix)", "ocv15k"),
    ("+ tilt fix (marker-plane cut)",    "ocv15k_groundfix"),
    ("+ ROI + marker (full cull)",       "ocv15k_frust_paint"),
]


def inst_cells(ins):
    """instance P, R, F1, F1@25 (dashes if the run isn't instance-scored)."""
    if not ins:
        return "--- & --- & --- & ---"
    return f'{f(ins["precision"])} & {f(ins["recall"])} & {f(ins["f1"])} & {f(ins["f1_iou25"])}'


def table_0():
    """body rows: config & pixel(IoU,P,R,F1) & instance(P,R,F1,F1@25)."""
    print("% --- Table 0 body (field A/20250715, SAM1 per_tile @0.35, opencv 15k) ---")
    for lbl, name in TABLE_0:
        px = p(name)
        print(f'{lbl} & {f(px["iou"])} & {f(px["precision"])} & {f(px["recall"])} & {f(px["f1"])} & {inst_cells(i(name))} \\\\')


# ---------------- Table A: mask-gen + match sweep (A/0715) ----------------
# (label, pixel-run-name)  — instance pulled by the same name where present
TABLE_A = [
    ("pt SAM2 @0.35",            "ocv15k_conf035"),
    ("pt SAM2 @0.70",            "ocv15k_conf070"),
    ("pt SAM1 @0.70",            "ocv15k_sam1_conf070"),
    ("ph SAM2 @0.22",            "ocv15k_perhead_sam2_conf022"),
    ("ph SAM2 @0.35",            "ocv15k_perhead_sam2"),
    ("ph SAM2 @0.70",            "ocv15k_perhead_sam2_conf070"),
    ("ph SAM2 @0.70, IoU 0.6",   "ocv15k_perhead_sam2_conf070_iou06"),
]


def table_a():
    """body rows: config & pixel(IoU,P,R,F1) & instance(P,R,F1,F1@25) & nPred."""
    print("% --- Table A body (field A/20250715, full cull, opencv 15k) ---")
    for lbl, name in TABLE_A:
        px, ins = p(name), i(name)
        npred = str(ins["n_pred"]) if ins else "---"
        row = f'{lbl} & {f(px["iou"])} & {f(px["precision"])} & {f(px["recall"])} & {f(px["f1"])} & {inst_cells(ins)} & {npred} \\\\'
        if "IoU 0.6" in lbl:
            row = f'\\textbf{{{lbl}}} & ' + ' & '.join(row.split(' & ')[1:])
        print(row)


# ---------------- Table B: two-session winner + recon levers ----------------
# (label, A-run-name, D-run-name)
TABLE_B = [
    ("15k default (winner)", "ocv15k_perhead_sam2_conf070_iou06", "D0627_ocv15k_perhead_sam2_conf070_iou06"),
    ("+ AbsGS",              "A0715_absgrad_perhead_conf070_iou06", "D0627_absgrad_perhead_conf070_iou06"),
    ("+ 30k default",        "A0715_ocv30k_noabsgrad_perhead_conf070_iou06", None),
]


def cells(name):
    """the four cells for one session: pixel IoU, pixel F1, instance F1, instance F1@25."""
    if name is None:
        return "--- & --- & --- & ---"
    px, ins = p(name), i(name)
    return f'{f(px["iou"])} & {f(px["f1"])} & {f(ins["f1"]) if ins else "---"} & {f(ins["f1_iou25"]) if ins else "---"}'


def table_b():
    """body rows: config & A(IoU,F1,instF1,F1@25) & D(IoU,F1,instF1,F1@25)."""
    print("% --- Table B body (field A/20250715 and field D/20250627) ---")
    for lbl, a, d in TABLE_B:
        cell = f'{lbl} & {cells(a)} & {cells(d)} \\\\'
        if "winner" in lbl:
            cell = f'\\textbf{{{lbl}}} & ' + ' & '.join(cell.split(' & ')[1:])
        print(cell)


if __name__ == "__main__":
    table_0()
    print()
    table_a()
    print()
    table_b()
