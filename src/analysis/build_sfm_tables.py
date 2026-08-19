"""Builds every SfM Results table (A1-A4, B, C) for the thesis from the on-disk experiment
outputs, so each number traces to a file and a script rather than to memory.

Sources, per session:
  - registration + pose/rot/chamfer/focal/reproj : logs/compare_to_agisoft[_<variant>].json
      (fD/20250627 uses the native two-camera reference, the _2group files, to match the other
       seven sessions; see SFM_CLAIMS_LEDGER.md)
  - registered-image + sub-model count           : logs/colmap_summary.json (registered / input_images)
                                                   + count of <model>/distorted/sparse/* dirs
  - marker geometry (survey/tape cm)             : docs/analysis_results/method_ranking_groupA_rescore.txt
      (produced by src/analysis/rescore_models_geometry.py; incremental sparse/0 vs GLOMAP)
  - metric scale (CV%, tape deviation mm)        : logs/marker_scale.json
  - marker-detection accuracy (C1/C2)            : src/analysis/eval_marker_detection.py (subprocess)

Writes LaTeX table bodies to docs/analysis_results/sfm_results_tables.tex and prints them.
Read-only w.r.t. all experiment data. Run: python src/analysis/build_sfm_tables.py
"""
import json
import os
import re
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHONE = os.path.join(REPO, "input_plots", "phone")
RESCORE = os.path.join(REPO, "docs", "analysis_results", "method_ranking_groupA_rescore.txt")
OUT = os.path.join(REPO, "docs", "analysis_results", "sfm_results_tables.tex")

# the eight canonical sessions, in field/date order
SESSIONS = [
    ("field_A", "20250618"), ("field_A", "20250627"),
    ("field_A", "20250706"), ("field_A", "20250715"),
    ("field_D", "20250618"), ("field_D", "20250627"),
    ("field_D", "20250706"), ("field_D", "20250715"),
]

# short label for the tables
def slabel(field, plot):
    """Short session label like A/0618 for the table rows."""
    return f"{field[-1]}/{plot[4:]}"


def is_fd0627(field, plot):
    """field_D/20250627 is the only session with a native (two-camera) Agisoft reference we prefer."""
    return field == "field_D" and plot == "20250627"


def compare_path(field, plot, variant=""):
    """Path to a compare_to_agisoft json. Adds the _2group suffix for fD/0627 so its reference
    matches the other seven native-reference sessions."""
    stem = "compare_to_agisoft"
    if variant:
        stem += f"_{variant}"
    if is_fd0627(field, plot):
        stem += "_2group"
    return os.path.join(PHONE, field, plot, "logs", f"{stem}.json")


def load_compare(field, plot, variant=""):
    """Reads a compare json and pulls the numbers we tabulate; returns None if the file is absent."""
    p = compare_path(field, plot, variant)
    if not os.path.exists(p):
        # fD/0627 variants without a _2group flavor fall back to the default reference
        if is_fd0627(field, plot):
            alt = os.path.join(PHONE, field, plot, "logs",
                               f"compare_to_agisoft{('_' + variant) if variant else ''}.json")
            if os.path.exists(alt):
                p = alt
            else:
                return None
        else:
            return None
    d = json.load(open(p))
    return {
        "n_common": d.get("n_common"),
        "n_ours": d.get("n_ours"),
        "pose_mm": d["translation_error_m"]["median_m"] * 1000.0,
        "rot_deg": d["rotation_error_deg"]["median_deg"],
        "chamfer_mm": d["point_cloud"]["symmetric_chamfer_median_mm"],
        "focal_pct": d["intrinsics"]["focal_diff_pct"],
        "reproj_px": d["reprojection"]["ours_recomputed"]["mean_px"],
    }


def load_registration(field, plot, model_subdir=""):
    """Registered-image count + sub-model count for one reconstruction. model_subdir='' is the
    baseline (session root); e.g. 'sift_2048' reads the variant folder. Sub-models are counted from
    the distorted/sparse/* directories the mapper wrote."""
    base = os.path.join(PHONE, field, plot, model_subdir) if model_subdir else os.path.join(PHONE, field, plot)
    summ = os.path.join(base, "logs", "colmap_summary.json")
    reg, inp = None, None
    if os.path.exists(summ):
        s = json.load(open(summ))
        reg, inp = s.get("registered"), s.get("input_images")
    dist = os.path.join(base, "distorted", "sparse")
    nsub = None
    if os.path.isdir(dist):
        nsub = len([x for x in os.listdir(dist) if os.path.isdir(os.path.join(dist, x))])
    return {"registered": reg, "input": inp, "n_submodels": nsub}


def parse_rescore():
    """Parses method_ranking_groupA_rescore.txt into {sesskey: {model: (survey_cm, tape_cm)}}.
    sesskey is 'field_A/20250618'; model is 'sparse/0' (incremental) or the glomap dir."""
    out = {}
    if not os.path.exists(RESCORE):
        return out
    cur = None
    hdr = re.compile(r"===\s+(field_[AD]/\d{8}):")
    row = re.compile(r"^\s*(\S+)\s+\d+\s+\d+\s+([\d.]+)\s+\(n\d+\)\s+([\d.]+)\s+\(n\d+\)")
    for line in open(RESCORE):
        m = hdr.search(line)
        if m:
            cur = m.group(1)
            out.setdefault(cur, {})
            continue
        r = row.match(line)
        if r and cur is not None:
            model, survey, tape = r.group(1), float(r.group(2)), float(r.group(3))
            # the file re-lists some sessions in a later spot-check block (only sparse/0 + aliked, no
            # glomap) -> keep the FIRST occurrence so the incremental-vs-GLOMAP pair survives
            out[cur].setdefault(model, (survey, tape))
    return out


def load_scale(field, plot):
    """Metric-scale consistency (CV of per-pair conversions, as %) and tape deviation (mm)."""
    p = os.path.join(PHONE, field, plot, "logs", "marker_scale.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {"cv_pct": d.get("scale_ratio_cv", 0.0) * 100.0,
            "tape_dev_mm": d.get("ours_vs_tape_mean_abs_mm")}


def run_verified_geometry(field, plot):
    """Runs eval_marker_geometry_gt.py and returns the VERIFIED-GT marker geometry (survey_cm,
    tape_cm) for the session -- detection-free, used for the fD/0627 baseline-vs-markers cell so it
    matches the Setup ('verified positions on field D/0627'). Also returns the our-detector pair."""
    cmd = [sys.executable, os.path.join(REPO, "src", "analysis", "eval_marker_geometry_gt.py"),
           "--field", field, "--plot", plot]
    try:
        txt = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
    except Exception:
        return None
    def grab(tag):
        m = re.search(tag + r".*?vs SURVEY\s+([\d.]+) cm.*?vs TAPE\s+([\d.]+) cm", txt)
        return (float(m.group(1)), float(m.group(2))) if m else None
    return {"verified": grab("VERIFIED GT"), "detector": grab("our detector")}


def run_detection(field, plot, gt_source):
    """Runs eval_marker_detection.py and parses recall/precision/median-localization from its
    printed report. gt_source: 'demoanlage' (vs Agisoft's detector) or 'input_plots' (verified GT)."""
    cmd = [sys.executable, os.path.join(REPO, "src", "analysis", "eval_marker_detection.py"),
           "--field", field, "--plot", plot, "--gt_source", gt_source]
    try:
        txt = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
    except Exception:
        return None
    def grab(pat):
        m = re.search(pat, txt)
        return float(m.group(1)) if m else None
    rec = grab(r"RECALL.*?([\d.]+)%")
    prec = grab(r"PRECISION.*?([\d.]+)%")
    loc = grab(r"median\s+([\d.]+)")
    nreg = grab(r"reference sightings \(Pinned\):\s+(\d+)")
    return {"recall": rec, "precision": prec, "loc_px": loc, "n_gt": int(nreg) if nreg else None}


# ----------------------------------------------------------------------------- table builders
def fnum(x, fmt):
    """Format a number, or an em-dash if it is missing."""
    return (fmt % x) if x is not None else "--"


def tbl_a1():
    """A1 front-end: registration and median pose error vs Agisoft for SIFT at its default 3200-pixel
    size, SIFT at the matched 2048-pixel size, and ALIKED, per session. Pose is reported only where a
    front-end forms one connected model covering most of the images (a collapse carries none)."""
    def reg_cell(d):
        """One reconstruction as 'reg/input' + model count (a missing count means one model)."""
        reg = f"{d['registered']}/{d['input']}" if d['registered'] else "--"
        mod = d['n_submodels'] if d['n_submodels'] else 1
        return reg, mod
    def pose_ok(d):
        """Pose is meaningful only for one connected model that also registers most images."""
        frac = (d['registered'] / d['input']) if (d['registered'] and d['input']) else 0
        mod = d['n_submodels'] if d['n_submodels'] else 1
        return mod == 1 and frac >= 0.5
    lines = []
    for f, p in SESSIONS:
        d3 = load_registration(f, p, "sift")        # SIFT default (3200 px)
        d2 = load_registration(f, p, "sift_2048")   # SIFT matched to ALIKED (2048 px)
        al = load_registration(f, p)                # ALIKED baseline
        r3, m3 = reg_cell(d3)
        r2, m2 = reg_cell(d2)
        ra, ma = reg_cell(al)
        p3 = load_compare(f, p, "sift")             # SIFT@3200 pose vs Agisoft
        p2 = load_compare(f, p, "sift2048")         # SIFT@2048 pose vs Agisoft
        pa = load_compare(f, p)                      # ALIKED pose vs Agisoft
        p3s = fnum(p3['pose_mm'], '%.1f') if (pose_ok(d3) and p3) else "--"
        p2s = fnum(p2['pose_mm'], '%.1f') if (pose_ok(d2) and p2) else "--"
        pas = fnum(pa['pose_mm'], '%.1f') if pa else "--"
        lines.append(f"{slabel(f,p)} & {r3} & {m3} & {p3s} & {r2} & {m2} & {p2s} & {ra} & {ma} & {pas} \\\\")
    return ("% A1 front-end: session & SIFT@3200 reg mod pose & SIFT@2048 reg mod pose & "
            "ALIKED reg mod pose\n" + "\n".join(lines))


def tbl_a2():
    """A2 pairing: exhaustive (baseline) vs sequential pose + rotation."""
    lines = []
    for f, p in SESSIONS:
        ex = load_compare(f, p)
        sq = load_compare(f, p, "seq")
        lines.append(
            f"{slabel(f,p)} & {fnum(ex['pose_mm'],'%.1f')} & {fnum(ex['rot_deg'],'%.2f')} & "
            f"{fnum(sq['pose_mm'],'%.1f') if sq else '--'} & {fnum(sq['rot_deg'],'%.2f') if sq else '--'} \\\\")
    return "% A2 pairing: session & exhaustive mm & exhaustive deg & sequential mm & sequential deg\n" + "\n".join(lines)


def tbl_a3(rescore):
    """A3 mapper: incremental vs GLOMAP marker-geometry (survey cm)."""
    lines = []
    for f, p in SESSIONS:
        key = f"{f}/{p}"
        d = rescore.get(key, {})
        incr = d.get("sparse/0")
        glo = d.get("sparse_glomap_20260721/0")
        lines.append(
            f"{slabel(f,p)} & {fnum(incr[0] if incr else None,'%.2f')} & "
            f"{fnum(glo[0] if glo else None,'%.1f')} \\\\")
    return "% A3 mapper: session & incremental survey cm & GLOMAP survey cm\n" + "\n".join(lines)


def tbl_a4():
    """A4 camera model: pinhole/radial/opencv pose + internal reprojection (all 8 sessions), plus
    FULL_OPENCV registration only -- it collapses to 2-3 images per session, so it has no valid
    pose or reprojection and only the registered-image count is shown (and was only run on 4
    sessions, so the rest show '--')."""
    lines = []
    for f, p in SESSIONS:
        pin = load_compare(f, p)
        rad = load_compare(f, p, "radial")
        opc = load_compare(f, p, "opencv")
        fop = load_registration(f, p, "full_opencv")
        def cell(d):
            return (f"{fnum(d['pose_mm'],'%.1f') if d else '--'} & "
                    f"{fnum(d['reproj_px'],'%.3f') if d else '--'}")
        fop_reg = f"{fop['registered']}/{fop['input']}" if fop['registered'] else "--"
        lines.append(f"{slabel(f,p)} & {cell(pin)} & {cell(rad)} & {cell(opc)} & {fop_reg} \\\\")
    return ("% A4 camera model: session & pinhole mm & pinhole px & radial mm & radial px & "
            "opencv mm & opencv px & full_opencv reg\n" + "\n".join(lines))


def tbl_b_agisoft():
    """B baseline vs Agisoft: registration, pose, rotation, chamfer, focal%, reproj."""
    lines = []
    for f, p in SESSIONS:
        d = load_compare(f, p)
        reg = load_registration(f, p)
        rr = f"{reg['registered']}/{reg['input']}" if reg['registered'] else "--"
        lines.append(
            f"{slabel(f,p)} & {rr} & {fnum(d['pose_mm'],'%.1f')} & {fnum(d['rot_deg'],'%.2f')} & "
            f"{fnum(d['chamfer_mm'],'%.1f')} & {fnum(d['focal_pct'],'%+.2f')} & {fnum(d['reproj_px'],'%.3f')} \\\\")
    return ("% B vs Agisoft: session & reg & pose mm & rot deg & chamfer mm & focal % & reproj px\n"
            + "\n".join(lines))


def tbl_b_markers(rescore):
    """B baseline vs markers: geometry (survey cm, tape cm), scale CV%, tape deviation mm."""
    lines = []
    for f, p in SESSIONS:
        key = f"{f}/{p}"
        geo = rescore.get(key, {}).get("sparse/0")   # our-detector geometry, all 8 (uniform tool)
        if is_fd0627(f, p):
            # Setup says the marker metric uses the VERIFIED positions on this session
            vg = run_verified_geometry(f, p)
            if vg and vg["verified"]:
                geo = vg["verified"]
        sc = load_scale(f, p)
        lines.append(
            f"{slabel(f,p)} & {fnum(geo[0] if geo else None,'%.2f')} & {fnum(geo[1] if geo else None,'%.2f')} & "
            f"{fnum(sc['cv_pct'] if sc else None,'%.2f')} & {fnum(sc['tape_dev_mm'] if sc else None,'%.1f')} \\\\")
    return ("% B vs markers: session & geom survey cm & geom tape cm & scale CV % & tape dev mm\n"
            "% (D/0627 geometry = VERIFIED GT; our-detector gives 0.55/0.72 cm)\n"
            + "\n".join(lines))


def tbl_c1():
    """C1 detection agreement vs Agisoft's detector, all 8 sessions."""
    lines = []
    for f, p in SESSIONS:
        r = run_detection(f, p, "demoanlage")
        if not r:
            lines.append(f"{slabel(f,p)} & -- & -- & -- \\\\")
            continue
        lines.append(f"{slabel(f,p)} & {fnum(r['recall'],'%.1f')} & {fnum(r['precision'],'%.1f')} & "
                     f"{fnum(r['loc_px'],'%.2f')} \\\\")
    return "% C1 detection vs Agisoft: session & recall % & precision % & localization px\n" + "\n".join(lines)


def tbl_c2():
    """C2 detection vs hand-verified GT, field_D/20250627 only."""
    r = run_detection("field_D", "20250627", "input_plots")
    if not r:
        return "% C2: (eval failed)\nD/0627 & -- & -- & -- \\\\"
    return ("% C2 detection vs verified GT: session & precision % & recall % & localization px\n"
            f"D/0627 & {fnum(r['precision'],'%.1f')} & {fnum(r['recall'],'%.1f')} & {fnum(r['loc_px'],'%.2f')} \\\\")


def main():
    rescore = parse_rescore()
    blocks = [
        ("TABLE A1  front-end (registration + pose)", tbl_a1()),
        ("TABLE A2  image pairing (pose)", tbl_a2()),
        ("TABLE A3  mapper (marker geometry, survey cm)", tbl_a3(rescore)),
        ("TABLE A4  camera model (pose + reproj)", tbl_a4()),
        ("TABLE B1  baseline vs Agisoft", tbl_b_agisoft()),
        ("TABLE B2  baseline vs markers", tbl_b_markers(rescore)),
        ("TABLE C1  detection vs Agisoft detector", tbl_c1()),
        ("TABLE C2  detection vs verified GT", tbl_c2()),
    ]
    parts = []
    for title, body in blocks:
        parts.append(f"% ===== {title} =====\n{body}\n")
    text = "\n".join(parts)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)
    print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
