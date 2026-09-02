"""Step 3 of the marker pipeline: turn the triangulated 3D markers into a METRIC scale.

The triangulated points (`logs/marker_points3d.json` from triangulate_markers.py) live in COLMAP's
arbitrary-scale world frame. This script recovers the scale factor (COLMAP units -> metres). It has
two modes (config `scale_source`):

  survey  (default)  surveyed XYZ  demoanlage2025_v0/metadata/markers/field_<L>_coordinates.txt
           (total-station / RTK coords in CH1903+/LV95, metres) -> all 15 pairwise distances +
           a rigorous Umeyama similarity fit (scale + R + t) giving a per-marker residual in mm. The
           tape xlsx is loaded as an independent CROSS-CHECK.

  tape               TAPE distances ONLY  Demoanlage-2025-markers-manual-distances.xlsx, sheet
           "plot <L>" -> scale = median(tape_dist / our_recon_dist) over the measured pairs. NO survey
           needed (survey is loaded only as an optional extra check if the file happens to be present).
           This gives SIZE only (a uniform scale, no absolute world frame) — which is all phenotyping
           needs — and sidesteps the RTK-GPS ~2 cm survey error. See docs/preprocessing/markers/MARKER_INTEGRATION_PLAN.md.

Marker target<->code map (from the spec PDF, see docs/preprocessing/markers/MARKER_CODE_STRUCTURE.md):
    target 1->113  target 2->105  target 3->89  target 4->101  target 5->85  target 6->77

Output (READ-ONLY on the data, writes into the plot's logs/):
    logs/marker_scale.json   scale factor, per-pair distances, residuals, and the cross-check.

The scale + the survey XYZ are exactly what a later GCP step would feed COLMAP; this step both gives
metric output now and validates the markers are good enough to anchor it. See MARKER_INTEGRATION_PLAN.md.

Usage:
    python src/preprocessing/marker_scale.py field=field_A plot=20250609
    python src/preprocessing/marker_scale.py field=field_A plot=20250609 scale_source=tape
"""

import itertools
import json
import os
import re
import zipfile

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# target number (Agisoft label) -> our decoded 12-bit code. Ground truth from the spec PDF.
TARGET_TO_CODE = {1: 113, 2: 105, 3: 89, 4: 101, 5: 85, 6: 77}


def field_letter(field):
    """'field_A' -> 'A' (the survey file + xlsx sheet are keyed by the bare plot letter)."""
    return field.split("_")[-1]


def load_survey(path):
    """Parse a '<survey> field_<L>_coordinates.txt' -> {code: np.array([X,Y,Z])} in metres.

    Lines look like 'target 1,2693807.271,1255676.304,534.912'; '#' lines are header/CRS. We map the
    target number to our code via TARGET_TO_CODE so everything downstream is keyed by code."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            m = re.match(r"target\s+(\d+)", parts[0], re.I)
            if not m:
                continue
            tnum = int(m.group(1))
            if tnum not in TARGET_TO_CODE:
                continue
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            out[TARGET_TO_CODE[tnum]] = xyz
    return out


def load_tape(xlsx_path, sheet_name):
    """Parse one plot sheet of the manual-distances xlsx -> {(code_a,code_b): metres}.

    The sheet is an upper-triangular 6x6 distance matrix in CM. Row r (Excel rows 2..7) is the
    distances FROM target r-1 to targets r-1+1 .. 6, listed left-to-right. We read each row's numeric
    cells (values > 10 == cm distances; the 1..6 header/label cells are <= 6 so they filter out) in
    column order and assign them to consecutive destination targets. Returns metric, keyed by code."""
    z = zipfile.ZipFile(xlsx_path)
    wb = z.read("xl/workbook.xml").decode("utf8", "ignore")
    names = re.findall(r'<sheet[^>]*name="([^"]*)"', wb)
    if sheet_name not in names:
        raise ValueError(f"sheet '{sheet_name}' not in {names}")
    sheet_file = f"xl/worksheets/sheet{names.index(sheet_name) + 1}.xml"
    xml = z.read(sheet_file).decode("utf8", "ignore")

    # collect (col_letters, row_num, value) for every cell that has a numeric <v>, grouped by row
    rows = {}
    for m in re.finditer(r'<c r="([A-Z]+)(\d+)"[^>]*?>(.*?)</c>', xml):
        col, row, body = m.group(1), int(m.group(2)), m.group(3)
        vm = re.search(r"<v>(.*?)</v>", body)
        if not vm:
            continue
        try:
            val = float(vm.group(1))
        except ValueError:
            continue
        if row == 1 or col == "A":  # header row / label column
            continue
        if val <= 10:  # the stray 1..6 target labels, not a distance
            continue
        rows.setdefault(row, []).append((col, val))

    out = {}
    for row in sorted(rows):
        src_target = row - 1
        ordered = [v for _, v in sorted(rows[row], key=lambda cv: _col_index(cv[0]))]
        for k, val_cm in enumerate(ordered):
            dst_target = src_target + 1 + k
            if src_target in TARGET_TO_CODE and dst_target in TARGET_TO_CODE:
                a, b = TARGET_TO_CODE[src_target], TARGET_TO_CODE[dst_target]
                out[tuple(sorted((a, b)))] = val_cm / 100.0  # cm -> m
    return out


def _col_index(letters):
    """Excel column letters -> 0-based index ('A'->0, 'B'->1, ... 'Z'->25, 'AA'->26)."""
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def load_ours(points_json):
    """Read triangulate_markers.py's marker_points3d.json -> {code(int): np.array([x,y,z])}.
    Skips unsolved markers (None) so it doesn't crash on a partial reconstruction."""
    d = json.load(open(points_json))
    return {int(code): np.array(p["xyz"], dtype=np.float64)
            for code, p in d["points3d"].items() if p}


# --- marker quality guard --------------------------------------------------------------------
# A triangulated marker must clear ALL of these to be trusted as a metric-scale anchor. Weak
# markers (low triangulation parallax, few inlier views, high reprojection error) have unreliable
# 3D positions that poison the distance-ratio scale — empirically the ~5deg-parallax markers in a
# bad Route-2 run blew the tape CV up to 196%. On every good run all 6 markers clear these (lowest
# seen: parallax 36.7deg, inliers 6, max-reproj 6.0px) so the guard is a NO-OP on good data.
# Set any threshold to 0 to disable that gate. See docs/preprocessing/markers/MARKER_INTEGRATION_PLAN.md open item (a).
QUALITY_DEFAULTS = {
    "quality_min_parallax_deg": 10.0,  # triangulation angle; < this = depth poorly constrained
    "quality_min_inlier_views": 4,     # inlier views backing the 3D point
    "quality_max_reproj_px": 8.0,      # max per-view reprojection error of the 3D point
}


def quality_thresholds(cfg):
    """Pull the three guard thresholds from cfg (DictConfig or dict), falling back to QUALITY_DEFAULTS.
    Single place that reads the knobs so marker_scale, apply, and the orchestrator failsafe agree."""
    def g(k):
        return cfg.get(k, QUALITY_DEFAULTS[k]) if cfg is not None else QUALITY_DEFAULTS[k]
    return (float(g("quality_min_parallax_deg")),
            int(g("quality_min_inlier_views")),
            float(g("quality_max_reproj_px")))


def marker_quality_ok(q, min_par, min_inl, max_rep):
    """Does one marker's quality dict (from marker_points3d.json) clear the guard?
    Returns (ok, reasons) — reasons lists each failed gate for the log. A threshold of 0 = gate off."""
    reasons = []
    if min_par > 0 and q.get("parallax_deg", 0.0) < min_par:
        reasons.append(f"parallax {q.get('parallax_deg', 0.0):.1f}<{min_par:g}")
    if min_inl > 0 and q.get("n_inliers", 0) < min_inl:
        reasons.append(f"inliers {q.get('n_inliers', 0)}<{min_inl}")
    if max_rep > 0 and q.get("max_reproj_px", 1e9) > max_rep:
        reasons.append(f"reproj {q.get('max_reproj_px', 0.0):.1f}>{max_rep:g}")
    return (len(reasons) == 0), reasons


def load_ours_full(points_json):
    """Read marker_points3d.json -> {code(int): quality dict with 'xyz' as np.array}. Skips unsolved."""
    d = json.load(open(points_json))
    out = {}
    for code, p in d["points3d"].items():
        if not p:
            continue
        q = dict(p)
        q["xyz"] = np.array(p["xyz"], dtype=np.float64)
        out[int(code)] = q
    return out


def filter_ours(cfg, points_json):
    """Load triangulated markers AND apply the quality guard. Returns (ours_xyz, report).

    SINGLE SOURCE OF TRUTH: marker_scale.py (the report), apply_metric_transform.py (the applied
    transform) and run_preprocessing.py (the failsafe count) all call this, so they agree on exactly
    which markers anchor the metric scale. report = {thresholds, kept, dropped[{code,reasons}]}."""
    full = load_ours_full(points_json)
    min_par, min_inl, max_rep = quality_thresholds(cfg)
    kept, dropped = {}, []
    for code in sorted(full):
        ok, reasons = marker_quality_ok(full[code], min_par, min_inl, max_rep)
        if ok:
            kept[code] = full[code]["xyz"]
        else:
            dropped.append({"code": code, "reasons": reasons})
    report = {
        "thresholds": {"min_parallax_deg": min_par, "min_inlier_views": min_inl,
                       "max_reproj_px": max_rep},
        "kept": sorted(kept), "dropped": dropped,
    }
    return kept, report


def pairwise(points):
    """{code: xyz} -> {(code_a,code_b): distance} over all unordered pairs."""
    out = {}
    for a, b in itertools.combinations(sorted(points), 2):
        out[(a, b)] = float(np.linalg.norm(points[a] - points[b]))
    return out


def umeyama(src, dst):
    """Least-squares similarity (scale s, rotation R, translation t) mapping src->dst (Umeyama 1991).

    src, dst are (N,3). Returns (s, R, t) minimising || dst - (s R src + t) ||. Solving scale jointly
    with pose is more rigorous than a raw distance-ratio: it uses every coordinate, not just lengths."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    Sigma = (Xd.T @ Xs) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (Xs ** 2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / var_s)
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def tape_scale(ours, tape, mad_k=0.0):
    """Tape-only scale = median(tape_dist / our_recon_dist) over pairs measured by BOTH.

    Returns (scale, ratio_cv, shared_pairs, dropped_pairs). Pure SIZE — no orientation/position,
    since tape gives no world frame. Median makes it robust to a single bad pair. Single source of
    truth, reused by apply_metric_transform.py so the model uses the exact same scale this report
    computes. When mad_k>0 a robust outlier reject drops any pair whose ratio is > mad_k MADs from the
    median (catches a wrong-distance pair, e.g. the known tape entry error) before the final median;
    mad_k=0 (default) → no rejection and the first three returns are byte-identical to before."""
    d_ours = pairwise(ours)
    shared_pairs = sorted(p for p in d_ours if p in tape)
    if not shared_pairs:
        raise SystemExit("tape-only: no marker pair is both triangulated and tape-measured")
    ratios = np.array([tape[p] / d_ours[p] for p in shared_pairs])
    dropped_pairs = []
    if mad_k and len(ratios) >= 4:
        med = np.median(ratios)
        mad = float(np.median(np.abs(ratios - med)))
        if mad > 0:
            keep = np.abs(ratios - med) <= mad_k * 1.4826 * mad  # 1.4826 = MAD→sigma for normal
            if keep.sum() >= 2 and not keep.all():
                dropped_pairs = [shared_pairs[i] for i in range(len(keep)) if not keep[i]]
                shared_pairs = [shared_pairs[i] for i in range(len(keep)) if keep[i]]
                ratios = ratios[keep]
    return float(np.median(ratios)), float(ratios.std() / ratios.mean()), shared_pairs, dropped_pairs


def run_tape_only(cfg, src, letter, ours, qreport=None):
    """Tape-only metric scale: scale = median(tape_dist / our_recon_dist) over the measured pairs.

    Needs NO survey XYZ — gives SIZE only (a uniform scale, no absolute world frame), which is all
    phenotyping needs, and avoids the RTK-GPS error. Survey is loaded ONLY as an optional cross-check
    when the file is present. Writes the same logs/marker_scale.json (with scale_source='tape').
    `ours` is already quality-filtered upstream; qreport records which markers the guard dropped."""
    tape = load_tape(cfg.tape_xlsx, f"plot {letter}")  # REQUIRED in this mode
    print(f"tape distances loaded: {len(tape)} pairs from sheet 'plot {letter}'")

    mad_k = float(cfg.get("quality_ratio_mad_k", 3.5))
    d_ours = pairwise(ours)
    scale, ratio_cv, shared_pairs, dropped_pairs = tape_scale(ours, tape, mad_k=mad_k)
    codes = sorted({c for p in shared_pairs for c in p})

    # optional survey cross-check — never required, just printed if the file exists
    survey = {}
    try:
        survey = load_survey(cfg.survey_file.replace("<L>", letter))
    except Exception as e:  # noqa: BLE001
        print(f"(no survey cross-check: {e})")
    d_survey = pairwise({c: survey[c] for c in codes if c in survey}) if survey else {}

    pair_rows, ours_vs_tape, ours_vs_survey = [], [], []
    for p in shared_pairs:
        ours_m = d_ours[p] * scale
        row = {
            "pair": list(p),
            "ours_m": round(ours_m, 4),
            "tape_m": round(tape[p], 4),
            "survey_m": round(d_survey[p], 4) if p in d_survey else None,
            "ours_minus_tape_mm": round((ours_m - tape[p]) * 1000, 1),
        }
        ours_vs_tape.append(abs(ours_m - tape[p]) * 1000)
        if p in d_survey:
            row["ours_minus_survey_mm"] = round((ours_m - d_survey[p]) * 1000, 1)
            ours_vs_survey.append(abs(ours_m - d_survey[p]) * 1000)
        pair_rows.append(row)

    warn_cv = float(cfg.get("quality_warn_cv", 0.05))
    result = {
        "field": cfg.field,
        "plot": cfg.plot,
        "scale_source": "tape",
        "shared_markers": codes,
        "scale_metric": scale,                  # the applied scale (m / colmap-unit)
        "scale_tape_ratio_median": scale,
        "scale_ratio_cv": ratio_cv,
        "n_tape_pairs": len(shared_pairs),
        "ours_vs_tape_mean_abs_mm": round(float(np.mean(ours_vs_tape)), 2),
        "ours_vs_survey_mean_abs_mm": round(float(np.mean(ours_vs_survey)), 2) if ours_vs_survey else None,
        # quality guard: which markers/pairs the guard removed before scaling (open item (a))
        "quality_guard": qreport or {},
        "mad_dropped_pairs": [list(p) for p in dropped_pairs],
        "ratio_cv_warn_threshold": warn_cv,
        "scale_reliable": bool(ratio_cv <= warn_cv),
        "pairs": pair_rows,
    }
    out_path = os.path.join(src, cfg.output_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=1)

    print("\n" + "=" * 70)
    print(f"  METRIC SCALE (TAPE-ONLY)  {cfg.field}/{cfg.plot}   "
          f"(markers: {len(codes)}, pairs: {len(shared_pairs)})")
    print("=" * 70)
    if qreport and qreport.get("dropped"):
        for d in qreport["dropped"]:
            print(f"  [quality guard] dropped marker {d['code']}: {', '.join(d['reasons'])}")
    if dropped_pairs:
        print(f"  [quality guard] MAD reject (k={mad_k:g}) dropped pair(s): "
              + ", ".join(str(tuple(p)) for p in dropped_pairs))
    print(f"  scale (tape ratio)     : {scale:.6f}  m / colmap-unit   (CV {ratio_cv*100:.2f}%)")
    if ratio_cv > warn_cv:
        print(f"  !!! WARNING: ratio CV {ratio_cv*100:.2f}% > {warn_cv*100:.1f}% — scale is UNRELIABLE "
              f"(markers/tape disagree on size). Treat the metric model with caution.")
    print(f"  ours vs tape (dist)    : {np.mean(ours_vs_tape):.2f} mm mean-abs over "
          f"{len(shared_pairs)} pairs")
    if ours_vs_survey:
        print(f"  ours vs survey (dist)  : {np.mean(ours_vs_survey):.2f} mm mean-abs  "
              f"[survey present — cross-check only, NOT used for scale]")
    print("-" * 70)
    print(f"  {'pair':>12}  {'ours_m':>8}  {'tape_m':>8}  {'survey_m':>9}  {'o-t_mm':>7}")
    for r in pair_rows:
        sv = f"{r['survey_m']:.4f}" if r["survey_m"] is not None else "   -   "
        print(f"  {str(tuple(r['pair'])):>12}  {r['ours_m']:>8.4f}  {r['tape_m']:>8.4f}  "
              f"{sv:>9}  {r['ours_minus_tape_mm']:>7.1f}")
    print("=" * 70)
    print(f"wrote {out_path}")


@hydra.main(version_base=None, config_path="../../../configs/preprocessing", config_name="marker_scale")
def main(cfg: DictConfig):
    """Recover metric scale from triangulated markers — survey (default) or tape-only mode."""
    src = cfg.source_path
    letter = field_letter(cfg.field)
    points_json = os.path.join(src, cfg.points_json)
    print(OmegaConf.to_yaml(cfg))

    # quality guard: drop weak markers (low parallax / few views / high reproj) BEFORE scaling so
    # they can't poison the scale — single source of truth shared with apply + the orchestrator.
    ours, qreport = filter_ours(cfg, points_json)
    if qreport["dropped"]:
        print("[quality guard] dropped weak markers before scaling:")
        for d in qreport["dropped"]:
            print(f"   {d['code']}: {', '.join(d['reasons'])}")
    print(f"[quality guard] markers anchoring the scale: {qreport['kept']}")
    if len(ours) < 2:
        raise SystemExit(f"quality guard left < 2 usable markers ({qreport['kept']}); "
                         f"cannot recover scale. Loosen quality_* thresholds or improve the capture.")

    # tape-only mode: scale straight from the tape distances, no survey needed
    if str(cfg.get("scale_source", "survey")) == "tape":
        run_tape_only(cfg, src, letter, ours, qreport)
        return

    # --- survey mode (default, unchanged) ---
    survey_path = cfg.survey_file.replace("<L>", letter)
    survey = load_survey(survey_path)
    codes = sorted(set(ours) & set(survey))
    print(f"markers in both ours+survey: {codes}")
    if len(codes) < 2:
        raise SystemExit("need >= 2 shared markers to recover scale")

    # tape is optional (a second reference); don't fail the run if the xlsx is missing
    tape = {}
    try:
        tape = load_tape(cfg.tape_xlsx, f"plot {letter}")
        print(f"tape distances loaded: {len(tape)} pairs from sheet 'plot {letter}'")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: tape xlsx not loaded ({e}); continuing with survey only")

    d_ours = pairwise({c: ours[c] for c in codes})
    d_survey = pairwise({c: survey[c] for c in codes})

    # --- distance-ratio scale (simple, robust): survey_metres / ours_units per pair ---
    ratios = np.array([d_survey[p] / d_ours[p] for p in d_ours])
    scale_ratio = float(np.median(ratios))
    ratio_spread = float(ratios.std() / ratios.mean())  # coefficient of variation

    # --- rigorous Umeyama fit (scale+R+t) on the shared markers ---
    P_ours = np.array([ours[c] for c in codes])
    P_survey = np.array([survey[c] for c in codes])
    s, R, t = umeyama(P_ours, P_survey)
    fitted = (s * (R @ P_ours.T)).T + t
    resid_mm = np.linalg.norm(fitted - P_survey, axis=1) * 1000.0
    rms_mm = float(np.sqrt((resid_mm ** 2).mean()))

    # --- per-pair table: ours (scaled to m), survey, tape ---
    pair_rows = []
    tape_vs_survey = []
    ours_vs_survey = []
    for p in sorted(d_ours):
        ours_m = d_ours[p] * s
        row = {
            "pair": list(p),
            "ours_m": round(ours_m, 4),
            "survey_m": round(d_survey[p], 4),
            "tape_m": round(tape[p], 4) if p in tape else None,
            "ours_minus_survey_mm": round((ours_m - d_survey[p]) * 1000, 1),
        }
        ours_vs_survey.append(abs(ours_m - d_survey[p]) * 1000)
        if p in tape:
            row["tape_minus_survey_mm"] = round((tape[p] - d_survey[p]) * 1000, 1)
            tape_vs_survey.append(abs(tape[p] - d_survey[p]) * 1000)
        pair_rows.append(row)

    result = {
        "field": cfg.field,
        "plot": cfg.plot,
        "shared_markers": codes,
        "scale_umeyama": s,
        "scale_distance_ratio_median": scale_ratio,
        "scale_ratio_cv": ratio_spread,
        "umeyama_residual_mm": {str(c): round(float(r), 2) for c, r in zip(codes, resid_mm)},
        "umeyama_rms_mm": round(rms_mm, 2),
        "ours_vs_survey_mean_abs_mm": round(float(np.mean(ours_vs_survey)), 2),
        "tape_vs_survey_mean_abs_mm": round(float(np.mean(tape_vs_survey)), 2) if tape_vs_survey else None,
        "n_tape_pairs": len(tape_vs_survey),
        "quality_guard": qreport,
        "pairs": pair_rows,
    }
    out_path = os.path.join(src, cfg.output_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=1)

    # --- console summary ---
    print("\n" + "=" * 70)
    print(f"  METRIC SCALE  {cfg.field}/{cfg.plot}   (shared markers: {len(codes)})")
    print("=" * 70)
    print(f"  scale (Umeyama)        : {s:.6f}  m / colmap-unit")
    print(f"  scale (distance ratio) : {scale_ratio:.6f}  (CV {ratio_spread*100:.2f}%)")
    print(f"  Umeyama fit RMS        : {rms_mm:.2f} mm   (per-marker: "
          + ", ".join(f'{c}:{r:.1f}' for c, r in zip(codes, resid_mm)) + ")")
    print(f"  ours vs survey (dist)  : {np.mean(ours_vs_survey):.2f} mm mean-abs over 15 pairs")
    if tape_vs_survey:
        print(f"  tape vs survey (dist)  : {np.mean(tape_vs_survey):.2f} mm mean-abs over "
              f"{len(tape_vs_survey)} pairs  [independent check]")
    print("-" * 70)
    print(f"  {'pair':>12}  {'ours_m':>8}  {'survey_m':>9}  {'tape_m':>8}  {'o-s_mm':>7}  {'t-s_mm':>7}")
    for r in pair_rows:
        print(f"  {str(tuple(r['pair'])):>12}  {r['ours_m']:>8.4f}  {r['survey_m']:>9.4f}  "
              f"{(f'{r['tape_m']:.4f}' if r['tape_m'] is not None else '   -   '):>8}  "
              f"{r['ours_minus_survey_mm']:>7.1f}  "
              f"{(f'{r['tape_minus_survey_mm']:.1f}' if 'tape_minus_survey_mm' in r else '   -  '):>7}")
    print("=" * 70)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
