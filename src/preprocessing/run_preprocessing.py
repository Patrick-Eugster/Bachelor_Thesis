"""Orchestrator for the phone-image preprocessing pipeline.

Base SfM steps (always available, toggle to skip any):
    1. preprocess_uniform_size.py — center-crop to one resolution (or symlink if already uniform)
    2. run_colmap.py             — COLMAP SfM: feature extraction → matching → mapper → undistortion
    3. compare_to_agisoft.py     — per-camera translation/rotation error vs Agisoft reference (optional)

Marker layer (steps 4-8, only when run_markers=true; each is fail-soft — a missing survey file or
local-only pycolmap warns and continues instead of aborting). These live in the markers/ subpackage:
    4. markers/detect_markers_v8_cct.py  — CCT decode per image + manifest filter
    5. markers/triangulate_markers.py    — lift detections to one 3D point per marker
    6. markers/marker_scale.py           — metric scale: survey XYZ or tape (marker_scale_source)
    7. markers/apply_metric_transform.py — Flavour 1: similarity → metric model (same scale source)
    8. markers/marker_gcp_ba.py          — Flavour 2: GCP-constrained BA (pycolmap)   (survey only; skipped in tape mode)

Each step is launched as a subprocess so it can be run individually too. Hydra config:
configs/preprocessing/config.yaml.

Typical usage:
    python src/preprocessing/run_preprocessing.py field=field_D plot=20250523
    python src/preprocessing/run_preprocessing.py field=field_A plot=20250618 run_compare=true
    python src/preprocessing/run_preprocessing.py plot=20250523 run_uniform=false   # already uniform/symlinked
    python src/preprocessing/run_preprocessing.py field=field_A plot=20250609 run_markers=true   # full marker layer
"""

import json
import os
import shutil
import subprocess
import sys
import time

import hydra
from omegaconf import DictConfig, OmegaConf


# Folders that COLMAP creates inside {source_path}/ — wiped when clean_before_colmap=true.
# We deliberately do NOT touch input/, input_uniform/, agisoft/, video/, logs/, or summary JSONs
# (the JSONs live under logs/ which is preserved).
_COLMAP_OUTPUT_FOLDERS = ("distorted", "sparse", "images", "stereo")


def _clean_colmap_outputs(source_path):
    """Remove COLMAP output folders so a re-run starts from scratch. Prints what was removed.
    Leaves input/, input_uniform/, agisoft/, video/, logs/ alone."""
    removed = []
    for name in _COLMAP_OUTPUT_FOLDERS:
        path = os.path.join(source_path, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
            removed.append(name)
    # also remove stale per-step summaries from previous run so we don't accidentally re-read them
    for fname in ("colmap_summary.json", "compare_summary.json"):
        p = os.path.join(source_path, "logs", fname)
        if os.path.isfile(p):
            os.remove(p)
    if removed:
        print(f"[clean] removed {source_path}/{{{','.join(removed)}}}/")
    else:
        print(f"[clean] nothing to remove in {source_path}")


def _read_summary(source_path, filename):
    """Read one of the per-step summary JSONs that each child script drops in {source_path}/logs/.
    Returns None if missing (e.g. step was skipped or crashed)."""
    path = os.path.join(source_path, "logs", filename)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _tape_gate(cfg, source_path, read_summary):
    """Decide whether to run Flavour 2 (GCP-BA) for this field, from the Step-3 tape↔survey agreement.
    Returns True = run GCP-BA (survey trusted), False = skip it (survey suspect, keep Flavour 1).
    When tape_gate is off, always True (current behaviour). Writes logs/metric_choice.json so a
    downstream consumer (e.g. feeding 3DGS) knows which metric model was chosen and why."""
    if not cfg.get("tape_gate", False):
        return True
    sc = read_summary(source_path, "marker_scale.json")
    tvs = sc.get("tape_vs_survey_mean_abs_mm") if sc else None
    thr = float(cfg.get("tape_gate_threshold_mm", 12.0))
    if tvs is None:                       # no tape → can't gate, don't block
        print(f"\n[tape gate] enabled but no tape↔survey data → not gating (running GCP-BA).")
        return True
    good = float(tvs) <= thr
    chosen = "sparse_metric_gcp" if good else "sparse_metric"
    print(f"\n[tape gate] tape↔survey {tvs:.1f} mm {'<=' if good else '>'} {thr:.1f} mm "
          f"→ survey {'GOOD' if good else 'SUSPECT'} → chosen metric model: {chosen}/")
    choice = {"chosen_model": chosen, "survey_quality": "good" if good else "suspect",
              "tape_vs_survey_mm": round(float(tvs), 2), "threshold_mm": thr, "ran_gcp_ba": bool(good)}
    try:
        json.dump(choice, open(os.path.join(source_path, "logs", "metric_choice.json"), "w"), indent=1)
    except OSError:
        pass
    return good


def _count_quality_markers(source_path, read_summary, cfg):
    """How many markers pass the QUALITY guard (parallax / inlier-views / reproj) — the ones that can
    actually anchor the metric scale. Reads marker_points3d.json and applies the exact same gate as
    markers/marker_scale.py (single source of truth), so the failsafe doesn't green-light metric steps on weak
    markers that would poison the scale. Returns (n_quality, weak) where weak = [(code, reasons)].
    (0, []) if the file is missing (triangulation skipped or failed)."""
    pts = read_summary(source_path, "marker_points3d.json")
    if not pts:
        return 0, []
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from markers import marker_scale  # markers/ subpackage; reuse the guard so thresholds + logic never drift
    min_par, min_inl, max_rep = marker_scale.quality_thresholds(cfg)
    n_quality, weak = 0, []
    for code, v in pts.get("points3d", {}).items():
        if not v:
            continue
        ok, reasons = marker_scale.marker_quality_ok(v, min_par, min_inl, max_rep)
        if ok:
            n_quality += 1
        else:
            weak.append((code, reasons))
    return n_quality, weak


def _emit_marker_block(source_path, read_summary):
    """Re-print the headline marker numbers at the very end (scale, RMS, GCP residual).
    Each value is pulled from the marker scripts' own JSONs and shown only if that step ran."""
    pts   = read_summary(source_path, "marker_points3d.json")
    scale = read_summary(source_path, "marker_scale.json")
    gcp   = read_summary(source_path, "metric_gcp_ba.json")
    if not any((pts, scale, gcp)):
        print("\n[marker summary missing — marker steps may have been skipped or failed]")
        return
    print("\n" + "="*50)
    print("      MARKER LAYER SUMMARY")
    print("="*50)
    if pts:
        solved = sum(1 for v in pts.get("points3d", {}).values() if v)
        print(f"{'Markers solved (3D):':<28} {solved} / {len(pts.get('manifest', []))}")
        print(f"{'Snapped / reprojected:':<28} {pts.get('n_snapped', '-')} / {pts.get('n_reprojected', '-')}")
    if scale:
        # tape-mode marker_scale.json has scale_metric / ours_vs_tape_*; survey-mode has
        # scale_umeyama / umeyama_rms_mm / ours_vs_survey_*. Read whichever applies.
        ssrc = scale.get("scale_source", "survey")
        sval = scale.get("scale_metric", scale.get("scale_umeyama"))
        print(f"{'Metric scale (m/unit):':<28} {sval:.4f}  (CV {100*scale.get('scale_ratio_cv', 0):.2f}%)  [{ssrc}]")
        if ssrc == "tape":
            print(f"{'Ours vs tape (dist):':<28} {scale.get('ours_vs_tape_mean_abs_mm', float('nan')):.1f} mm")
        else:
            print(f"{'Umeyama RMS vs survey:':<28} {scale['umeyama_rms_mm']:.1f} mm")
            print(f"{'Ours vs survey (dist):':<28} {scale['ours_vs_survey_mean_abs_mm']:.1f} mm")
    if gcp:
        mb, ma = gcp["marker_reproj_px_before"], gcp["marker_reproj_px_after"]
        print(f"{'GCP marker reproj (px):':<28} {mb['mean']:.1f} → {ma['mean']:.1f}  (scene {gcp['scene_reproj_px_after']:.2f})")
    choice = read_summary(source_path, "metric_choice.json")
    if choice:
        print(f"{'Tape gate → chosen model:':<28} {choice['chosen_model']}/  "
              f"(survey {choice['survey_quality']}, tape↔survey {choice['tape_vs_survey_mm']} mm)")
    print("="*50)


def fmt_time(seconds):
    """Format seconds into h:mm:ss string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def run_step(name, cmd, timings, fatal=True):
    """Run one pipeline step as a subprocess; returns True on success.
    fatal=True aborts the whole pipeline on failure (base SfM steps); fatal=False just warns and
    continues (marker steps — a missing survey file or local-only pycolmap shouldn't kill the run)."""
    print(f"\n{'='*60}\n  STEP: {name}\n{'='*60}")
    print(f">>> {' '.join(cmd)}\n")
    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0
    timings[name] = elapsed
    print(f"\n>>> {name} finished in {fmt_time(elapsed)}")
    if result.returncode != 0:
        if fatal:
            print(f"!!! ERROR: {name} exited with code {result.returncode}. Aborting pipeline.")
            sys.exit(result.returncode)
        print(f"!!! WARNING: {name} exited with code {result.returncode}. "
              f"Continuing (marker step is non-fatal).")
        return False
    return True


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing/config")
def main(cfg: DictConfig):
    """Drive the three preprocessing scripts based on cfg toggles. Each step shares field=...
    and plot=... so the user only sets them once at the orchestrator level."""
    print("--- preprocessing orchestrator config ---")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------------------------------")

    common_args = [f"field={cfg.field}", f"plot={cfg.plot}"]
    timings = {}
    t_start = time.perf_counter()

    # str() because numeric-looking plot names (e.g. 20250523) get parsed by Hydra/YAML as int
    source_path = os.path.join(cfg.dataset.input_dir, str(cfg.field), str(cfg.plot))

    # optional cleanup: nuke previous COLMAP output before step 2 so re-runs start fresh
    if cfg.run_colmap and cfg.get("clean_before_colmap", False):
        print(f"\n[clean_before_colmap=true] cleaning previous COLMAP output...")
        _clean_colmap_outputs(source_path)

    if cfg.run_uniform:
        run_step("1. uniform-size", [
            "python", "src/preprocessing/preprocess_uniform_size.py", *common_args,
        ], timings)

    # ROUTE 2 (markers_in_sfm=true): detect markers on the DISTORTED input_uniform — the space the
    # database keypoints live in — BEFORE colmap, then have run_colmap inject them between matcher and
    # mapper so the second SfM bakes markers in (survey-free). Needs run_uniform to have made
    # input_uniform first. Default off → normal single-SfM run. See docs/preprocessing/markers/MARKER_INTEGRATION_PLAN.md.
    inject_arg = []
    if cfg.get("markers_in_sfm", False) and cfg.run_colmap:
        det_json = cfg.get("marker_inputspace_json", "logs/marker_det_inputspace.json")
        run_step("1b. marker detect (input_uniform → SfM injection)", [
            "python", "src/preprocessing/markers/detect_markers_v8_cct.py", *common_args,
            "image_subdir=input_uniform", f"output_json={det_json}",
            "output_vis_dir=marker_vis_inputspace",
        ], timings, fatal=False)
        inject_arg = [f"inject_markers_json={det_json}"]

    if cfg.run_colmap:
        run_step("2. COLMAP SfM" + (" + marker inject" if inject_arg else ""), [
            "python", "src/preprocessing/run_colmap.py", *common_args, *inject_arg,
        ], timings)

    if cfg.run_compare:
        run_step("3. compare to Agisoft", [
            "python", "src/preprocessing/compare_to_agisoft.py", *common_args,
        ], timings)

    # --- marker layer (steps 4-8) — only when run_markers=true; each step is fail-soft so a missing
    # survey file or local-only pycolmap warns and continues instead of aborting the whole run ---
    if cfg.get("run_markers", False):
        if cfg.get("run_marker_detect", True):
            # write the manifest-named JSON that triangulate (step 5) reads by default — detect's own
            # default is logs/marker_detections_v8.json, which wouldn't match the downstream name.
            run_step("4. marker detect (CCT v8)", [
                "python", "src/preprocessing/markers/detect_markers_v8_cct.py", *common_args,
                "output_json=logs/marker_detections_v8_manifest.json",
            ], timings, fatal=False)

        if cfg.get("run_marker_triangulate", True):
            run_step("5. marker triangulate", [
                "python", "src/preprocessing/markers/triangulate_markers.py", *common_args,
            ], timings, fatal=False)

        # FAILSAFE: the metric steps (6-8) fit a similarity to the survey, which needs enough solved
        # 3D markers to be reliable (>= 3 non-degenerate points for a unique scale+R+t; we default to
        # a stricter min_markers). Too few solved → skip 6-8 and keep the model in relative scale,
        # rather than anchor metric size on 1-2 markers and silently produce a wrong metric frame.
        min_markers = int(cfg.get("min_markers", 4))
        n_quality, weak = _count_quality_markers(source_path, _read_summary, cfg)
        if weak:
            print("\n[quality guard] solved but TOO WEAK to anchor scale (excluded from the count): "
                  + "; ".join(f"{c} ({', '.join(r)})" for c, r in weak))
        markers_ok = n_quality >= min_markers
        if not markers_ok:
            print(f"\n!!! MARKER FAILSAFE: only {n_quality} quality marker(s) (need >= {min_markers}). "
                  f"Skipping metric steps 6-8 — reconstruction stays in relative (non-metric) scale.")
        else:
            print(f"\n[marker failsafe] {n_quality} quality markers (>= {min_markers}) — metric steps enabled.")

        # scale source for steps 6+7: 'survey' (Umeyama onto GPS) or 'tape' (tape distances only, no
        # survey/GPS). Passed to both so the report and the applied model agree.
        marker_scale_source = str(cfg.get("marker_scale_source", "survey"))
        scale_arg = [f"scale_source={marker_scale_source}"]

        if markers_ok and cfg.get("run_marker_scale", True):
            run_step("6. marker metric scale", [
                "python", "src/preprocessing/markers/marker_scale.py", *common_args, *scale_arg,
            ], timings, fatal=False)

        if markers_ok and cfg.get("run_marker_metric", True):
            run_step("7. metric model (Flavour 1)", [
                "python", "src/preprocessing/markers/apply_metric_transform.py", *common_args, *scale_arg,
            ], timings, fatal=False)

        # TAPE GATE: auto-decide whether Flavour 2 (GCP-BA) is trustworthy for this field. The LOMO
        # experiment (docs/preprocessing/markers/MARKER_COLMAP_RERUN_EXPERIMENT.md) showed anchoring markers HELPS when the
        # survey is good and HURTS when it's off — and the Step-3 tape↔survey agreement tells us which
        # IN ADVANCE. So: tape agrees (small) → survey trusted → run GCP-BA (Flavour 2 chosen); tape
        # disagrees (large) → survey suspect → skip GCP-BA, keep Flavour 1 as the metric model.
        # Default OFF (run whatever run_marker_gcp says). Writes logs/metric_choice.json for downstream.
        gcp_allowed = _tape_gate(cfg, source_path, _read_summary) if markers_ok else False

        # GCP-BA is survey-ANCHORED (needs the surveyed XYZ as GCPs) → it makes no sense in tape-only
        # mode. Skip it entirely when scale_source=tape, regardless of run_marker_gcp / the tape gate.
        if markers_ok and marker_scale_source == "tape":
            print("  [tape-only] skipping step 8 (GCP-BA) — it is survey-anchored; "
                  "Flavour 1 (sparse_metric/, tape-scaled) is the metric model.")
        elif markers_ok and cfg.get("run_marker_gcp", True):
            if gcp_allowed:
                run_step("8. GCP-BA (Flavour 2)", [
                    "python", "src/preprocessing/markers/marker_gcp_ba.py", *common_args,
                ], timings, fatal=False)
            else:
                print("  [tape gate] skipping step 8 (GCP-BA) — survey suspect; "
                      "keeping Flavour 1 (sparse_metric/) as the metric model.")

    total = time.perf_counter() - t_start

    # Aggregate per-step JSON summaries (each child script drops one in {source_path}/logs/)
    # str() because numeric-looking plot names (e.g. 20250523) are parsed by Hydra/YAML as int
    source_path = os.path.join(cfg.dataset.input_dir, str(cfg.field), str(cfg.plot))
    s_uniform = _read_summary(source_path, "uniform_size_summary.json") if cfg.run_uniform else None
    s_colmap  = _read_summary(source_path, "colmap_summary.json")       if cfg.run_colmap  else None
    s_compare = _read_summary(source_path, "compare_summary.json")      if cfg.run_compare else None

    # Re-print each child script's full summary block at the very end. COLMAP dumps a lot of
    # output during step 2 that buries the step-1 summary, so we reproduce all three blocks
    # back-to-back here in the same boxed format the children used.

    def _emit_uniform_block(s):
        minutes, seconds = divmod(int(s["elapsed_s"]), 60)
        print("\n" + "="*50)
        print("      UNIFORM-SIZE SUMMARY")
        print("="*50)
        print(f"{'Plot:':<28} {s['field']}/{s['plot']}")
        print(f"{'Total images:':<28} {s['n_images']}")
        print(f"{'Source sizes:':<28} {s.get('source_sizes', '-')}")
        print(f"{'Target size:':<28} {s.get('target_size', '-')}")
        print(f"{'Outcome:':<28} {s['mode']}")
        if s.get("n_cropped", 0) > 0 or s.get("n_copied", 0) > 0:
            print(f"{'Copied as-is:':<28} {s.get('n_copied', 0)}")
            print(f"{'Center-cropped:':<28} {s.get('n_cropped', 0)}")
        print("-" * 50)
        print(f"{'TOTAL TIME:':<28} {minutes}m {seconds}s  ({s['elapsed_s']:.1f}s)")
        print("="*50)

    def _emit_colmap_block(s):
        minutes, seconds = divmod(int(s["elapsed_s"]), 60)
        subs = s.get("submodels", [])
        subs_str = ", ".join(f"{x['name']}={x['images']}" for x in subs) if subs else "-"
        print("\n" + "="*50)
        print("      COLMAP SUMMARY")
        print("="*50)
        print(f"{'Plot:':<28} {s['field']}/{s['plot']}")
        print(f"{'Camera model:':<28} {s['camera']}  (single_camera={s['single_camera']})")
        print(f"{'Matcher:':<28} {s['matcher']}")
        print(f"{'GPU enabled (SIFT+match):':<28} {s['gpu']}")
        print(f"{'Threads (feat+match+map):':<28} {s['num_threads']}")
        print("-" * 50)
        print(f"{'Input images:':<28} {s['input_images']}")
        print(f"{'Sub-models from mapper:':<28} {len(subs)}  ({subs_str})")
        print(f"{'Registered in largest:':<28} {s['registered']} / {s['input_images']}")
        if s['input_images'] > 0:
            print(f"{'Registration rate:':<28} {100.0*s['registered']/s['input_images']:.1f}%")
        print("-" * 50)
        print(f"{'TOTAL TIME:':<28} {minutes}m {seconds}s  ({s['elapsed_s']:.0f}s)")
        print("="*50)

    def _emit_compare_block(s):
        minutes, seconds = divmod(int(s["elapsed_s"]), 60)
        print("\n" + "="*50)
        print("      COMPARE_TO_AGISOFT SUMMARY")
        print("="*50)
        print(f"{'Plot:':<28} {s['field']}/{s['plot']}")
        print(f"{'Cameras ours:':<28} {s['n_ours']}")
        print(f"{'Cameras agisoft:':<28} {s['n_agisoft']}")
        print(f"{'Common (matched):':<28} {s['n_common']}")
        print("-" * 50)
        print(f"{'Umeyama scale (our→m):':<28} {s['scale']:.6f}")
        print(f"{'Mean translation err:':<28} {s['mean_trans_mm']:.2f} mm")
        print(f"{'Median translation err:':<28} {s['median_trans_mm']:.2f} mm")
        if s.get("mean_rot_deg") is not None:
            print(f"{'Mean rotation err:':<28} {s['mean_rot_deg']:.3f}°")
            print(f"{'Median rotation err:':<28} {s['median_rot_deg']:.3f}°")
        print("-" * 50)
        print(f"{'TOTAL TIME:':<28} {minutes}m {seconds}s  ({s['elapsed_s']:.1f}s)")
        print("="*50)

    print("\n" + "#"*60)
    print(f"#  RECAP — all per-step summaries re-printed below for {cfg.field}/{cfg.plot}")
    print("#"*60)

    if s_uniform: _emit_uniform_block(s_uniform)
    elif cfg.run_uniform: print("\n[uniform-size summary missing — step may have failed before writing JSON]")

    if s_colmap: _emit_colmap_block(s_colmap)
    elif cfg.run_colmap: print("\n[colmap summary missing — step may have failed before writing JSON]")

    if s_compare: _emit_compare_block(s_compare)
    elif cfg.run_compare: print("\n[compare summary missing — step may have failed before writing JSON]")

    # marker recap — light block (the marker scripts already print their own boxes; this just pulls
    # the headline numbers from their JSONs so the metric result is visible at the very end too)
    if cfg.get("run_markers", False):
        _emit_marker_block(source_path, _read_summary)

    # final pipeline-wide timing table
    print("\n" + "="*60)
    print("      PREPROCESSING PIPELINE — TIMING SUMMARY")
    print("="*60)
    print(f"{'Plot:':<28} {cfg.field}/{cfg.plot}")
    print(f"{'Source path:':<28} {source_path}")
    print("-" * 60)
    for name, t in timings.items():
        print(f"   {name:<28} {fmt_time(t)}")
    print(f"   {'TOTAL':<28} {fmt_time(total)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
