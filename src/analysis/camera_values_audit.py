"""Extract EVERY camera value from every phone image and audit its constancy on two axes:
  (A) WITHIN a session  -> is the value the same for all images of that session?
  (B) ACROSS sessions   -> is the value the same from session to session?

Why this matters:
  - GEOMETRIC values (resolution, focal length) MUST be constant within a session for the
    single_camera=true SfM assumption to be valid. If the phone changed focal/zoom mid-capture,
    forcing one shared camera would be wrong and would corrupt COLMAP's geometry.
  - PHOTOGRAPHIC values (shutter, ISO, exposure bias, brightness) are ALLOWED to vary per image
    (auto-exposure) — they change image quality (blur/brightness), NOT the projection geometry,
    so they cannot break the single-camera assumption. We still track them to explain sharpness.
  - ACROSS sessions, focal/model change (Samsung 5.4mm vs Pixel 4.38mm), which is exactly why the
    two 'lisa' Pixel sessions are a different camera and handled separately.

It also pulls the intrinsics COLMAP ESTIMATED (sparse/0/cameras.txt: f, cx, cy) so you can see the
phone-given millimetre focal next to the pixel focal COLMAP solved for.

Outputs (all saved for reproducibility):
  docs/analysis_results/camera_params/camera_values_per_image.csv   <- raw dump, one row per image
  docs/analysis_results/camera_params/CAMERA_VALUES_AUDIT.md        <- the given/estimated/forced
                                                                       table + within/across constancy
  docs/analysis_results/camera_params/camera_values_audit.json      <- machine-readable summary

Usage:
  python src/analysis/camera_values_audit.py
"""
import os
import csv
import glob
import json
import argparse
from collections import defaultdict, Counter

# reuse the readers we already wrote (same folder is on sys.path when run as a script)
from camera_params_over_sessions import read_one_image
from check_intrinsics_constant import read_colmap_cameras

# the per-image fields we dump + how we classify them for the report
GEOMETRIC = ["resolution", "focal_mm", "fnumber"]          # must be constant within a session
PHOTOGRAPHIC = ["iso", "exposure_s", "exposure_bias", "white_balance", "brightness", "pitch"]
META = ["model", "make", "focal_35mm"]                      # identity / derived fields
ALL_FIELDS = GEOMETRIC + PHOTOGRAPHIC + META

GIVEN_FORCED_TABLE = """\
## What was GIVEN by the phone vs. ESTIMATED by COLMAP vs. FORCED constant

The crucial split is **geometric values** (the intrinsics — the only things that can break COLMAP's
single-camera assumption) vs. **photographic settings** (which vary but *cannot* break it).

| Value | From the phone? | Varied within a session? | Who set the final number | Can its variation hurt COLMAP? |
|---|---|---|---|---|
| **Resolution** 4032x3024 | given | no (verified) | phone | — (constant) |
| **Focal length** | given as **5.4 mm** | no — identical on every image (verified) | **COLMAP estimated** the pixel value (~3090 px), seeded from 5.4 mm then refined; **we forced it shared** via `single_camera=true` | **Yes — but only if it had varied. It didn't, so forcing is valid.** |
| **Principal point** `cx,cy` | not in EXIF | — | **COLMAP fixed it at image center** (2016, 1512); by default COLMAP does not refine it | no (not estimated, just centered) |
| **Distortion** | not given | — | **not modeled** (`SIMPLE_PINHOLE` = zero distortion) | no |
| **Shutter / ExposureTime** | given | **yes — varies every shot** (auto-exposure) | phone | **No — not geometry, only blur/brightness** |
| **ISO** | given | no (pinned at 50 on Samsung) | phone | no |
| **Aperture** f/1.8 | given | no (fixed lens) | phone | no |
| **Exposure bias / 35mm-equiv** | given | minor flip | phone | no |

**Bottom line:** the one value we both *estimated* and *forced constant* is the **focal length in
pixels**. Forcing it is legitimate because the phone genuinely held focal fixed (5.4 mm everywhere) —
proven by `check_intrinsics_constant.py`. Everything the phone *did* change during a session (shutter,
exposure) is non-geometric and harmless to COLMAP; the only indirect effect is that a slow shutter can
blur a frame, which weakens feature matching (a sharpness issue, not a geometry issue).
"""


def val_str(v):
    """Stringify a value for grouping (round floats so 5.40000001 == 5.4)."""
    if isinstance(v, float):
        return f"{round(v, 4)}"
    return str(v)


def flatten(im):
    """Turn one read_one_image dict into the flat field->value map we audit."""
    return {
        "resolution": f"{im['width']}x{im['height']}",
        "focal_mm": im["focal_mm"],
        "fnumber": im["fnumber"],
        "iso": im["iso"],
        "exposure_s": im["exposure_s"],
        "exposure_bias": im["exposure_bias"],
        "white_balance": im["white_balance"],
        "brightness": im["brightness"],
        "pitch": im["pitch"],
        "model": im["model"],
        "make": im["make"],
        "focal_35mm": im["focal_35mm"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="input_plots/phone")
    ap.add_argument("--subdir", default="input")
    ap.add_argument("--out", default="docs/analysis_results/camera_params")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "camera_values_per_image.csv")
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["field", "date", "image"] + ALL_FIELDS +
                    ["colmap_model", "colmap_f_px", "colmap_cx", "colmap_cy", "colmap_n_cams"])

    # session -> {field -> Counter(values)}   and remember colmap intrinsics per session
    within = {}
    colmap_intr = {}
    session_order = []

    for sdir in sorted(glob.glob(os.path.join(args.root, "field_*", "*"))):
        if not os.path.isdir(sdir):
            continue
        field = os.path.basename(os.path.dirname(sdir))
        date = os.path.basename(sdir)
        jpgs = sorted(glob.glob(os.path.join(sdir, args.subdir, "*.jpg")))
        if not jpgs:
            continue
        key = f"{field}/{date}"
        session_order.append(key)

        cams = read_colmap_cameras(os.path.join(sdir, "sparse", "0"))
        if cams:
            cid, model, w, h, params = cams[0]
            colmap_intr[key] = {"model": model, "f_px": round(params[0], 2),
                                "cx": round(params[1], 1), "cy": round(params[2], 1),
                                "n_cams": len(cams)}
        else:
            colmap_intr[key] = {"model": None, "f_px": None, "cx": None, "cy": None, "n_cams": 0}
        ci = colmap_intr[key]

        counters = {fld: Counter() for fld in ALL_FIELDS}
        for p in jpgs:
            try:
                flat = flatten(read_one_image(p))
            except Exception as e:
                print(f"  WARN {p}: {e}")
                continue
            for fld in ALL_FIELDS:
                counters[fld][val_str(flat[fld])] += 1
            writer.writerow([field, date, os.path.basename(p)] +
                            [flat[fld] for fld in ALL_FIELDS] +
                            [ci["model"], ci["f_px"], ci["cx"], ci["cy"], ci["n_cams"]])
        within[key] = counters

    csv_f.close()

    # --- WITHIN-session constancy: for each field, how many sessions had it constant? ---
    within_summary = {}
    for fld in ALL_FIELDS:
        constant_sessions = []
        varying_sessions = {}
        for key in session_order:
            c = within[key][fld]
            if len(c) <= 1:
                constant_sessions.append(key)
            else:
                varying_sessions[key] = dict(c)
        within_summary[fld] = {
            "constant_in": len(constant_sessions),
            "total": len(session_order),
            "varying_sessions": varying_sessions,
        }

    # --- ACROSS-session constancy: collect each session's value-set, check if all sessions agree ---
    across_summary = {}
    for fld in ALL_FIELDS:
        per_session_values = {}
        for key in session_order:
            # a session's "value" for a field = its set of distinct values (usually size 1)
            per_session_values[key] = sorted(within[key][fld].keys())
        # distinct across sessions = union of all the (frozen) value-sets
        distinct = sorted({tuple(v) for v in per_session_values.values()})
        across_summary[fld] = {
            "constant_across_sessions": len(distinct) == 1,
            "distinct_value_sets": [list(d) for d in distinct],
            "per_session": per_session_values,
        }

    # ---- write JSON ----
    json_path = os.path.join(args.out, "camera_values_audit.json")
    with open(json_path, "w") as f:
        json.dump({
            "n_sessions": len(session_order),
            "sessions": session_order,
            "colmap_intrinsics": colmap_intr,
            "within_session": within_summary,
            "across_session": across_summary,
        }, f, indent=2)

    # ---- write MD ----
    md_path = os.path.join(args.out, "CAMERA_VALUES_AUDIT.md")
    with open(md_path, "w") as f:
        f.write("# Camera values audit — within-session & across-session constancy\n\n")
        f.write(f"Source: EXIF of every raw `{args.subdir}/*.jpg` over {len(session_order)} phone "
                f"sessions + COLMAP `sparse/0/cameras.txt`. Generated by `camera_values_audit.py`. "
                f"Raw per-image dump: `camera_values_per_image.csv`.\n\n")

        f.write(GIVEN_FORCED_TABLE + "\n")

        f.write("## (A) Constant WITHIN a session (all images of one session equal)?\n\n")
        f.write("*Geometric fields must be constant here for `single_camera` to be valid.*\n\n")
        f.write("| field | class | constant in | notes |\n|---|---|---|---|\n")
        for fld in ALL_FIELDS:
            cls = ("geometric" if fld in GEOMETRIC else
                   "photographic" if fld in PHOTOGRAPHIC else "meta")
            s = within_summary[fld]
            note = "always constant" if not s["varying_sessions"] else \
                   f"varies in {len(s['varying_sessions'])} session(s): " + \
                   ", ".join(s["varying_sessions"].keys())
            f.write(f"| {fld} | {cls} | {s['constant_in']}/{s['total']} | {note} |\n")

        f.write("\n## (B) Constant ACROSS sessions (session to session equal)?\n\n")
        f.write("| field | class | constant across sessions? | distinct values seen |\n|---|---|---|---|\n")
        for fld in ALL_FIELDS:
            cls = ("geometric" if fld in GEOMETRIC else
                   "photographic" if fld in PHOTOGRAPHIC else "meta")
            a = across_summary[fld]
            vals = a["distinct_value_sets"]
            # compact preview (flatten single-element sets)
            preview = "; ".join("/".join(v) if len(v) > 1 else (v[0] if v else "?") for v in vals)
            if len(preview) > 90:
                preview = preview[:87] + "..."
            f.write(f"| {fld} | {cls} | {'YES' if a['constant_across_sessions'] else 'no'} | {preview} |\n")

        f.write("\n## COLMAP-estimated intrinsics per session (pixels)\n\n")
        f.write("*`f_px` is what COLMAP solved for (seeded from the phone's 5.4 mm); "
                "`cx,cy` are fixed at image center; `n_cams`=1 confirms one shared camera.*\n\n")
        f.write("| session | model | f_px | cx | cy | n_cams |\n|---|---|--:|--:|--:|--:|\n")
        for key in session_order:
            ci = colmap_intr[key]
            f.write(f"| {key} | {ci['model']} | {ci['f_px']} | {ci['cx']} | {ci['cy']} | {ci['n_cams']} |\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    # quick console recap
    print("\nWITHIN-session constancy (constant_in / total):")
    for fld in ALL_FIELDS:
        s = within_summary[fld]
        print(f"  {fld:14} {s['constant_in']}/{s['total']}")
    print("\nACROSS-session constant?:")
    for fld in ALL_FIELDS:
        print(f"  {fld:14} {'YES' if across_summary[fld]['constant_across_sessions'] else 'no'}")


if __name__ == "__main__":
    main()
