"""Write a markdown report of SHUTTER SPEED (exposure time) per phone session, with the
'slow shutter -> motion blur' explanation.

Reads the per-session stats produced by camera_params_over_sessions.py (so run that first),
and emits a sorted table + a plain-language note on why shutter speed is the capture-quality
lever that matters for our orbit captures.

Usage:
  python src/analysis/camera_params_over_sessions.py          # produces the json this reads
  python src/analysis/shutter_speed_report.py
"""
import os
import json
import argparse


def shutter_str(sec):
    """Pretty-print an exposure time as 1/x s (how camera UIs show it)."""
    if sec is None or sec <= 0:
        return "?"
    return f"1/{round(1/sec)}s" if sec < 1 else f"{sec:.2f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="docs/analysis_results/camera_params/camera_params_over_sessions.json")
    ap.add_argument("--out", default="docs/analysis_results/camera_params/SHUTTER_SPEED_OVER_SESSIONS.md")
    args = ap.parse_args()

    with open(args.json) as f:
        data = json.load(f)
    sessions = data["sessions"]

    # sort slowest-max-shutter first (highest motion-blur risk at the top)
    def slowest(s):
        return s["exposure_s"]["max"] if s["exposure_s"] else 0
    sessions_sorted = sorted(sessions, key=slowest, reverse=True)

    lines = []
    lines.append("# Shutter speed (exposure time) over phone sessions\n")
    lines.append("Source: EXIF `ExposureTime` of every raw `input/*.jpg`, aggregated by "
                 "`camera_params_over_sessions.py`; formatted here by `shutter_speed_report.py`.\n")
    lines.append("Sorted by **slowest shutter (max exposure time) first** — the top rows carry the "
                 "highest motion-blur risk.\n")

    lines.append("## Why slow shutter = blur\n")
    lines.append(
        "The shutter speed (exposure time) is how long the sensor collects light for one frame. "
        "During that window **any relative motion between the phone and the scene smears across "
        "pixels** — this is motion blur. Two things move in our captures:\n\n"
        "1. **The phone itself** — we hand-hold and walk an orbit around the plot, so the camera is "
        "never perfectly still. The longer the shutter is open, the further the camera travels "
        "during the exposure, and the more the whole image smears.\n"
        "2. **The wheat** — heads and awns sway in wind. Even a perfectly steady camera blurs a "
        "moving head if the shutter is slow.\n\n"
        "The phone runs auto-exposure: in **bright** light it can use a **fast** shutter "
        "(e.g. 1/2000 s) and freeze motion; in **dim** light (overcast, evening, shade) it "
        "**slows** the shutter to gather enough light, which is exactly when blur creeps in. "
        "Because ISO is pinned at 50 and the aperture is fixed (f/1.8), **shutter is the only "
        "exposure lever that moves between our sessions** — so it is the single best EXIF predictor "
        "of capture sharpness.\n\n"
        "**Why blur hurts the pipeline:** SfM (COLMAP) needs crisp, repeatable feature points to "
        "match across views; blur washes them out. 3DGS then has to fit fuzzy evidence, giving a "
        "softer reconstruction, and the 3D segmentation (which matches rendered vs. real masks) "
        "degrades where the model is soft. So a slow-shutter session tends to be a low-sharpness "
        "session — cross-check any suspicious row here against `analyze_sharpness.py`.\n")

    lines.append("## Per-session shutter speed\n")
    lines.append("| field | date | n | camera | shutter fastest | shutter slowest | ISO |")
    lines.append("|---|---|--:|---|---|---|---|")
    for s in sessions_sorted:
        cam = "/".join(k.split()[-1] for k in s["models"].keys())
        exp = s["exposure_s"]
        fast = shutter_str(exp["min"]) if exp else "?"   # min time = fastest
        slow = shutter_str(exp["max"]) if exp else "?"   # max time = slowest
        iso = f"{s['iso']['min']}-{s['iso']['max']}" if s["iso"] else "?"
        lines.append(f"| {s['field']} | {s['date']} | {s['n_images']} | {cam} | {fast} | {slow} | {iso} |")

    lines.append("\n*Fastest = shortest exposure time (least blur); slowest = longest (most blur risk).*\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.out}  ({len(sessions_sorted)} sessions)")


if __name__ == "__main__":
    main()
