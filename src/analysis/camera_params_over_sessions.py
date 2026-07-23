"""Audit the phone CAMERA SETTINGS (EXIF) across every capture session.

Why this matters for the thesis:
  - The SfM pipeline assumes ONE shared camera per session (single_camera=true). That only
    holds if focal length / sensor stay constant within a session. This checks it.
  - Two phones were used: most sessions on a Samsung SM-G781B (Galaxy S20 FE, ~5.4mm f/1.8),
    a couple on a Google Pixel 6a (~4.38mm) — different intrinsics, worth flagging per session.
  - Shutter (ExposureTime) + ISO explain capture quality: long shutter or high ISO => motion
    blur / noise, which is the real driver behind the "bad" sessions (e.g. blurry 20250530).

For each session's raw input/*.jpg it reads: Make, Model, resolution, FocalLength,
FocalLengthIn35mmFilm, FNumber, ISO, ExposureTime, ExposureBias, WhiteBalance, BrightnessValue,
and the device Yaw/Pitch/Roll baked into the OpenCamera UserComment. It aggregates per session
(min/median/max where numeric, unique-set otherwise), then reports which params are constant
across the whole dataset vs which vary.

Output (both saved for reproducibility):
  docs/analysis_results/camera_params/camera_params_over_sessions.json   (full per-session data)
  docs/analysis_results/camera_params/camera_params_over_sessions.md     (human-readable table)

Usage:
  python src/analysis/camera_params_over_sessions.py
  python src/analysis/camera_params_over_sessions.py --root input_plots/phone --subdir input
"""
import os
import sys
import glob
import json
import argparse
import statistics
from collections import defaultdict

from PIL import Image, ExifTags

# reverse lookup name -> we read by name so the code is readable
_NAME2TAG = {v: k for k, v in ExifTags.TAGS.items()}


def _get(d, name):
    """Pull an EXIF value out of a tag dict by its human name, or None if absent."""
    tag = _NAME2TAG.get(name)
    return d.get(tag) if tag is not None else None


def parse_yaw_pitch_roll(user_comment):
    """OpenCamera writes 'Yaw:..,Pitch:..,Roll:..' into the EXIF UserComment.
    Returns (yaw, pitch, roll) as floats, or None if it can't be parsed."""
    if not user_comment:
        return None
    # UserComment is often bytes prefixed with an 'ASCII\x00\x00\x00' charset tag
    if isinstance(user_comment, bytes):
        user_comment = user_comment.decode("ascii", "ignore")
    out = {}
    for key in ("Yaw", "Pitch", "Roll"):
        idx = user_comment.find(key + ":")
        if idx < 0:
            return None
        rest = user_comment[idx + len(key) + 1:]
        # value runs until the next comma or end
        num = rest.split(",")[0].strip()
        try:
            out[key] = float(num)
        except ValueError:
            return None
    return (out["Yaw"], out["Pitch"], out["Roll"])


def read_one_image(path):
    """Read the camera settings we care about from a single JPEG's EXIF.
    Returns a flat dict; fields that are missing come back as None."""
    img = Image.open(path)
    w, h = img.size
    exif = img.getexif()
    ifd = exif.get_ifd(0x8769)  # the Exif sub-IFD holds the actual capture settings

    exp = _get(ifd, "ExposureTime")
    ypr = parse_yaw_pitch_roll(_get(ifd, "UserComment"))
    return {
        "make": _get(exif, "Make"),
        "model": _get(exif, "Model"),
        "width": w,
        "height": h,
        "focal_mm": _to_float(_get(ifd, "FocalLength")),
        "focal_35mm": _to_float(_get(ifd, "FocalLengthIn35mmFilm")),
        "fnumber": _to_float(_get(ifd, "FNumber")),
        "iso": _to_int(_get(ifd, "ISOSpeedRatings")),
        "exposure_s": _to_float(exp),
        "exposure_bias": _to_float(_get(ifd, "ExposureBiasValue")),
        "white_balance": _get(ifd, "WhiteBalance"),
        "brightness": _to_float(_get(ifd, "BrightnessValue")),
        "pitch": ypr[1] if ypr else None,  # camera tilt, the interesting one for orbit capture
    }


def _to_float(v):
    """Coerce an EXIF rational/int/str to float, or None."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v):
    """Coerce to int (ISO can arrive as a tuple on some phones), or None."""
    if v is None:
        return None
    if isinstance(v, (tuple, list)):
        v = v[0]
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _stats(values):
    """min / median / max of a list of numbers, skipping None. Returns None if all missing."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {
        "min": round(min(vals), 5),
        "median": round(statistics.median(vals), 5),
        "max": round(max(vals), 5),
    }


def _uniq(values):
    """Sorted unique non-None values with per-value counts, e.g. {'4032x3024': 90}."""
    counts = defaultdict(int)
    for v in values:
        if v is not None:
            counts[v] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def aggregate_session(field, date, images):
    """Roll up all per-image dicts of one session into a compact summary."""
    res = [f"{im['width']}x{im['height']}" for im in images]
    return {
        "field": field,
        "date": date,
        "n_images": len(images),
        "models": _uniq([im["model"] for im in images]),
        "makes": _uniq([im["make"] for im in images]),
        "resolutions": _uniq(res),
        "focal_mm": _stats([im["focal_mm"] for im in images]),
        "focal_35mm": _stats([im["focal_35mm"] for im in images]),
        "fnumber": _uniq([im["fnumber"] for im in images]),
        "iso": _stats([im["iso"] for im in images]),
        "exposure_s": _stats([im["exposure_s"] for im in images]),
        "exposure_bias": _uniq([im["exposure_bias"] for im in images]),
        "white_balance": _uniq([im["white_balance"] for im in images]),
        "brightness": _stats([im["brightness"] for im in images]),
        "pitch_deg": _stats([im["pitch"] for im in images]),
    }


def shutter_str(sec):
    """Pretty-print an exposure time as 1/x s (how camera UIs show it)."""
    if sec is None or sec <= 0:
        return "?"
    return f"1/{round(1/sec)}s" if sec < 1 else f"{sec:.2f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="input_plots/phone",
                    help="phone dataset root holding field_*/DATE/ folders")
    ap.add_argument("--subdir", default="input",
                    help="which image subfolder to read EXIF from (input = raw, has original EXIF)")
    ap.add_argument("--out", default="docs/analysis_results/camera_params",
                    help="output folder for the json + md report")
    args = ap.parse_args()

    session_dirs = sorted(glob.glob(os.path.join(args.root, "field_*", "*")))
    sessions = []
    skipped = []
    for sdir in session_dirs:
        if not os.path.isdir(sdir):
            continue
        field = os.path.basename(os.path.dirname(sdir))
        date = os.path.basename(sdir)
        jpgs = sorted(glob.glob(os.path.join(sdir, args.subdir, "*.jpg")))
        if not jpgs:
            skipped.append(f"{field}/{date}")
            continue
        images = []
        for p in jpgs:
            try:
                images.append(read_one_image(p))
            except Exception as e:
                print(f"  WARN could not read {p}: {e}")
        if images:
            sessions.append(aggregate_session(field, date, images))
            s = sessions[-1]
            print(f"{field}/{date:>16}  n={s['n_images']:>3}  "
                  f"{'/'.join(s['models'].keys())}  "
                  f"foc={s['focal_mm']['median'] if s['focal_mm'] else '?'}mm  "
                  f"ISO {s['iso']['min']}-{s['iso']['max']}  "
                  f"shutter {shutter_str(s['exposure_s']['min'])}-{shutter_str(s['exposure_s']['max'])}")

    # figure out which params are constant across the WHOLE dataset vs which vary
    all_models = set()
    all_res = set()
    all_fnum = set()
    for s in sessions:
        all_models.update(s["models"].keys())
        all_res.update(s["resolutions"].keys())
        all_fnum.update(str(k) for k in s["fnumber"].keys())

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, "camera_params_over_sessions.json")
    with open(json_path, "w") as f:
        json.dump({
            "root": args.root,
            "subdir": args.subdir,
            "n_sessions": len(sessions),
            "skipped_no_images": skipped,
            "dataset_wide": {
                "models": sorted(all_models),
                "resolutions": sorted(all_res),
                "fnumbers": sorted(all_fnum),
            },
            "sessions": sessions,
        }, f, indent=2)

    md_path = os.path.join(args.out, "camera_params_over_sessions.md")
    with open(md_path, "w") as f:
        f.write("# Phone camera settings over sessions\n\n")
        f.write(f"Source: `{args.root}/field_*/DATE/{args.subdir}/*.jpg`  |  "
                f"{len(sessions)} sessions read, {len(skipped)} skipped (no local JPEGs).\n\n")
        f.write("**Dataset-wide:** "
                f"models = {sorted(all_models)}; "
                f"resolutions = {sorted(all_res)}; "
                f"f-numbers = {sorted(all_fnum)}.\n\n")
        if skipped:
            f.write(f"Skipped (agisoft-only / no local input): {', '.join(skipped)}\n\n")
        f.write("| field | date | n | model | res | focal (mm) | f | ISO min–max | "
                "shutter min–max | expo-bias | pitch° med |\n")
        f.write("|---|---|--:|---|---|---|---|---|---|---|--:|\n")
        for s in sessions:
            model = "/".join(k.split()[-1] for k in s["models"].keys())  # short model
            res = "/".join(s["resolutions"].keys())
            foc = f"{s['focal_mm']['min']}-{s['focal_mm']['max']}" if s["focal_mm"] else "?"
            fnum = "/".join(str(k) for k in s["fnumber"].keys())
            iso = f"{s['iso']['min']}-{s['iso']['max']}" if s["iso"] else "?"
            sh = f"{shutter_str(s['exposure_s']['min'])}-{shutter_str(s['exposure_s']['max'])}" if s["exposure_s"] else "?"
            eb = "/".join(str(k) for k in s["exposure_bias"].keys())
            pitch = s["pitch_deg"]["median"] if s["pitch_deg"] else "?"
            f.write(f"| {s['field']} | {s['date']} | {s['n_images']} | {model} | {res} | "
                    f"{foc} | {fnum} | {iso} | {sh} | {eb} | {pitch} |\n")

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Sessions: {len(sessions)}  Skipped: {len(skipped)} -> {skipped}")


if __name__ == "__main__":
    main()
