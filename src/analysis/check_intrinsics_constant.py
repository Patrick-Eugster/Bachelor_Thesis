"""Check whether the CAMERA INTRINSICS are constant within each phone session.

Intrinsics = the internal camera parameters that map a 3D camera-frame point to a 2D pixel:
focal length f (pixels) + principal point (cx, cy)  [+ distortion for richer models].
They are distinct from EXTRINSICS (the per-shot camera position + rotation, which change every image).

Our SfM runs with single_camera=true, which FORCES COLMAP to fit one shared camera for the whole
session. That's only a valid assumption if the phone didn't actually change its optics mid-capture.
This script verifies it at two layers:

  1. EXIF (physical) layer: for every raw input image, are FocalLength(mm), FocalLengthIn35mmFilm,
     FNumber and resolution all IDENTICAL across the session? Any drift here (e.g. the phone
     silently switching lens / digital zoom) would break the single-camera assumption.

  2. COLMAP (calibrated) layer: parse sparse/0/cameras.txt and confirm there is exactly ONE camera
     record (model + f + cx,cy). >1 camera => COLMAP did NOT share intrinsics.

Read-only. Prints a per-session PASS/FAIL table; nothing is written unless --out is given.

Usage:
  python src/analysis/check_intrinsics_constant.py
  python src/analysis/check_intrinsics_constant.py --out docs/analysis_results/camera_params/intrinsics_constant.json
"""
import os
import glob
import json
import argparse
from collections import Counter

from PIL import Image, ExifTags

_NAME2TAG = {v: k for k, v in ExifTags.TAGS.items()}


def _get(d, name):
    """EXIF value by human tag name, or None."""
    tag = _NAME2TAG.get(name)
    return d.get(tag) if tag is not None else None


def exif_intrinsic_fingerprint(path):
    """Return (hard_fp, focal35) for one image.
    hard_fp = the TRUE physical intrinsics that must be constant for single_camera to be valid:
              (resolution, FocalLength_mm, FNumber). We deliberately EXCLUDE FocalLengthIn35mmFilm
              from this — it's a DERIVED field the phone sometimes writes as 0 ('not reported') and
              sometimes as its real value (26 for the S20 FE). That 0/26 flip is a metadata quirk,
              NOT an optics change, so it must not count as an intrinsic difference. We still return
              focal35 separately so the report can note the reporting inconsistency."""
    img = Image.open(path)
    w, h = img.size
    ifd = img.getexif().get_ifd(0x8769)
    focal = _get(ifd, "FocalLength")
    focal35 = _get(ifd, "FocalLengthIn35mmFilm")
    fnum = _get(ifd, "FNumber")
    # round floats so 5.4 vs 5.40000001 don't look different
    hard_fp = (
        f"{w}x{h}",
        round(float(focal), 3) if focal is not None else None,
        round(float(fnum), 3) if fnum is not None else None,
    )
    return hard_fp, (int(focal35) if focal35 is not None else None)


def read_colmap_cameras(sparse_dir):
    """Parse cameras.txt -> list of (cam_id, model, w, h, params). Empty list if missing."""
    path = os.path.join(sparse_dir, "cameras.txt")
    if not os.path.isfile(path):
        return None
    cams = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id, model, w, h = parts[0], parts[1], int(parts[2]), int(parts[3])
            params = [float(x) for x in parts[4:]]
            cams.append((cam_id, model, w, h, params))
    return cams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="input_plots/phone")
    ap.add_argument("--subdir", default="input")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    rows = []
    for sdir in sorted(glob.glob(os.path.join(args.root, "field_*", "*"))):
        if not os.path.isdir(sdir):
            continue
        field = os.path.basename(os.path.dirname(sdir))
        date = os.path.basename(sdir)
        jpgs = sorted(glob.glob(os.path.join(sdir, args.subdir, "*.jpg")))
        if not jpgs:
            continue

        # --- EXIF layer: count distinct HARD intrinsic fingerprints (+ track the 35mm quirk) ---
        fps = Counter()
        focal35s = Counter()
        for p in jpgs:
            try:
                hard_fp, f35 = exif_intrinsic_fingerprint(p)
                fps[hard_fp] += 1
                focal35s[f35] += 1
            except Exception as e:
                print(f"  WARN {p}: {e}")
        exif_constant = (len(fps) == 1)
        focal35_consistent = (len(focal35s) == 1)

        # --- COLMAP layer: how many cameras did SfM fit? ---
        cams = read_colmap_cameras(os.path.join(sdir, "sparse", "0"))
        if cams is None:
            colmap_note = "no cameras.txt"
            colmap_ok = None
        else:
            colmap_ok = (len(cams) == 1)
            if cams:
                cid, model, w, h, params = cams[0]
                colmap_note = f"{len(cams)} cam(s); {model} f={params[0]:.1f} pp=({params[1]:.0f},{params[2]:.0f})"
            else:
                colmap_note = "0 cameras"

        rows.append({
            "field": field, "date": date, "n_images": len(jpgs),
            "exif_constant": exif_constant,
            "exif_fingerprints": {str(k): v for k, v in fps.items()},
            "focal35_consistent": focal35_consistent,
            "focal35_values": {str(k): v for k, v in focal35s.items()},
            "colmap_shared_camera": colmap_ok,
            "colmap_note": colmap_note,
        })

        status = "PASS" if exif_constant else "FAIL"
        cstat = {True: "1cam", False: "MULTI", None: "-"}[colmap_ok]
        fp_str = " | ".join(f"{k}x{v}" for k, v in fps.items())
        # note the harmless 35mm-equiv reporting flip if present
        f35_note = "" if focal35_consistent else f"  [35mm-equiv reported as {dict(focal35s)} — harmless metadata quirk]"
        print(f"{field}/{date:>16}  EXIF {status:4} ({len(fps)} distinct)  COLMAP {cstat:5}  {colmap_note}{f35_note}")
        if not exif_constant:
            print(f"      fingerprints: {fp_str}")

    n = len(rows)
    exif_pass = sum(1 for r in rows if r["exif_constant"])
    colmap_pass = sum(1 for r in rows if r["colmap_shared_camera"])
    print(f"\nEXIF intrinsics constant within session: {exif_pass}/{n}")
    print(f"COLMAP fit a single shared camera:      {colmap_pass}/{n}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"n_sessions": n, "exif_pass": exif_pass,
                       "colmap_pass": colmap_pass, "sessions": rows}, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
