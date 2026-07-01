"""Diagnose whether a phone session's JPEGs are a MIX of encodings (e.g. HDR vs non-HDR).

A mix matters: different encoders in one session = different image characteristics feeding the
pipeline. The symptom we noticed is libjpeg's "Invalid SOS parameters for sequential JPEG" warning,
which fires when a frame's SOF marker says "sequential/baseline" but its SOS scan header carries
progressive-style parameters. That mismatch is exactly the kind of thing a different (HDR) encoder
produces, so it's a good fingerprint.

For each image this:
  1. parses the raw JPEG markers -> SOF type (baseline / extended-sequential / progressive),
     precision, dimensions, chroma subsampling, and the SOS scan params (Ss,Se,Ah,Al),
  2. computes whether libjpeg WOULD warn (sequential SOF but non-baseline SOS params), and also
     actually captures the C-level warning by decoding the file,
  3. reads a few EXIF fields (Make/Model/Software/DateTime) + scans the raw bytes for HDR hints,
then groups images by their encoding fingerprint and prints how many of each variant exist.

Usage:
  python src/analysis/inspect_jpeg_variants.py input_plots/phone/field_A/20250715/images
"""
import os
import sys
import glob
from collections import defaultdict

import cv2
from PIL import Image, ExifTags

# SOF marker bytes -> human name. The ones that matter: C0 baseline, C1 extended sequential
# (both "sequential" -> SOS must be Ss=0,Se=63,Ah=0,Al=0), C2 progressive (SOS varies legitimately).
SOF_NAMES = {
    0xC0: "baseline-sequential", 0xC1: "extended-sequential", 0xC2: "progressive",
    0xC3: "lossless", 0xC5: "diff-sequential", 0xC6: "diff-progressive",
    0xC7: "diff-lossless", 0xC9: "ext-seq-arith", 0xCA: "progressive-arith",
    0xCB: "lossless-arith",
}
SEQUENTIAL_SOFS = {0xC0, 0xC1, 0xC9}  # frames where SOS params must be the baseline 0/63/0/0


def parse_markers(path):
    """Walk the JPEG segment markers and pull out the SOF (frame) and first SOS (scan) info.
    Returns a dict; on any parse problem returns {'parse_error': ...}."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] != b"\xff\xd8":
        return {"parse_error": "not a JPEG (no SOI)"}

    out = {"sof": None, "precision": None, "w": None, "h": None,
           "subsampling": None, "sos": None}
    i = 2
    n = len(data)
    while i < n - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        # standalone markers with no length payload
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            i += 2
            continue
        if i + 4 > n:
            break
        seg_len = (data[i + 2] << 8) | data[i + 3]
        seg = data[i + 4: i + 2 + seg_len]
        if marker in SOF_NAMES:
            out["sof"] = marker
            out["precision"] = seg[0]
            out["h"] = (seg[1] << 8) | seg[2]
            out["w"] = (seg[3] << 8) | seg[4]
            ncomp = seg[5]
            comps = []
            for c in range(ncomp):
                hv = seg[6 + c * 3 + 1]
                comps.append((hv >> 4, hv & 0x0F))  # (H sampling, V sampling)
            out["subsampling"] = comps[0] if comps else None  # luma sampling factors
        elif marker == 0xDA:  # SOS — record scan params then stop (entropy data follows)
            ns = seg[0]
            p = 1 + ns * 2
            ss, se, ahal = seg[p], seg[p + 1], seg[p + 2]
            out["sos"] = (ss, se, ahal >> 4, ahal & 0x0F)  # (Ss, Se, Ah, Al)
            break
        i += 2 + seg_len
    return out


def would_warn(markers):
    """True if libjpeg would print 'Invalid SOS parameters for sequential JPEG' for this frame:
    a sequential SOF (baseline/extended) whose SOS params aren't the baseline (0,63,0,0)."""
    if markers.get("sof") in SEQUENTIAL_SOFS and markers.get("sos") is not None:
        return markers["sos"] != (0, 63, 0, 0)
    return False


def captures_warning(path):
    """Actually decode the file with cv2 and capture the C-level (libjpeg) stderr, so we KNOW
    whether the warning really fires for this file (not just predicted from the markers)."""
    r, w = os.pipe()
    old = os.dup(2)
    os.dup2(w, 2)
    try:
        img = cv2.imread(path)
        ok = img is not None
    finally:
        sys.stderr.flush()
        os.dup2(old, 2)
        os.close(old)
        os.close(w)
    msg = os.read(r, 1 << 16)
    os.close(r)
    return ok, (b"Invalid SOS" in msg)


def read_exif(path):
    """Pull a few identifying EXIF fields + scan raw bytes for HDR hints (OpenCamera/GCam write
    HDR markers into EXIF Software or XMP)."""
    info = {"Make": "", "Model": "", "Software": "", "DateTime": ""}
    try:
        im = Image.open(path)
        exif = im.getexif()
        name_by_id = {v: k for k, v in ExifTags.TAGS.items()}
        for key in info:
            tag = name_by_id.get(key)
            if tag in exif:
                info[key] = str(exif[tag]).strip()
    except Exception as e:
        info["exif_error"] = str(e)
    with open(path, "rb") as f:
        head = f.read(200000)  # XMP/HDR markers live early in the file
    info["hdr_hint"] = any(t in head for t in (b"HDR", b"hdr", b"HDR+", b"GCamera", b"hdrgm"))
    return info


def main():
    img_dir = sys.argv[1] if len(sys.argv) > 1 else "input_plots/phone/field_A/20250715/images"
    paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    paths = [p for p in paths if p.lower().endswith((".jpg", ".jpeg"))]
    print(f"Inspecting {len(paths)} JPEGs in {img_dir}\n")

    groups = defaultdict(list)
    rows = []
    for p in paths:
        m = parse_markers(p)
        ok, warned = captures_warning(p)
        ex = read_exif(p)
        sof_name = SOF_NAMES.get(m.get("sof"), f"0x{m.get('sof'):02X}" if m.get("sof") else "?")
        # the fingerprint that defines a "variant"
        key = (sof_name, m.get("precision"), (m.get("w"), m.get("h")),
               m.get("subsampling"), m.get("sos"), warned,
               ex.get("Software", ""), ex.get("hdr_hint"))
        groups[key].append(os.path.basename(p))
        rows.append((os.path.basename(p), sof_name, m.get("sos"), warned, ex.get("Software", ""), ex.get("hdr_hint")))

    print("=" * 100)
    print(f"FOUND {len(groups)} DISTINCT ENCODING VARIANT(S):\n")
    for gi, (key, files) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1])), 1):
        sof, prec, wh, sub, sos, warned, software, hdr = key
        print(f"  VARIANT {gi}:  {len(files)} images")
        print(f"     SOF (frame type) : {sof}   precision={prec}-bit   size={wh[0]}x{wh[1]}")
        print(f"     chroma subsample : luma H,V = {sub}  ({'4:2:0' if sub==(2,2) else '4:2:2' if sub==(2,1) else '4:4:4' if sub==(1,1) else sub})")
        print(f"     SOS scan params  : (Ss,Se,Ah,Al)={sos}")
        print(f"     libjpeg warns    : {'YES — Invalid SOS for sequential JPEG' if warned else 'no'}")
        print(f"     EXIF Software    : {software or '(none)'}")
        print(f"     HDR byte-hint    : {hdr}")
        print(f"     e.g. {files[0]}  ...  ({len(files)} total)")
        print()

    n_warn = sum(1 for r in rows if r[3])
    print("=" * 100)
    print(f"SUMMARY: {n_warn}/{len(rows)} images trigger the libjpeg warning.")
    if len(groups) > 1:
        print(">>> This session contains MORE THAN ONE encoding variant — worth knowing.")
    else:
        print(">>> All images share ONE encoding variant.")


if __name__ == "__main__":
    main()
