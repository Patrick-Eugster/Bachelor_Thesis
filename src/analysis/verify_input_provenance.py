"""Verify we copied the CORRECT demoanlage images into input_plots/phone/.

The demoanlage source has THREE image variants per session, easy to mix up:
  1. raw/OpenCamera/images/            4032x3024, names WITHOUT _<seq>  = RAW DISTORTED phone stills
  2. processed/colmap/images/          ~3916x2898, names WITH _<seq>    = Agisoft UNDISTORTED
  3. (additions) processed/colmap_distorted/images/  4032x3024, WITH _<seq> = Agisoft DISTORTED-space (marker GT)

Our pipeline expects:
  input_plots/phone/<field>/<date>/input/          <- must equal (1) raw distorted  (COLMAP runs on these)
  input_plots/phone/<field>/<date>/agisoft/images/ <- must equal (2) Agisoft undistorted (benchmark ref)

This script md5-compares our copies against the demoanlage sources, per session, and reports OK / mismatch
/ missing / extra files. md5 = byte identity, so it catches a wrong-folder copy, a partial copy, or an
accidental re-encode. Read-only.

Usage:
  python src/analysis/verify_input_provenance.py
  python src/analysis/verify_input_provenance.py --demo demoanlage2025_v0 --sample 0   # 0 = hash ALL files
  python src/analysis/verify_input_provenance.py --sample 5   # hash only 5 files/folder (fast spot-check)
"""
import os
import glob
import json
import hashlib
import argparse

IMG_EXTS = (".jpg", ".jpeg", ".png")


def md5(path, chunk=1 << 20):
    """Full-file md5 hex digest (streamed so big JPEGs don't blow up RAM)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def folder_files(d):
    """Sorted list of image basenames in a folder, or None if the folder doesn't exist."""
    if not d or not os.path.isdir(d):
        return None
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS))


def find_raw_dir(demo_session):
    """Locate the raw phone stills for a demoanlage session — usually raw/OpenCamera/images,
    but fall back to raw/OpenCamera/*.jpg or raw/*.jpg layouts."""
    cands = [
        os.path.join(demo_session, "raw", "OpenCamera", "images"),
        os.path.join(demo_session, "raw", "OpenCamera"),
        os.path.join(demo_session, "raw"),
    ]
    for c in cands:
        if os.path.isdir(c) and any(f.lower().endswith(IMG_EXTS) for f in os.listdir(c)):
            return c
    return None


def compare(our_dir, src_dir, sample):
    """md5-compare two image folders. Returns a status dict.
    sample>0 hashes only the first `sample` common files (fast); sample==0 hashes all."""
    our_files = folder_files(our_dir)
    src_files = folder_files(src_dir)
    if our_files is None:
        return {"status": "OUR_MISSING", "our_dir": our_dir}
    if src_files is None:
        return {"status": "SRC_MISSING", "src_dir": src_dir}

    our_set, src_set = set(our_files), set(src_files)
    only_our = sorted(our_set - src_set)
    only_src = sorted(src_set - our_set)
    common = sorted(our_set & src_set)

    to_hash = common if sample <= 0 else common[:sample]
    mismatched = []
    for name in to_hash:
        if md5(os.path.join(our_dir, name)) != md5(os.path.join(src_dir, name)):
            mismatched.append(name)

    ok = (not only_our and not only_src and not mismatched)
    return {
        "status": "OK" if ok else "MISMATCH",
        "n_our": len(our_files), "n_src": len(src_files),
        "n_common": len(common), "n_hashed": len(to_hash),
        "only_in_ours": only_our[:10], "only_in_src": only_src[:10],
        "content_mismatched": mismatched[:10],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="input_plots/phone")
    ap.add_argument("--demo", default="demoanlage2025_v0")
    ap.add_argument("--sample", type=int, default=3,
                    help="files to md5 per folder (0 = ALL, thorough but slower)")
    ap.add_argument("--out", default="docs/analysis_results/camera_params/input_provenance.json")
    args = ap.parse_args()

    rows = []
    for sdir in sorted(glob.glob(os.path.join(args.root, "field_*", "*"))):
        if not os.path.isdir(sdir):
            continue
        field = os.path.basename(os.path.dirname(sdir))
        date = os.path.basename(sdir)
        demo_session = os.path.join(args.demo, field, date)

        raw_dir = find_raw_dir(demo_session)
        agi_src = os.path.join(demo_session, "processed", "colmap", "images")

        r_input = compare(os.path.join(sdir, "input"), raw_dir, args.sample)
        r_agi = compare(os.path.join(sdir, "agisoft", "images"), agi_src, args.sample)

        rows.append({"session": f"{field}/{date}", "input_vs_raw": r_input,
                     "agisoft_vs_processed_colmap": r_agi})

        def tag(r):
            s = r["status"]
            if s == "OK":
                return f"OK ({r['n_hashed']}/{r['n_common']} hashed)"
            if s in ("OUR_MISSING", "SRC_MISSING"):
                return s
            bits = []
            if r.get("only_in_ours"): bits.append(f"+{len(r['only_in_ours'])}ours")
            if r.get("only_in_src"): bits.append(f"+{len(r['only_in_src'])}src")
            if r.get("content_mismatched"): bits.append(f"{len(r['content_mismatched'])}diff")
            return "MISMATCH " + ",".join(bits)

        print(f"{field}/{date:>16}   input↔raw: {tag(r_input):28}   agisoft↔processed: {tag(r_agi)}")

    # summary counts
    def count(key, status):
        return sum(1 for r in rows if r[key]["status"] == status)
    n = len(rows)
    print(f"\ninput↔raw:              OK {count('input_vs_raw','OK')}/{n}   "
          f"MISMATCH {count('input_vs_raw','MISMATCH')}   "
          f"our_missing {count('input_vs_raw','OUR_MISSING')}   src_missing {count('input_vs_raw','SRC_MISSING')}")
    print(f"agisoft↔processed:      OK {count('agisoft_vs_processed_colmap','OK')}/{n}   "
          f"MISMATCH {count('agisoft_vs_processed_colmap','MISMATCH')}   "
          f"our_missing {count('agisoft_vs_processed_colmap','OUR_MISSING')}   "
          f"src_missing {count('agisoft_vs_processed_colmap','SRC_MISSING')}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"root": args.root, "demo": args.demo, "sample": args.sample, "sessions": rows}, f, indent=2)
    print(f"\nWrote {args.out}")
    if args.sample > 0:
        print(f"(hashed {args.sample} files/folder — re-run with --sample 0 to md5 EVERY file)")


if __name__ == "__main__":
    main()
