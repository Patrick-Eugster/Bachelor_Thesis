"""Phase 0 Step B, redone with Agisoft GT crops.

The first Step B fed CCTDecode crops seeded from v6 detections, which are mostly
wrong -> garbage in, garbage out. Now we cut crops at the *correct* positions from
Agisoft's marker_projections.csv (sub-pixel, verified to land on the fiducial), so
the decode core gets correctly-centred real markers. This is the proper test of
"can CCTDecode read a real field marker?".

Bonus: we group the decoded codes BY Agisoft target. If every crop of 'target 6'
decodes to the same code, our decoder is internally CONSISTENT (the property we need)
and we get the Agisoft-target <-> our-code map for free.

READ-ONLY w.r.t. the dataset. Decode overlays + raw crops -> OUT_DIR.

Run:
  python src/preprocessing/test_cct_phase0b_gtcrops.py
  python src/preprocessing/test_cct_phase0b_gtcrops.py --limit 10 --halfwidth 250
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
import csv
import argparse
import shutil
from collections import defaultdict, Counter

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "cctdecode"))
import CCTDecodeRelease as cct  # noqa: E402

N_BITS = 12
COLOR = "black"          # black marks on white plate
CIRC_THRESH = 0.6
# artifact codes: 0 (no bits), 4095 (all 12 bits = solid disk), 2047 (11 bits = disk-ish).
# NOTE: '7' is ALSO a frequent artifact (a single code-arc read as the whole marker), but
# we can't blanket-exclude it because a real ID could legitimately be 7 -> see the
# per-target distribution to spot it instead.
DEGENERATE = {0, 4095, 2047}


def gt_path(field, plot):
    return os.path.join(REPO, "demoanlage2025_v0", "demoanlage2025_v0_additions",
                        field, plot, "processed", "marker_projections.csv")


def classify(good_codes, consensus):
    """Per-crop status vs the target's consensus code, so problems are browsable.
    OK = decoded the consensus; WRONG = decoded a different real code;
    DEGENERATE = only degenerate codes (disk/arc artifacts); NONE = nothing."""
    if consensus is not None and consensus in good_codes:
        return "ok"
    if good_codes:
        return "wrong"
    return "none"   # nothing non-degenerate decoded (covers degenerate-only too)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="field_A")
    ap.add_argument("--plot", default="20250609")
    ap.add_argument("--halfwidth", type=int, default=250,
                    help="crop half-width in px around each GT point")
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    args = ap.parse_args()

    session = os.path.join(REPO, "input_plots", "phone", args.field, args.plot)
    images_dir = os.path.join(session, "images")
    out_dir = os.path.join(session, "marker_vis_cct_phase0b")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    # one subfolder per status so the problem crops are easy to browse by eye
    for sub in ("ok", "wrong", "none"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    csv_path = gt_path(args.field, args.plot)
    by_cam = defaultdict(list)
    for r in csv.DictReader(open(csv_path)):
        by_cam[r["Camera"]].append((r["Marker"], float(r["X"]), float(r["Y"])))
    cams = sorted(by_cam.keys())
    if args.limit:
        cams = cams[:args.limit]

    print(f"GT crops from {csv_path}")
    print(f"crop half-width: {args.halfwidth}px   images: {len(cams)}\n")

    # --- PASS 1: decode every GT crop, remember its result ---
    hw = args.halfwidth
    records = []   # (cam, mk, x, y, codes, good_codes, vis)
    codes_by_target = defaultdict(Counter)
    for cam in cams:
        img = cv2.imread(os.path.join(images_dir, cam + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        for mk, x, y in by_cam[cam]:
            xi, yi = int(round(x)), int(round(y))
            crop = img[max(0, yi - hw):min(H, yi + hw),
                       max(0, xi - hw):min(W, xi + hw)]
            try:
                table, vis = cct.CCT_extract(crop, N_BITS, CIRC_THRESH, COLOR)
            except Exception as e:
                table, vis = [], crop.copy()
                print(f"  {cam} {mk}: decode ERROR {type(e).__name__}")
            codes = [int(c[0]) for c in table]
            good = [c for c in codes if c not in DEGENERATE]
            for c in good:
                codes_by_target[mk][c] += 1
            records.append((cam, mk, x, y, codes, good, vis))

    # consensus code per target = most common non-degenerate
    consensus = {mk: (c.most_common(1)[0][0] if c else None)
                 for mk, c in codes_by_target.items()}

    # --- PASS 2: classify, save crops into status folders, write CSV ---
    rep_csv = os.path.join(out_dir, "decode_report.csv")
    status_count = Counter()
    with open(rep_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Camera", "Marker", "GT_X", "GT_Y", "consensus",
                    "decoded_codes", "status"])
        for cam, mk, x, y, codes, good, vis in records:
            st = classify(good, consensus.get(mk))
            status_count[st] += 1
            w.writerow([cam, mk, f"{x:.1f}", f"{y:.1f}", consensus.get(mk),
                        "|".join(map(str, codes)) or "-", st])
            tag = f"{cam}_{mk.replace(' ', '')}_{('-'.join(map(str, good)) or 'none')}.png"
            cv2.imwrite(os.path.join(out_dir, st, tag), vis)

    n = len(records)
    print(f"GT markers fed: {n}")
    print(f"status: ok={status_count['ok']} ({100*status_count['ok']/max(1,n):.0f}%)  "
          f"wrong={status_count['wrong']}  none={status_count['none']}")
    print(f"\nproblem crops to eyeball (occlusion etc.):")
    print(f"  WRONG (decoded a different code): {out_dir}/wrong/")
    print(f"  NONE  (decoded nothing/degenerate): {out_dir}/none/")
    print(f"per-crop report CSV: {rep_csv}")
    print("\nCONSISTENCY — decoded codes per target (consensus = dominant):")
    for mk in sorted(codes_by_target):
        c = codes_by_target[mk]
        print(f"  {mk:>10}: consensus={consensus[mk]}   all={dict(c)}")


if __name__ == "__main__":
    main()
