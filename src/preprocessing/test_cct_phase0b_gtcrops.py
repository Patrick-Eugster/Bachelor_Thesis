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
    os.makedirs(out_dir, exist_ok=True)

    csv_path = gt_path(args.field, args.plot)
    by_cam = defaultdict(list)
    for r in csv.DictReader(open(csv_path)):
        by_cam[r["Camera"]].append((r["Marker"], float(r["X"]), float(r["Y"])))
    cams = sorted(by_cam.keys())
    if args.limit:
        cams = cams[:args.limit]

    print(f"GT crops from {csv_path}")
    print(f"crop half-width: {args.halfwidth}px   images: {len(cams)}\n")

    # --- PASS 1: decode every GT crop, remember the decoded code(s) per marker ---
    hw = args.halfwidth
    records = []   # (cam, mk, x, y, good_codes)
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
                table, _ = cct.CCT_extract(crop, N_BITS, CIRC_THRESH, COLOR)
            except Exception as e:
                table = []
                print(f"  {cam} {mk}: decode ERROR {type(e).__name__}")
            good = [int(c[0]) for c in table if int(c[0]) not in DEGENERATE]
            for c in good:
                codes_by_target[mk][c] += 1
            records.append((cam, mk, x, y, good))

    # each target's "expected" ID = the most common code it decoded (its mode)
    mode = {mk: (c.most_common(1)[0][0] if c else None)
            for mk, c in codes_by_target.items()}
    # a mode shared by >1 target is BOGUS (distinct markers must have distinct IDs)
    mode_owners = defaultdict(list)
    for mk, m in mode.items():
        if m is not None:
            mode_owners[m].append(mk)
    collisions = {m: ts for m, ts in mode_owners.items() if len(ts) > 1}

    # pick the single decoded id we report per marker + an honest status:
    #   unreliable = this target's mode is a COLLISION (shared id) -> no trustworthy id
    #   consistent = decoded this target's unique mode id
    #   inconsistent = decoded a different id
    #   no-decode = decoded nothing
    def reported_id(good, mk):
        m = mode.get(mk)
        if m is not None and m in collisions:
            return (good[0] if good else m), "unreliable"
        if m in good:
            return m, "consistent"
        if good:
            return good[0], "inconsistent"
        return None, "no-decode"

    # --- write ONE flat CSV, sorted by marker then camera (all of a target together) ---
    rep_csv = os.path.join(out_dir, "decode_per_image.csv")
    with open(rep_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Marker", "expected_id", "Camera", "decoded_id", "status"])
        for cam, mk, x, y, good in sorted(records, key=lambda r: (r[1], r[0])):
            rid, st = reported_id(good, mk)
            w.writerow([mk, mode.get(mk), cam, rid if rid is not None else "-", st])

    # --- ONE overlay per image: decoded id drawn on each marker, colour = status ---
    by_cam_rec = defaultdict(list)
    for rec in records:
        by_cam_rec[rec[0]].append(rec)
    COL = {"consistent": (0, 200, 0), "inconsistent": (0, 0, 255),
           "unreliable": (0, 165, 255), "no-decode": (160, 160, 160)}  # green/red/orange/gray
    for cam, recs in by_cam_rec.items():
        img = cv2.imread(os.path.join(images_dir, cam + ".jpg"))
        if img is None:
            continue
        rad = max(16, int(0.014 * max(img.shape[:2])))
        for _, mk, x, y, good in recs:
            rid, st = reported_id(good, mk)
            col = COL[st]
            p = (int(round(x)), int(round(y)))
            cv2.circle(img, p, rad, col, 5)
            txt = f"{mk.replace('target ','T')}={rid if rid is not None else '?'}"
            org = (p[0] + rad + 4, p[1])
            # draw a dark-grey outline first (thicker), then the colour on top -> readable
            cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 50, 50), 9,
                        cv2.LINE_AA)
            cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1.5, col, 4,
                        cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, cam + ".jpg"), img)

    # --- the headline: a 6-line summary that flags the problem automatically ---
    n = len(records)
    n_consistent = sum(1 for _, mk, _, _, g in records if reported_id(g, mk)[1] == "consistent")
    print(f"GT markers decoded: {n}   trustworthy (green): {n_consistent} "
          f"({100*n_consistent/max(1,n):.0f}%)   [only non-colliding IDs count]\n")
    print("PER-TARGET SUMMARY (expected_id = most common decode):")
    print(f"  {'target':>9} {'expected':>9} {'consistent':>11}   note")
    for mk in sorted(codes_by_target):
        c = codes_by_target[mk]
        m = mode[mk]
        same = c[m]
        tot = sum(c.values())
        note = ""
        if m in collisions:
            note = f"!! COLLISION: id {m} also claimed by {[t for t in collisions[m] if t!=mk]}"
        print(f"  {mk:>9} {str(m):>9} {f'{same}/{tot}':>11}   {note}")
    if collisions:
        print("\n  -> colliding IDs are BOGUS: distinct physical markers can't share an ID,")
        print("     so those targets are NOT being decoded reliably by stock CCTDecode.")
    print(f"\noverlays (id drawn on each marker, red=inconsistent): {out_dir}/")
    print(f"flat CSV (grouped by target): {rep_csv}")


if __name__ == "__main__":
    main()
