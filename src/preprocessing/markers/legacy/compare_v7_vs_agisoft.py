"""Compare our marker detector (v7 by default) against Agisoft's marker projections.

The GT is ASYMMETRIC (Agisoft has ~zero false positives but MISSES many visible plates), so:
  - a GT marker we ALSO find            -> HIT   (good)
  - a GT marker we DON'T find           -> MISS  (a real problem: Agisoft's markers are all real)
  - a marker WE find with no GT nearby  -> EXTRA (a CANDIDATE we found that Agisoft missed; NOT
                                                  counted as wrong — confirmed by our decode)

So the headline number is RECALL of Agisoft's markers (hits / GT). Misses are the problem set;
extras are upside (potentially out-recalling Agisoft), never penalised.

Because we also decode an ID, we report, per Agisoft target, which ID(s) we assigned its hits —
a clean detector gives ONE consistent ID per target (and reveals our-id <-> Agisoft-target map).

READ-ONLY. Prints a report, writes a misses CSV, and (optional) overlays for images with misses.

Usage:
  python src/preprocessing/compare_v7_vs_agisoft.py
  python src/preprocessing/compare_v7_vs_agisoft.py --version v7 --tol 25 --overlay
"""
import os
import csv
import json
import math
import argparse
from collections import defaultdict, Counter

import cv2

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def gt_csv(field, plot):
    return os.path.join(REPO, "demoanlage2025_v0", "demoanlage2025_v0_additions",
                        field, plot, "processed", "marker_projections.csv")


def load_gt(field, plot):
    by_cam = defaultdict(list)
    for r in csv.DictReader(open(gt_csv(field, plot))):
        by_cam[r["Camera"]].append((r["Marker"], float(r["X"]), float(r["Y"])))
    return by_cam


def load_dets(field, plot, version):
    p = os.path.join(REPO, "input_plots", "phone", field, plot, "logs",
                     f"marker_detections_{version}.json")
    per = json.load(open(p))["per_image"]
    out = {}
    for cam, lst in per.items():
        name = cam[:-4] if cam.lower().endswith(".jpg") else cam
        out[name] = [(float(d["center"][0]), float(d["center"][1]), d.get("id"))
                     for d in lst] if isinstance(lst, list) else []
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="field_A")
    ap.add_argument("--plot", default="20250609")
    ap.add_argument("--version", default="v7")
    ap.add_argument("--tol", type=float, default=25.0,
                    help="px: a GT marker counts as HIT if a detection is within this distance")
    ap.add_argument("--overlay", action="store_true",
                    help="also write overlays for images that have a MISS (to inspect why)")
    args = ap.parse_args()

    gt = load_gt(args.field, args.plot)
    dets = load_dets(args.field, args.plot, args.version)
    session = os.path.join(REPO, "input_plots", "phone", args.field, args.plot)

    hits = misses = extras = 0
    per_target = defaultdict(lambda: {"hit": 0, "miss": 0, "ids": Counter()})
    per_image = {}      # cam -> {gt, hit, miss, missed_targets[]}
    miss_rows = []
    miss_images = set()

    for cam, gmarks in gt.items():
        dlist = dets.get(cam, [])
        used = set()
        img_hit = img_miss = 0
        img_missed_targets = []
        # each GT marker -> nearest unused detection within tol
        for mk, gx, gy in gmarks:
            best_i, best_d = -1, 1e9
            for i, (dx, dy, _id) in enumerate(dlist):
                if i in used:
                    continue
                dd = math.hypot(dx - gx, dy - gy)
                if dd < best_d:
                    best_d, best_i = dd, i
            if best_i >= 0 and best_d <= args.tol:
                hits += 1
                img_hit += 1
                used.add(best_i)
                per_target[mk]["hit"] += 1
                per_target[mk]["ids"][dlist[best_i][2]] += 1
            else:
                misses += 1
                img_miss += 1
                img_missed_targets.append(mk.replace("target ", "T"))
                per_target[mk]["miss"] += 1
                miss_rows.append((cam, mk, round(gx, 1), round(gy, 1)))
                miss_images.add(cam)
        per_image[cam] = {"gt": len(gmarks), "hit": img_hit, "miss": img_miss,
                          "missed_targets": img_missed_targets}
        extras += len(dlist) - len(used)   # detections not matched to any GT = candidate extras

    n_gt = hits + misses
    print(f"=== {args.version} vs Agisoft GT  ({args.field}/{args.plot}, tol={args.tol:.0f}px) ===")
    print(f"Agisoft GT markers:   {n_gt}")
    print(f"  HIT  (we found):    {hits}  ({100*hits/max(1,n_gt):.0f}%  <- recall of Agisoft)")
    print(f"  MISS (we didn't):   {misses}   <- the problem set")
    print(f"EXTRA (we found, no GT): {extras}   <- candidate markers Agisoft missed (NOT wrong)\n")

    print(f"{'target':>9} {'GTviews':>7} {'hit':>4} {'miss':>4}   our id(s) for its hits")
    for mk in sorted(per_target):
        t = per_target[mk]
        ids = ", ".join(f"{i}×{n}" for i, n in t["ids"].most_common())
        print(f"  {mk:>9} {t['hit']+t['miss']:>7} {t['hit']:>4} {t['miss']:>4}   {ids}")

    # write per-MARKER misses CSV (one row per missed marker)
    out_csv = os.path.join(session, "logs", f"compare_{args.version}_vs_agisoft_misses.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Camera", "Marker", "GT_X", "GT_Y"])
        w.writerows(miss_rows)

    # write per-IMAGE summary CSV in TWO orderings (same data) so it's easy to go through:
    #   _bymiss = most-missed first (triage)   _byname = folder/filename order (walk the folder)
    def write_per_image(path, ordered):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Camera", "gt_markers", "hit", "missed", "missed_targets"])
            for cam, s in ordered:
                w.writerow([cam, s["gt"], s["hit"], s["miss"], "|".join(s["missed_targets"])])

    base = os.path.join(session, "logs", f"compare_{args.version}_vs_agisoft_per_image")
    by_miss = sorted(per_image.items(), key=lambda kv: (-kv[1]["miss"], kv[0]))
    by_name = sorted(per_image.items(), key=lambda kv: kv[0])     # folder/natural order
    write_per_image(base + "_bymiss.csv", by_miss)
    write_per_image(base + "_byname.csv", by_name)

    n_img_with_miss = sum(1 for s in per_image.values() if s["miss"])
    print(f"\nper-marker misses ({len(miss_rows)}) -> {out_csv}")
    print(f"per-image summary -> {base}_bymiss.csv  (most-missed first)")
    print(f"                  -> {base}_byname.csv  (folder order)")
    print(f"images with >=1 miss: {n_img_with_miss}/{len(per_image)}")
    print("top missed images:  ", ", ".join(f"{c}({s['miss']})" for c, s in by_miss[:6] if s["miss"]))

    if args.overlay and miss_images:
        out_dir = os.path.join(session, f"marker_vis_compare_{args.version}")
        os.makedirs(out_dir, exist_ok=True)
        images_dir = os.path.join(session, "images")
        miss_set = {(c, m) for c, m, _, _ in miss_rows}
        for cam in sorted(miss_images):
            img = cv2.imread(os.path.join(images_dir, cam + ".jpg"))
            if img is None:
                continue
            rad = max(16, int(0.014 * max(img.shape[:2])))
            for mk, gx, gy in gt[cam]:
                p = (int(gx), int(gy))
                col = (0, 0, 255) if (cam, mk) in miss_set else (0, 200, 0)  # red miss / green hit
                cv2.circle(img, p, rad, col, 4)
                cv2.putText(img, mk.replace("target ", "T"), (p[0]+rad, p[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
            for dx, dy, _id in dets.get(cam, []):    # our extras in magenta
                if min((math.hypot(dx-gx, dy-gy) for _, gx, gy in gt[cam]), default=999) > args.tol:
                    cv2.circle(img, (int(dx), int(dy)), rad, (255, 0, 255), 4)
            s = 1600 / img.shape[1]
            cv2.imwrite(os.path.join(out_dir, cam + ".jpg"),
                        cv2.resize(img, None, fx=s, fy=s))
        print(f"miss overlays (red=miss, green=hit, magenta=our extra) -> {out_dir}/")


if __name__ == "__main__":
    main()
