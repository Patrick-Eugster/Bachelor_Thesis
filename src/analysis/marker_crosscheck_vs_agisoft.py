"""Cross-checks our triangulated 3D markers (pinhole/opencv/radial arms) against the Agisoft arm to
find markers that are 'hard wrong' in our reconstructions. Each arm lives in its own frame/scale, so we
can't compare raw xyz: instead we compare the scale-aligned pairwise-distance geometry (a marker's
distances to the other markers), which is frame-independent. Flags two things per arm:
  - INTERNAL-OUTLIER: a marker whose median distance to the others is >3x the global median (a blown-up
    triangulation, e.g. a marker landing 30 units away from the plot).
  - VS-AGISOFT: after scaling our distances to Agisoft's (median ratio), any marker whose relative
    distance error exceeds a threshold (default 15%).
Run: python src/analysis/marker_crosscheck_vs_agisoft.py   (prints a per-session/arm table).
Writes nothing by default; pass --out <file> to also dump the table.
"""
import json, os, itertools, argparse
import numpy as np

BASE = "input_plots/phone"
SESSIONS = [("field_A", d) for d in ("20250618", "20250627", "20250706", "20250715")] + \
           [("field_D", d) for d in ("20250618", "20250627", "20250706", "20250715")]
# arm name -> variant subfolder ("" = the SIMPLE_PINHOLE baseline, which lives at the session root)
ARMS = {"pinhole": "", "opencv": "opencv", "radial": "radial", "agisoft": "agisoft"}


def load_markers(field, date, sub):
    """Load one arm's triangulated markers as {id: xyz}, or None if that arm has no marker json yet."""
    root = os.path.join(BASE, field, date, sub) if sub else os.path.join(BASE, field, date)
    p = os.path.join(root, "logs", "marker_points3d.json")
    if not os.path.isfile(p):
        return None
    pts = json.load(open(p))["points3d"]
    return {int(k): np.array(v["xyz"], float) for k, v in pts.items()}


def pairwise(mk, ids):
    """Pairwise distances between markers, keyed by a sorted (id,id) tuple."""
    return {tuple(sorted((a, b))): float(np.linalg.norm(mk[a] - mk[b]))
            for a, b in itertools.combinations(ids, 2)}


def check_arm(mk, ref, rel_thresh=0.15):
    """Compare one arm's markers to the Agisoft reference. Returns (scale_to_agisoft, per_marker_max_err,
    internal_outliers, rel_bad). per_marker_max_err[i] = max relative distance error of marker i vs Agisoft
    after scale alignment."""
    ids = sorted(set(mk) & set(ref))
    dm, dr = pairwise(mk, ids), pairwise(ref, ids)
    # internal outlier check (no reference needed): a marker sitting far from all the others
    permed = {i: float(np.median([dm[tuple(sorted((i, j)))] for j in ids if j != i])) for i in ids}
    gmed = float(np.median(list(permed.values())))
    internal_bad = [i for i in ids if permed[i] > 3 * gmed]
    # scale our distances onto Agisoft's, then measure per-marker relative geometry error
    ratios = [dr[k] / dm[k] for k in dm if dm[k] > 1e-9]
    s = float(np.median(ratios)) if ratios else float("nan")
    permaxerr = {}
    for i in ids:
        errs = [abs(s * dm[tuple(sorted((i, j)))] - dr[tuple(sorted((i, j)))]) / dr[tuple(sorted((i, j)))]
                for j in ids if j != i and dr[tuple(sorted((i, j)))] > 1e-9]
        permaxerr[i] = max(errs) if errs else 0.0
    rel_bad = {i: round(permaxerr[i], 2) for i in ids if permaxerr[i] > rel_thresh}
    return ids, s, permaxerr, internal_bad, rel_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="optional file to also write the table to")
    ap.add_argument("--rel_thresh", type=float, default=0.15)
    args = ap.parse_args()
    lines = []
    for field, date in SESSIONS:
        arms = {name: load_markers(field, date, sub) for name, sub in ARMS.items()}
        ref = arms.get("agisoft")
        lines.append(f"\n=== {field}/{date} ===")
        if ref is None:
            lines.append("  no agisoft reference"); continue
        for name in ("pinhole", "opencv", "radial", "agisoft"):
            mk = arms.get(name)
            if mk is None:
                lines.append(f"  {name:8s} (missing)"); continue
            ids, s, permaxerr, internal_bad, rel_bad = check_arm(mk, ref, args.rel_thresh)
            flag = ""
            if internal_bad:
                flag += f"  INTERNAL-OUTLIER={internal_bad}"
            if rel_bad and name != "agisoft":
                flag += f"  VS-AGI>{int(args.rel_thresh*100)}%={rel_bad}"
            worst = max(permaxerr, key=permaxerr.get)
            lines.append(f"  {name:8s} scale2agi={s:.3f} worst={worst}({permaxerr[worst]:.0%}){flag}")
    out = "\n".join(lines)
    print(out)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
