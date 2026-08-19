"""Reads the refreshed phone results.json (metrics rerun on Euler, Aug 2026, now carrying
inner/roi/markers with LPIPS) and tabulates the camera-model arms per session, so the
opencv-vs-agisoft reconstruction question can be answered on a fair region (ROI = plot polygon)
with a uniform LPIPS.

Read-only. Run: python src/analysis/opencv_vs_agisoft_recon.py
"""
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT = os.path.join(REPO, "results", "reconstruction", "phone")

SESSIONS = [("field_A", "20250618"), ("field_A", "20250627"),
            ("field_A", "20250706"), ("field_A", "20250715"),
            ("field_D", "20250618"), ("field_D", "20250627"),
            ("field_D", "20250706"), ("field_D", "20250715")]
ARMS = [("pinhole", ""), ("opencv", "opencv"), ("radial", "radial"), ("agisoft", "agisoft")]


def sl(f, p):
    return f"{f[-1]}/{p[4:]}"


def load(f, p, sub):
    """Latest-iteration metric block for one arm, or None."""
    base = os.path.join(ROOT, f, p, sub, "vanilla_3dgs", "baseline") if sub \
        else os.path.join(ROOT, f, p, "vanilla_3dgs", "baseline")
    rj = os.path.join(base, "results.json")
    if not os.path.exists(rj):
        return None
    d = json.load(open(rj))
    return d[sorted(d)[-1]]


def g(m, region, key):
    """Metric from a region ('' = whole/top-level), or None."""
    if m is None:
        return None
    src = m if region == "" else m.get(region)
    if not src:
        return None
    return src.get(key)


def cell(x, fmt="%.2f"):
    return (fmt % x) if isinstance(x, (int, float)) else " -- "


def main():
    # per region, print opencv vs agisoft (with pinhole/radial for context)
    for region, rlabel in [("", "WHOLE"), ("inner", "INNER"), ("roi", "ROI")]:
        print(f"\n===================== {rlabel}  (PSNR / LPIPS / sharp-ratio) =====================")
        print(f"{'session':8} | {'pinhole':>20} | {'opencv':>20} | {'radial':>20} | {'agisoft':>20}")
        wins_o = wins_a = 0
        for f, p in SESSIONS:
            row = f"{sl(f,p):8} |"
            vals = {}
            for name, sub in ARMS:
                m = load(f, p, sub)
                psnr = g(m, region, "PSNR"); lpips = g(m, region, "LPIPS"); sr = g(m, region, "sharpness_ratio")
                vals[name] = (psnr, lpips)
                row += f" {cell(psnr):>6}/{cell(lpips,'%.3f'):>5}/{cell(sr,'%.2f'):>4} |"
            print(row)
            # opencv vs agisoft tally on PSNR (this region)
            po, pa = vals["opencv"][0], vals["agisoft"][0]
            if isinstance(po, (int, float)) and isinstance(pa, (int, float)):
                if po > pa: wins_o += 1
                elif pa > po: wins_a += 1
        print(f"  [{rlabel} PSNR] opencv wins {wins_o}/8, agisoft wins {wins_a}/8")

    # head-to-head summary on the fair basis: ROI PSNR + ROI LPIPS (lower better)
    print("\n===================== opencv - agisoft  (ROI, the fair region) =====================")
    print(f"{'session':8} | {'dPSNR':>7} | {'dLPIPS':>7}  (dPSNR>0 opencv better; dLPIPS<0 opencv better)")
    for f, p in SESSIONS:
        mo, ma = load(f, p, "opencv"), load(f, p, "agisoft")
        po, pa = g(mo, "roi", "PSNR"), g(ma, "roi", "PSNR")
        lo, la = g(mo, "roi", "LPIPS"), g(ma, "roi", "LPIPS")
        dp = (po - pa) if isinstance(po, (int, float)) and isinstance(pa, (int, float)) else None
        dl = (lo - la) if isinstance(lo, (int, float)) and isinstance(la, (int, float)) else None
        print(f"{sl(f,p):8} | {cell(dp,'%+.2f'):>7} | {cell(dl,'%+.3f'):>7}")
    print("\nNote: D/0706 ROI is marker-unreliable (all arms) -> discount that row.")


if __name__ == "__main__":
    main()
