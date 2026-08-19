"""Builds the AbsGS vs baseline A/B table on the phone opencv arm, all 8 sessions.
Compares PSNR/SSIM/LPIPS + sharpness_ratio across the 4 metric regions
(whole / inner / roi / markers). Prints per-region tables + mean deltas.
Reproducible source for the thesis absgrad phone A/B."""
import json, os

ROOT = "/workspace/results/reconstruction/phone"
SESSIONS = [("field_A", d) for d in ("20250618","20250627","20250706","20250715")] + \
           [("field_D", d) for d in ("20250618","20250627","20250706","20250715")]
REGIONS = ["whole", "inner", "roi", "markers"]

def load(field, plot, exp):
    """Reads results.json for one experiment; returns the ours_15000 dict."""
    p = os.path.join(ROOT, field, plot, "opencv", "vanilla_3dgs", exp, "results.json")
    with open(p) as fh:
        return json.load(fh)["ours_15000"]

def region_vals(d, region):
    """Pulls (PSNR, SSIM, LPIPS, sharpness_ratio) for a region.
    'whole' lives at the top level; the others are nested sub-dicts."""
    src = d if region == "whole" else d[region]
    return src["PSNR"], src["SSIM"], src["LPIPS"], src["sharpness_ratio"]

def short(field, plot):
    return field[-1] + "/" + plot[4:]

rows = {}  # (field,plot) -> {region: (base tuple, abs tuple)}
for field, plot in SESSIONS:
    base = load(field, plot, "baseline")
    absg = load(field, plot, "absgrad")
    rows[(field, plot)] = {r: (region_vals(base, r), region_vals(absg, r)) for r in REGIONS}

for region in REGIONS:
    print(f"\n===== REGION: {region} =====")
    print(f"{'sess':7} | {'PSNR base→abs Δ':>22} | {'SSIM Δ':>8} | {'LPIPS Δ':>9} | {'sharp base→abs Δ':>22}")
    dP=dS=dL=dR=0.0
    for field, plot in SESSIONS:
        (bP,bS,bL,bR),(aP,aS,aL,aR) = rows[(field,plot)][region]
        dP+=aP-bP; dS+=aS-bS; dL+=aL-bL; dR+=aR-bR
        print(f"{short(field,plot):7} | {bP:6.2f}→{aP:6.2f} {aP-bP:+5.2f} | {aS-bS:+7.3f} | {aL-bL:+8.3f} | {bR:5.3f}→{aR:5.3f} {aR-bR:+6.3f}")
    n=len(SESSIONS)
    print(f"{'MEAN Δ':7} | {'':13}{dP/n:+6.2f} | {dS/n:+7.3f} | {dL/n:+8.3f} | {'':15}{dR/n:+6.3f}")
    up = sum(1 for f,p in SESSIONS if rows[(f,p)][region][1][3] > rows[(f,p)][region][0][3])
    print(f"   sharpness_ratio: absgrad higher on {up}/{n} sessions")
