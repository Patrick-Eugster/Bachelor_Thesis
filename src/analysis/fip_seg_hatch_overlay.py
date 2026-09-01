"""Hatched TP/FP/FN overlay for the FIP 3D-seg binary evaluation, in the same style as the
mask-gen figure (green solid TP, FP = vertical red hatch, FN = horizontal blue hatch) so that
overlapping error regions stay legible instead of one color painting over the other.

Binary GT (manual_label/<stem>_gt_mask.png) vs the pipeline's rendered binary seg
(<model>/test/segmentation/<stem>.png), blended over a dimmed copy of the RGB frame.
Read-only apart from the PNGs it writes under docs/analysis_results/. Reused for any plot/view.
"""
import argparse
import os

import cv2
import numpy as np

# BGR, FIP convention: green correct / red FP / blue FN
C_TP, C_FP, C_FN = (0, 180, 0), (0, 0, 255), (255, 0, 0)
ALPHA = 0.45          # TP fill opacity over the photo (lower = greener kept subdued)
DIM = 0.45            # darken the base photo so hatches pop
HATCH_PERIOD = 20     # stripe every N px (at full res) -- larger = more gap between stripes
HATCH_THICK = 5       # stripe thickness in px -- larger = bolder stripes
OUTLINE = 3           # thickness of the colored outline drawn around each error region
THR = 128


def hatched_overlay(gt_path, pred_path, rgb_path):
    """one hatched TP/FP/FN overlay (BGR uint8) for a single config on one frame."""
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) >= THR
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred = pred >= THR

    base = cv2.imread(rgb_path)
    if base.shape[:2] != gt.shape:
        base = cv2.resize(base, (gt.shape[1], gt.shape[0]))
    # desaturate to grayscale so the green wheat canopy does not compete with the TP green;
    # only the overlay colors then carry meaning. dim it so the hatches pop.
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    out = (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32) * DIM).astype(np.uint8)

    tp = gt & pred
    fp = pred & ~gt
    fn = gt & ~pred

    # LAYER ORDER: (1) green TP laid down first, blended over the gray base
    over = out.copy()
    over[tp] = C_TP
    out = cv2.addWeighted(over, ALPHA, out, 1 - ALPHA, 0)

    # (2) then the error hatches painted ON TOP so they are never covered by the green:
    #     FP = vertical red stripes, FN = horizontal blue stripes (crosshatch where both).
    col = (np.arange(gt.shape[1]) % HATCH_PERIOD) < HATCH_THICK
    row = (np.arange(gt.shape[0]) % HATCH_PERIOD) < HATCH_THICK
    out[fp & col[None, :]] = C_FP
    out[fn & row[:, None]] = C_FN

    # (3) a crisp colored outline around every error region so each one reads as the top layer
    #     regardless of hatch density: red around FP blobs, blue around FN blobs.
    for mask, color in ((fp, C_FP), (fn, C_FN)):
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, OUTLINE)
    return out


def main():
    """build one hatched overlay per config for a plot/view and a vertical stack of them."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default="plot_462")
    ap.add_argument("--stem", default="FPWW036_SR0462_1_FIP2_cam_11")
    ap.add_argument("--seg", default="seg_yv5_1280")
    ap.add_argument("--outdir", default="docs/analysis_results/fip462_ladder")
    args = ap.parse_args()

    cfgs = [("fipseg15k_ppoff", "baseline"), ("fipseg15k_pp", "pp"), ("fipseg15k_absgrad", "absgrad")]
    gt = f"input_plots/fip/{args.plot}/manual_label/{args.stem}_gt_mask.png"
    rgb = f"input_plots/fip/{args.plot}/images/{args.stem}.png"
    os.makedirs(args.outdir, exist_ok=True)

    imgs = []
    for exp, tag in cfgs:
        pred = (f"results/reconstruction/fip/{args.plot}/vanilla_3dgs/{exp}"
                f"/test/segmentation/{args.stem}.png")
        vis = hatched_overlay(gt, pred, rgb)
        small = cv2.resize(vis, (1300, int(1300 * vis.shape[0] / vis.shape[1])))
        f = os.path.join(args.outdir, f"{args.plot}_{tag}_hatch.png")
        cv2.imwrite(f, small)
        imgs.append(small)
        print("saved", f, small.shape[1], "x", small.shape[0])

    gap = np.full((20, imgs[0].shape[1], 3), 255, np.uint8)
    stack = imgs[0]
    for im in imgs[1:]:
        stack = np.vstack([stack, gap, im])
    fs = os.path.join(args.outdir, f"{args.plot}_ladder_hatch_stack.png")
    cv2.imwrite(fs, stack)
    print("saved", fs)


if __name__ == "__main__":
    main()
