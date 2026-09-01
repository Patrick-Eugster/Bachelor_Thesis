"""Hatched TP/FP/FN overlay for the PHONE 3D-seg binary evaluation, the phone counterpart of
fip_seg_hatch_overlay.py (same style: green solid TP, FP = vertical red hatch, FN = horizontal
blue hatch, so overlapping error regions stay legible).

Difference from the FIP version: the phone winner is an OpenCV-undistort run, so eval_seg_2d never
rendered a binary seg PNG for it (eval_2d is pinhole-only). Instead we binarize the pred straight
from the saved per-camera label map segmentation_3d/<exp>/2DSeg/<stem>.pt (pred = any head id > 0),
exactly as phone_seg_cpu_eval.py scores it, and pair it with the warped opencv GT + opencv RGB frame.

Read-only apart from the PNGs it writes under docs/analysis_results/. Reuses the FIP hatch helper.
Run from repo root.
"""
import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image

from src.analysis.fip_seg_hatch_overlay import C_TP, C_FP, C_FN, ALPHA, DIM, HATCH_PERIOD, HATCH_THICK, OUTLINE, THR


def _id_boundary(lab, pred):
    """boolean mask: pixel is a boundary if any 4-neighbour has a different id AND it is a predicted
    head (lab>0). Outlines each predicted instance plus head-vs-background edges."""
    b = np.zeros(lab.shape, bool)
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:, 1:]  |= lab[:, :-1] != lab[:, 1:]
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    b[1:, :]  |= lab[:-1, :] != lab[1:, :]
    return b & pred


def hatched_overlay_from_labelmap(gt_path, pt_path, rgb_path, outline_pred_ids=False, out_width=None):
    """one hatched TP/FP/FN overlay (BGR uint8) for a phone seg run on one frame.
    Pred comes from the .pt label map (any head id > 0), not a rendered binary PNG.
    If outline_pred_ids: draw a thin white boundary wherever the per-head id changes, so the
    individual predicted heads are visible instead of one merged blob (shows fragmentation).
    out_width: if set, the fill is downscaled to this width FIRST and the outlines are drawn on the
    downscaled image (crisp 1-px lines) instead of at full res then resized (which smears them)."""
    gt = np.array(Image.open(gt_path).convert("L")) >= THR
    lab = torch.load(pt_path, weights_only=True)
    lab = lab.numpy() if hasattr(lab, "numpy") else np.array(lab)
    pred = lab > 0
    if pred.shape != gt.shape:
        pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_NEAREST) > 0

    base = cv2.imread(rgb_path)
    if base.shape[:2] != gt.shape:
        base = cv2.resize(base, (gt.shape[1], gt.shape[0]))
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    out = (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32) * DIM).astype(np.uint8)

    tp = gt & pred
    fp = pred & ~gt
    fn = gt & ~pred

    # pixel eval classes are mutually exclusive (a pixel is exactly one of TP/FP/FN/TN), so there is
    # nothing to overlap -> no hatching needed, just solid-fill each class blended over the dimmed base.
    over = out.copy()
    over[tp] = C_TP
    over[fp] = C_FP
    over[fn] = C_FN
    FILL_ALPHA = 0.85     # stronger than the FIP hatch ALPHA (0.45) — solid classes read better opaque
    out = cv2.addWeighted(over, FILL_ALPHA, out, 1 - FILL_ALPHA, 0)

    # downscale the FILL first (INTER_AREA), so per-head outlines can be drawn crisp on the final size.
    if out_width is not None and out_width != out.shape[1]:
        h = int(round(out_width * out.shape[0] / out.shape[1]))
        out = cv2.resize(out, (out_width, h), interpolation=cv2.INTER_AREA)
        if outline_pred_ids:
            lab = cv2.resize(lab, (out_width, h), interpolation=cv2.INTER_NEAREST)
            pred = lab > 0

    # optional per-head outlines: outlines each predicted instance (+head-vs-background edges), so the
    # merged fill splits back into the individual heads the pipeline actually assigned. Drawn at the
    # final resolution (after any downscale) -> consistent 1-px white lines, no resample smearing.
    if outline_pred_ids:
        out[_id_boundary(lab, pred)] = (255, 255, 255)

    # simple counts so we can label the rows
    return out, dict(tp=int(tp.sum()), fp=int(fp.sum()), fn=int(fn.sum()))


def main():
    """build one hatched overlay per config for the phone scoring view + a vertical stack."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="field_A/20250715")
    ap.add_argument("--variant", default="opencv")
    ap.add_argument("--model", default="baseline")
    ap.add_argument("--stem", default="IMG_20250715_153912")
    ap.add_argument("--outdir", default="docs/analysis_results/phone_A0715_seg_ladder")
    args = ap.parse_args()

    S, V, M = args.session, args.variant, args.model
    # the divergence story in the eval: conf 0.22 floods with heads (lots of red FP) vs the winner
    cfgs = [
        ("conf022", "ocv15k_perhead_sam2_conf022"),
        ("winner",  "ocv15k_perhead_sam2_conf070_iou06"),
    ]
    gt = f"input_plots/phone/{S}/{V}/manual_label/{args.stem}_gt_mask.png"
    rgb = f"input_plots/phone/{S}/{V}/images/{args.stem}.jpg"
    os.makedirs(args.outdir, exist_ok=True)

    imgs = []
    for tag, exp in cfgs:
        pt = (f"results/reconstruction/phone/{S}/{V}/vanilla_3dgs/{M}"
              f"/segmentation_3d/{exp}/2DSeg/{args.stem}.pt")
        if not os.path.exists(pt):
            print("skip (missing):", pt)
            continue
        vis, c = hatched_overlay_from_labelmap(gt, pt, rgb)
        small = cv2.resize(vis, (1300, int(1300 * vis.shape[0] / vis.shape[1])))
        f = os.path.join(args.outdir, f"phone_{tag}_hatch.png")
        cv2.imwrite(f, small)
        imgs.append(small)
        print(f"saved {f} {small.shape[1]}x{small.shape[0]}  TP={c['tp']} FP={c['fp']} FN={c['fn']}")

    if len(imgs) > 1:
        gap = np.full((20, imgs[0].shape[1], 3), 255, np.uint8)
        stack = imgs[0]
        for im in imgs[1:]:
            stack = np.vstack([stack, gap, im])
        fs = os.path.join(args.outdir, "phone_seg_ladder_hatch_stack.png")
        cv2.imwrite(fs, stack)
        print("saved", fs)


if __name__ == "__main__":
    main()
