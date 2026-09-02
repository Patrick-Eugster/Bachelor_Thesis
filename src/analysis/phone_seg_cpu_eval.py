"""Read-only CPU eval of the EXISTING A/0715 phone 3D-seg runs against their GT masks, so we can compare
seg quality across the runs we already have without re-segging or rendering. For each run it loads the
saved per-camera label map (segmentation_3d/<exp>/2DSeg/<gt_stem>.pt), binarizes it (pred = any head id
> 0), loads the matching GT mask, and computes the same pixel metrics as eval_seg_2d (IoU/P/R/F1). The GT
is variant-specific: pinhole-recon runs score against the pinhole GT, opencv runs against the warped
opencv GT, the agisoft run against the warped agisoft GT (stem has a trailing _25).

STRICTLY READ-ONLY on results/: it only READS the .pt label maps and the GT PNGs and writes a single JSON
to docs/analysis_results/. Nothing under results/ is ever written. A self-check recomputes pin15k/pin30k
and compares to their stored metrics_2d.json (must match ~0) to validate the scorer AND the GT pairing.
Run from repo root."""
import json
import os
import numpy as np
import torch
from PIL import Image

S = "field_A/20250715"
BASE = f"results/reconstruction/phone/{S}"
GT_PIN = f"input_plots/phone/{S}/manual_label/IMG_20250715_153912_gt_mask.png"
GT_OCV = f"input_plots/phone/{S}/opencv/manual_label/IMG_20250715_153912_gt_mask.png"
GT_AGI = f"input_plots/phone/{S}/agisoft/manual_label/IMG_20250715_153912_25_gt_mask.png"
# second session (generalization) — field_D/20250627, its own opencv-warped GT
S_D0627 = "field_D/20250627"
GT_OCV_D0627 = f"input_plots/phone/{S_D0627}/opencv/manual_label/IMG_20250627_102602_gt_mask.png"
OUT = "docs/analysis_results/phone_seg_cpu_eval.json"

# (label, model_path (rel to session base), exp_name, gt_path, gt_stem, mask_tag, sfm, iters[, session])
# session is optional (9th element); when absent it defaults to S (field_A/20250715).
RUNS = [
    ("pin15k",        "vanilla_3dgs/baseline",          "pin15k_yolov5_pertile",             GT_PIN, "IMG_20250715_153912",    "yolov5_fullres+per_tile+ROI", "pinhole", 15000),
    ("pin30k",        "vanilla_3dgs/colmap_dense17k",   "pin30k_yolov5_pertile",             GT_PIN, "IMG_20250715_153912",    "yolov5_fullres+per_tile+ROI", "pinhole", 30000),
    ("ocv15k",        "opencv/vanilla_3dgs/baseline",   "ocv15k_yolov5_pertile",             GT_OCV, "IMG_20250715_153912",    "yolov5_fullres+per_tile+ROI", "opencv",  15000),
    ("ocv30k_a100",   "opencv/vanilla_3dgs/dense17k",   "ocv30k_yolov5_pertile_a100",        GT_OCV, "IMG_20250715_153912",    "yolov5_fullres+per_tile+ROI", "opencv",  30000),
    ("ocv15k_groundfix",       "opencv/vanilla_3dgs/baseline", "ocv15k_groundfix",              GT_OCV, "IMG_20250715_153912", "opt-suite (verify mask src)", "opencv", 15000),
    ("ocv15k_frust_paint",     "opencv/vanilla_3dgs/baseline", "ocv15k_frust_paint_fast",       GT_OCV, "IMG_20250715_153912", "opt-suite (verify mask src)", "opencv", 15000),
    ("ocv15k_roimark",         "opencv/vanilla_3dgs/baseline", "ocv15k_roimark_groundfix",      GT_OCV, "IMG_20250715_153912", "opt-suite (verify mask src)", "opencv", 15000),
    ("ocv15k_roimark_frustum", "opencv/vanilla_3dgs/baseline", "ocv15k_roimark_frustum_groundfix", GT_OCV, "IMG_20250715_153912", "opt-suite (verify mask src)", "opencv", 15000),
    # new full-cull A/B runs (submitted 2026-08-21) — SAM1 per_tile masks, full cull (frustum+roi+marker+tilt)
    ("pin15k_fullcull",    "vanilla_3dgs/baseline",        "pin15k_fullcull",   GT_PIN, "IMG_20250715_153912",    "SAM1 per_tile, full cull (tilt+roi+marker 0.075)", "pinhole", 15000),
    ("ocv15k_markerR065",  "opencv/vanilla_3dgs/baseline", "ocv15k_markerR065", GT_OCV, "IMG_20250715_153912",    "SAM1 per_tile, full cull, marker_radius_rel=0.065", "opencv", 15000),
    ("agi15k",        "agisoft/vanilla_3dgs/baseline",  "agi15k_groundfix",                  GT_AGI, "IMG_20250715_153912_25", "yolov5? (verify)",            "agisoft", 15000),
    # pending Euler runs (submitted 2026-08-21) — show as "missing" until their 2DSeg is rsynced, then auto-score
    ("agi15k_fullcull",          "agisoft/vanilla_3dgs/baseline", "agi15k_fullcull",         GT_AGI, "IMG_20250715_153912_25", "SAM1 per_tile, full cull",              "agisoft", 15000),
    ("ocv15k_conf035",           "opencv/vanilla_3dgs/baseline",  "ocv15k_conf035",          GT_OCV, "IMG_20250715_153912",    "SAM2 per_tile conf0.35, full cull",     "opencv",  15000),
    ("ocv15k_conf040",           "opencv/vanilla_3dgs/baseline",  "ocv15k_conf040",          GT_OCV, "IMG_20250715_153912",    "SAM2 per_tile conf0.40, full cull",     "opencv",  15000),
    ("ocv15k_conf055",           "opencv/vanilla_3dgs/baseline",  "ocv15k_conf055",          GT_OCV, "IMG_20250715_153912",    "SAM2 per_tile conf0.55, full cull",     "opencv",  15000),
    ("ocv15k_conf070",           "opencv/vanilla_3dgs/baseline",  "ocv15k_conf070",          GT_OCV, "IMG_20250715_153912",    "SAM2 per_tile conf0.70, full cull",     "opencv",  15000),
    ("ocv15k_absgrad_fullcull",  "opencv/vanilla_3dgs/absgrad",   "ocv15k_absgrad_fullcull", GT_OCV, "IMG_20250715_153912",    "SAM1 per_tile, full cull, ABSGRAD model","opencv",  15000),
    ("ocv15k_sam1_conf070",      "opencv/vanilla_3dgs/baseline",  "ocv15k_sam1_conf070",     GT_OCV, "IMG_20250715_153912",    "SAM1 per_tile conf0.70, full cull",     "opencv",  15000),
    # per-head granularity A/B (conf 0.35, full cull) — decides per_head SAM1 vs SAM2 (finished 2026-08-22)
    ("ocv15k_perhead_sam1",      "opencv/vanilla_3dgs/baseline",  "ocv15k_perhead_sam1",     GT_OCV, "IMG_20250715_153912",    "SAM1 per_head conf0.35, full cull",     "opencv",  15000),
    ("ocv15k_perhead_sam2",      "opencv/vanilla_3dgs/baseline",  "ocv15k_perhead_sam2",     GT_OCV, "IMG_20250715_153912",    "SAM2 per_head conf0.35, full cull",     "opencv",  15000),
    # per_head SAM2 conf-endpoints (divergence study) — A/0715 conf0.70; conf0.22 pending
    ("ocv15k_perhead_sam2_conf070", "opencv/vanilla_3dgs/baseline", "ocv15k_perhead_sam2_conf070", GT_OCV, "IMG_20250715_153912", "SAM2 per_head conf0.70, full cull", "opencv", 15000),
    # generalization to field_D/20250627 (own opencv-warped GT) — per_head SAM2 conf0.70; conf0.22 pending
    ("D0627_ocv15k_perhead_sam2_conf070", "opencv/vanilla_3dgs/baseline", "ocv15k_perhead_sam2_conf070", GT_OCV_D0627, "IMG_20250627_102602", "SAM2 per_head conf0.70, full cull (D/0627)", "opencv", 15000, S_D0627),
    # conf0.22 endpoint (the mask-optimal conf, other end of the divergence study) — A/0715 done, D/0627 still running
    ("ocv15k_perhead_sam2_conf022", "opencv/vanilla_3dgs/baseline", "ocv15k_perhead_sam2_conf022", GT_OCV, "IMG_20250715_153912", "SAM2 per_head conf0.22, full cull", "opencv", 15000),
    ("D0627_ocv15k_perhead_sam2_conf022", "opencv/vanilla_3dgs/baseline", "ocv15k_perhead_sam2_conf022", GT_OCV_D0627, "IMG_20250627_102602", "SAM2 per_head conf0.22, full cull (D/0627)", "opencv", 15000, S_D0627),
    # IoU-0.6 seg match threshold (reuses conf0.70 masks, only iou_threshold 0.5->0.6) — A/0715 + D/0627
    ("ocv15k_perhead_sam2_conf070_iou06", "opencv/vanilla_3dgs/baseline", "ocv15k_perhead_sam2_conf070_iou06", GT_OCV, "IMG_20250715_153912", "SAM2 per_head conf0.70, full cull, IoU0.6", "opencv", 15000),
    ("D0627_ocv15k_perhead_sam2_conf070_iou06", "opencv/vanilla_3dgs/baseline", "ocv15k_perhead_sam2_conf070_iou06", GT_OCV_D0627, "IMG_20250627_102602", "SAM2 per_head conf0.70, full cull, IoU0.6 (D/0627)", "opencv", 15000, S_D0627),
    # IoU-0.7 seg match threshold (reuses conf0.70 masks, only iou_threshold 0.6->0.7) — A/0715
    ("ocv15k_perhead_sam2_conf070_iou07", "opencv/vanilla_3dgs/baseline", "ocv15k_perhead_sam2_conf070_iou07", GT_OCV, "IMG_20250715_153912", "SAM2 per_head conf0.70, full cull, IoU0.7", "opencv", 15000),
    # AbsGS model on the winner masks (conf0.70 per_head + IoU0.6) — does denser cloud help seg? A/0715 + D/0627
    ("A0715_absgrad_perhead_conf070_iou06", "opencv/vanilla_3dgs/absgrad", "ocv15k_absgrad_perhead_sam2_conf070_iou06", GT_OCV, "IMG_20250715_153912", "SAM2 per_head conf0.70, full cull, IoU0.6, ABSGRAD model", "opencv", 15000),
    ("D0627_absgrad_perhead_conf070_iou06", "opencv/vanilla_3dgs/absgrad", "ocv15k_absgrad_perhead_sam2_conf070_iou06", GT_OCV_D0627, "IMG_20250627_102602", "SAM2 per_head conf0.70, full cull, IoU0.6, ABSGRAD model (D/0627)", "opencv", 15000, S_D0627),
    # plain 30k (dense17k_noabsgrad) on the winner masks — does longer default training help seg? A/0715
    ("A0715_ocv30k_noabsgrad_perhead_conf070_iou06", "opencv/vanilla_3dgs/dense17k_noabsgrad", "ocv30k_noabsgrad_perhead_sam2_conf070_iou06", GT_OCV, "IMG_20250715_153912", "SAM2 per_head conf0.70, full cull, IoU0.6, 30k default", "opencv", 30000),
    # SAHI opt-suite (pinhole recon, SAHI masks) — different mask source, kept for the runtime study
    ("sahi_cropcache",  "vanilla_3dgs/phone_sahi", "seg_cropcache",  GT_PIN, "IMG_20250715_153912", "SAHI (opt-suite)", "pinhole", 15000),
    ("sahi_cull_v3",    "vanilla_3dgs/phone_sahi", "seg_cull_v3",    GT_PIN, "IMG_20250715_153912", "SAHI (opt-suite)", "pinhole", 15000),
    ("sahi_nocull_v3",  "vanilla_3dgs/phone_sahi", "seg_nocull_v3",  GT_PIN, "IMG_20250715_153912", "SAHI (opt-suite)", "pinhole", 15000),
]


def metrics(gt, pred):
    """IoU/precision/recall/F1 over boolean GT vs pred (same definitions as eval_seg_2d)."""
    tp = int((gt & pred).sum()); fp = int((~gt & pred).sum()); fn = int((gt & ~pred).sum())
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(iou=iou, precision=p, recall=r, f1=f1, pred_heads=None)


def main():
    rows = []
    selfcheck = []
    for run in RUNS:
        name, mp, exp, gtp, stem, tag, sfm, iters = run[:8]
        sess = run[8] if len(run) > 8 else S      # optional per-run session override
        base = f"results/reconstruction/phone/{sess}"
        pt = f"{base}/{mp}/segmentation_3d/{exp}/2DSeg/{stem}.pt"
        if not (os.path.exists(pt) and os.path.exists(gtp)):
            rows.append(dict(name=name, status="missing", pt_exists=os.path.exists(pt), gt_exists=os.path.exists(gtp)))
            continue
        lab = torch.load(pt, weights_only=True)
        lab = lab.numpy() if hasattr(lab, "numpy") else np.array(lab)
        gt = np.array(Image.open(gtp).convert("L")) >= 128
        if lab.shape != gt.shape:                       # should not happen (verified equal), guard anyway
            rows.append(dict(name=name, status="shape_mismatch", label_shape=list(lab.shape), gt_shape=list(gt.shape)))
            continue
        pred = lab > 0
        m = metrics(gt, pred)
        m["pred_heads"] = int(len(np.unique(lab)) - (1 if 0 in np.unique(lab) else 0))
        rows.append(dict(name=name, status="ok", sfm=sfm, iters=iters, mask=tag,
                         **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}))
        # self-check against stored official eval for the two pinhole runs
        stored_json = f"{base}/{mp}/segmentation_3d/{exp}/eval_2d/metrics_2d.json"
        if os.path.exists(stored_json):
            st = json.load(open(stored_json)); st = st[0] if isinstance(st, list) else st
            selfcheck.append((name, abs(m["iou"] - st.get("iou", -9)), abs(m["precision"] - st.get("precision", -9))))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(session=S, runs=rows, selfcheck=[(n, round(di, 4), round(dp, 4)) for n, di, dp in selfcheck]),
              open(OUT, "w"), indent=2)

    # print table
    print(f"{'run':<24}{'sfm':<9}{'iters':>6}  {'IoU':>6}{'P':>7}{'R':>7}{'F1':>7}{'heads':>7}   mask")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['name']:<24}  -- {r['status']}"); continue
        print(f"{r['name']:<24}{r['sfm']:<9}{r['iters']:>6}  {r['iou']:>6.3f}{r['precision']:>7.3f}"
              f"{r['recall']:>7.3f}{r['f1']:>7.3f}{r['pred_heads']:>7}   {r['mask']}")
    if selfcheck:
        me = max(max(di, dp) for _, di, dp in selfcheck)
        print(f"\nself-check vs stored metrics_2d.json: max |Δ| = {me:.4f} (should be ~0)  -> {selfcheck}")
    print(f"\nwrote {OUT}   (nothing under results/ was modified)")


if __name__ == "__main__":
    main()
