"""Conf sweep + mask-level instance-segmentation scoring for the YOLO->SAM phone pipeline.

For a chosen SAM backend (sam1/sam2) and crop mode (full_frame/per_tile/per_head) this sweeps the YOLO
confidence threshold and, at each conf, scores the SAM masks that would survive that conf against the
hand-labelled GT instance masks. Output = a PR/F1 curve over conf + the best-F1 conf + an AP estimate,
so we can pick the production conf and compare SAM modes/backends on the SAME 6 GT images.

WHY two code paths (this is the whole point of the conf-equivalence work):
  full_frame / per_head : a head's mask is INDEPENDENT of the other boxes present (verified identical),
                          so we run SAM ONCE at a low conf floor, precompute each mask's GT overlap, and
                          then just DROP masks below each conf t — one SAM run gives the whole curve.
  per_tile              : the tile crop grows to contain its box group, so a kept head's mask depends on
                          which neighbours share its tile (verified: masks DIFFER, mean IoU ~0.90). The
                          low-floor->drop shortcut is INVALID, so we RE-RUN SAM at each conf t (per_tile is
                          fast, ~7-10 s/img, so this is cheap). This is the honest operating-point sweep.

Scoring is mask-vs-mask (NOT box-vs-mask): each predicted SAM mask is matched to a GT instance by mask-IoU,
greedy by YOLO conf (highest conf claims a GT first), one GT per match, unmatched preds = FP, unmatched GT
= FN. Metrics are micro-averaged (TP/FP/FN pooled over images).

RELATION to eval_masks_instance.py: that script scores masks-on-disk at ONE conf with Hungarian matching +
an IoU sweep, and deliberately has NO conf/AP because it feared borrowing the wrong confidence (bboxes/ and
bboxes_with_conf/ are different sets, so pairing by row index is unsafe). This script is the complement: it
runs SAM directly on bboxes_with_conf so each mask carries ITS OWN box's conf (no mismatch), which is what
makes an honest conf sweep possible. At IoU>=0.5 matches are ~unique, so greedy here ~= their Hungarian.

Smoke test (one image, one mode):
  python src/analysis/sweep_conf_mask_ap.py \
      --image input_plots/phone/field_A/20250715/images/IMG_20250715_153912.jpg \
      --gt    input_plots/phone/field_A/20250715/manual_label/IMG_20250715_153912_sets/set0_instances.png \
      --bbox  results/mask_generation/phone/field_A/20250715/yolo_sam_v1/metrics_v1/bboxes_with_conf/IMG_20250715_153912.pt \
      --backend sam1 --mode per_tile --thresholds 0.20 0.35 0.50

Full run (6 images via a manifest JSON list of {image, gt, bbox}):
  python src/analysis/sweep_conf_mask_ap.py --manifest configs/manifests/gt6.json \
      --backend sam1 --mode per_tile --sweep dense --out results/analysis/conf_sweep/sam1_per_tile.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mask_generation"))
from sam_v1.sam_v1_pipelined import _build_sam_backend, _infer_masks_dispatch  # noqa: E402

SAFE_MODES = ("full_frame", "per_head")   # box-independent -> one low-floor run covers the whole sweep


def _dense_sweep():
    """Default conf grid: 0.05 steps outside, 0.02 steps in the 0.2-0.6 region where the F1 optimum sits."""
    lo = np.arange(0.05, 0.20, 0.05)
    mid = np.arange(0.20, 0.601, 0.02)
    hi = np.arange(0.65, 0.951, 0.05)
    return sorted({round(float(x), 3) for x in np.concatenate([lo, mid, hi])})


def _load_gt(gt_path):
    """Load a uint16 GT instance mask -> (label image HxW int32, area-by-id array, list of nonzero ids)."""
    g = np.array(Image.open(gt_path)).astype(np.int32)
    if g.ndim == 3:                                   # be defensive: collapse a stray RGB-encoded label map
        g = g[:, :, 0].astype(np.int32)
    area = np.bincount(g.ravel())                     # area[id] = pixel count (index 0 = background)
    ids = [int(i) for i in np.unique(g) if i != 0]
    return g, area, ids


def _mask_candidates(pred_bool, gt_label, gt_area, iou_min):
    """For one predicted binary mask, return [(gt_id, iou), ...] for every GT instance it overlaps with
    IoU >= iou_min, sorted by IoU descending. Only touches the GT ids actually under the mask (cheap)."""
    pa = int(pred_bool.sum())
    if pa == 0:
        return []
    labs = gt_label[pred_bool]                        # GT ids at the predicted-mask pixels
    ids, inter = np.unique(labs, return_counts=True)
    out = []
    for gid, ic in zip(ids, inter):
        if gid == 0:
            continue
        iou = ic / (pa + gt_area[gid] - ic)
        if iou >= iou_min:
            out.append((int(gid), float(iou)))
    out.sort(key=lambda z: -z[1])
    return out


def _greedy(cands, confs, iou_thr, n_gt):
    """Greedy mask matching at one IoU threshold. cands[i] = sorted [(gid,iou)] for pred i (precomputed at
    a <= iou_thr floor). Highest-conf pred claims its best still-free GT with iou>=iou_thr. Returns tp,fp,fn."""
    order = np.argsort(-confs)                         # high conf first
    used = set()
    tp = 0
    for i in order:
        for gid, iou in cands[i]:
            if iou < iou_thr:                          # list is desc -> nothing better remains
                break
            if gid not in used:
                used.add(gid)
                tp += 1
                break
    fp = len(confs) - tp
    fn = n_gt - tp
    return tp, fp, fn


def _masks_to_candidates(masks, gt_label, gt_area, iou_min):
    """Turn a stack of (K,H,W) uint8 masks into per-mask GT-candidate lists, then let the caller free the
    stack. This is what keeps peak RAM at one run's mask array instead of holding masks across the sweep."""
    return [_mask_candidates(masks[j] > 0, gt_label, gt_area, iou_min) for j in range(len(masks))]


def _build_cfg(args):
    """Assemble the minimal cfg.method the SAM backend + dispatch read (mirrors the production config)."""
    return OmegaConf.create({"method": {
        "sam_backend": args.backend, "sam_checkpoint": args.sam_checkpoint,
        "sam_crop_mode": args.mode, "sam1_decode_batch": args.sam1_decode_batch,
        "sam_ul_decode_batch": 16, "sam_ul_chunk_on_oom": 64,
        "sam_tile_size": args.tile, "sam_tile_overlap": args.overlap, "sam_tile_pad_frac": 0.02,
        "sam_head_margin_frac": 0.4, "sam_head_min_pad": 16,
    }})


def _score_image(backend, state, cfg, mode, img_path, gt_path, bbox_path, thresholds, iou_thrs, dev):
    """Sweep one image. Returns per-(conf,iou) tp/fp/fn dict and n_gt. Safe modes run SAM once; per_tile
    re-runs per conf."""
    sam_image = np.array(Image.open(img_path).convert("RGB"))       # PIL loader = production SAM loader
    gt_label, gt_area, gt_ids = _load_gt(gt_path)
    n_gt = len(gt_ids)
    bwc = torch.load(bbox_path, weights_only=True)                  # (N,5) x1 y1 x2 y2 conf
    confs_all = bwc[:, 4].float().cpu().numpy()
    boxes_all = bwc[:, :4].float().to(dev)
    iou_min = min(iou_thrs)                                         # precompute candidates at the lowest thr
    print(f"  {os.path.basename(img_path)}: {len(bwc)} boxes (conf {confs_all.min():.3f}..{confs_all.max():.3f}), "
          f"GT heads={n_gt}")

    per = {}                                                        # (conf, iou_thr) -> [tp,fp,fn]

    if mode in SAFE_MODES:
        # ONE low-floor run -> per-mask GT candidates -> free masks -> sweep is pure bookkeeping
        masks = _infer_masks_dispatch(backend, state, sam_image, boxes_all, cfg)[0]
        cands_all = _masks_to_candidates(masks, gt_label, gt_area, iou_min)
        del masks
        for t in thresholds:
            idx = np.where(confs_all >= t)[0]
            sub_c = [cands_all[i] for i in idx]
            sub_conf = confs_all[idx]
            for iou_thr in iou_thrs:
                tp, fp, fn = _greedy(sub_c, sub_conf, iou_thr, n_gt)
                per[(t, iou_thr)] = [tp, fp, fn]
    else:
        # per_tile: RE-RUN SAM at each conf (crop grows to the kept boxes -> not post-filterable)
        for t in thresholds:
            keep = confs_all >= t
            idx = np.where(keep)[0]
            if len(idx) == 0:
                for iou_thr in iou_thrs:
                    per[(t, iou_thr)] = [0, 0, n_gt]
                continue
            m = _infer_masks_dispatch(backend, state, sam_image, boxes_all[torch.from_numpy(keep).to(dev)], cfg)[0]
            sub_c = _masks_to_candidates(m, gt_label, gt_area, iou_min)
            del m
            sub_conf = confs_all[idx]
            for iou_thr in iou_thrs:
                tp, fp, fn = _greedy(sub_c, sub_conf, iou_thr, n_gt)
                per[(t, iou_thr)] = [tp, fp, fn]
    return per, n_gt


def _prf(tp, fp, fn):
    """Precision, recall, F1 from counts (0 when undefined)."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _ap_from_points(points):
    """AP estimate = area under the (recall, precision) operating points, recall-sorted, with the standard
    precision envelope (monotone-decreasing interpolation). points = list of (recall, precision)."""
    if not points:
        return 0.0
    pts = sorted(points)                                           # by recall
    rec = [0.0] + [p[0] for p in pts]
    pre = [pts[0][1]] + [p[1] for p in pts]
    for i in range(len(pre) - 2, -1, -1):                          # make precision monotone from the right
        pre[i] = max(pre[i], pre[i + 1])
    ap = 0.0
    for i in range(1, len(rec)):
        ap += (rec[i] - rec[i - 1]) * pre[i]
    return float(ap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image"); ap.add_argument("--gt"); ap.add_argument("--bbox")
    ap.add_argument("--manifest", help="JSON list of {image, gt, bbox} to score together")
    ap.add_argument("--backend", default="sam1", choices=["sam1", "sam2"])
    ap.add_argument("--mode", default="per_tile", choices=["full_frame", "per_tile", "per_head"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=None, help="explicit conf grid")
    ap.add_argument("--sweep", choices=["dense"], default=None, help="use the built-in dense grid")
    ap.add_argument("--iou", nargs="+", type=float, default=[0.5], help="mask-IoU match threshold(s)")
    ap.add_argument("--weights_dir", default="src/mask_generation/weights")
    ap.add_argument("--sam_checkpoint", default="sam_vit_h_4b8939.pth")
    ap.add_argument("--sam1_decode_batch", type=int, default=16)
    ap.add_argument("--tile", type=int, default=1280); ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    if args.manifest:
        items = json.load(open(args.manifest))
    else:
        assert args.image and args.gt and args.bbox, "give --manifest OR --image/--gt/--bbox"
        items = [{"image": args.image, "gt": args.gt, "bbox": args.bbox}]

    thresholds = args.thresholds if args.thresholds else (_dense_sweep() if args.sweep == "dense" else
                                                          [0.20, 0.35, 0.50])
    iou_thrs = sorted(args.iou)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = _build_cfg(args)
    print(f"backend={args.backend} mode={args.mode} | {len(items)} image(s) | "
          f"{len(thresholds)} conf x {len(iou_thrs)} iou | device={dev}")
    print(f"Building SAM backend '{args.backend}' (torch.compile warmup can take ~1-2 min for sam1) ...")
    backend, state = _build_sam_backend(cfg, args.weights_dir)

    # pool tp/fp/fn over images per (conf, iou)
    pooled = {(t, i): [0, 0, 0] for t in thresholds for i in iou_thrs}
    total_gt = 0
    t_start = time.perf_counter()
    for it in items:
        per, n_gt = _score_image(backend, state, cfg, args.mode, it["image"], it["gt"], it["bbox"],
                                 thresholds, iou_thrs, dev)
        total_gt += n_gt
        for k, (tp, fp, fn) in per.items():
            pooled[k][0] += tp; pooled[k][1] += fp; pooled[k][2] += fn

    # assemble curves per IoU threshold
    results = {"backend": args.backend, "mode": args.mode, "n_images": len(items), "total_gt": total_gt,
               "iou_thresholds": iou_thrs, "curves": {}}
    for iou_thr in iou_thrs:
        rows = []
        for t in thresholds:
            tp, fp, fn = pooled[(t, iou_thr)]
            p, r, f = _prf(tp, fp, fn)
            rows.append({"conf": t, "tp": tp, "fp": fp, "fn": fn,
                         "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)})
        best = max(rows, key=lambda z: z["f1"])
        ap_est = _ap_from_points([(z["recall"], z["precision"]) for z in rows])
        results["curves"][str(iou_thr)] = {"rows": rows, "best_f1": best["f1"],
                                           "best_f1_conf": best["conf"], "ap_estimate": round(ap_est, 4)}
        print(f"\n[IoU>={iou_thr}] best F1={best['f1']:.4f} @ conf={best['conf']}  AP~={ap_est:.4f}")
        print("   conf     P      R      F1     TP   FP   FN")
        for z in rows:
            print(f"  {z['conf']:.2f}  {z['precision']:.3f}  {z['recall']:.3f}  {z['f1']:.3f}  "
                  f"{z['tp']:4d} {z['fp']:4d} {z['fn']:4d}")
    print(f"\nelapsed {time.perf_counter() - t_start:.1f}s")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
