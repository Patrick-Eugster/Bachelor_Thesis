"""
Instance-level evaluation of mask-generation output against the manual GT instance masks.

WHY THIS EXISTS: every other evaluator collapses masks to a UNION (score_sam_masks.py, eval_seg_2d.py),
and a union IoU CANNOT SEE A MERGE — two wheat heads fused into one blob score ~perfectly. On phone the
heads overlap heavily, so merging is the failure mode we actually care about. This script matches predicted
head masks to GT head masks one-to-one and reports precision/recall/F1/IoU PLUS explicit merge/split counts.
(No AP: that would need a per-mask confidence, see the note at the bottom of this docstring.)

Run from the workspace root:
  python src/mask_generation/evaluation/eval_masks_instance.py dataset=phone method=sahi_yolo_sam \
      mask_gen_experiment=initial

GT contract (written by the point-GT tool, src/mask_generation/gt_tool/):
  input_plots/{dataset}/{plot}/manual_label/{stem}_sets/manifest.json    -> names the "active" set
  input_plots/{dataset}/{plot}/manual_label/{stem}_sets/set{N}_instances.png -> uint16 instance map
      (0 = background, 1..N = one id per wheat head)
  ⚠ Read via the manifest. A top-level {stem}_instances.png is a STALE leftover from an older tool
    version (archived in archive/gt_tool_stale_instances/) — it froze at <50% of the labeled heads.

Predictions:
  results/mask_generation/{dataset}/{plot}/{method}/{exp}/masks/{stem}_NNN.png
      one FULL-RES binary mask per head ({stem}_union.png is ignored by the strict name regex)

Output:
  results/mask_generation/{dataset}/evaluation/{method}/masks_instance/{eval_experiment}/
      eval_masks_instance.json + config.yaml + viz/{stem}_instance_eval.png

NOTE on AP: SAM emits no mask confidence, so ranking would have to borrow the YOLO box score. Also
bboxes/ (good boxes -> masks) and bboxes_with_conf/ (ALL raw preds) are DIFFERENT sets — pairing them by
row index silently attaches the wrong confidence. Left out on purpose; F1/IoU/merge/split need no score.
"""

import os
import re
import glob
import json
import datetime

import numpy as np
import cv2
import hydra
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import find_objects


# ----------------------------------------------------------------------------- config / paths

def get_eval_experiment(cfg):
    """Resolve the evaluation output folder name — same 3-option logic as the rest of the pipeline."""
    if not cfg.eval_experiment:
        return datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    if cfg.eval_experiment == "initial":
        return "initial"
    if cfg.prepend_date:
        return f"{datetime.datetime.now().strftime('%Y-%m-%d')}_{cfg.eval_experiment}"
    return cfg.eval_experiment


def get_method_name(cfg):
    """Which results/<method>/ folder to score. method_name lets us score a folder that has no registered
    method config (e.g. the standalone run_yolo11_seg.py writes to yolo11_seg/ but isn't a method)."""
    return cfg.method_name if cfg.get("method_name") else cfg.method.name


def find_labeled_images(cfg):
    """Every image that has a GT instance map. The presence of a <stem>_sets/ folder IS the marker.
    dataset.plot_glob handles both FIP (plot_461/) and phone (field_A/20250715/)."""
    found = []
    for input_plot_dir in sorted(glob.glob(os.path.join(cfg.dataset.input_dir, cfg.dataset.plot_glob))):
        if not os.path.isdir(input_plot_dir):
            continue
        plot_name = os.path.relpath(input_plot_dir, cfg.dataset.input_dir)
        label_dir = os.path.join(input_plot_dir, "manual_label")
        for sets_dir in sorted(glob.glob(os.path.join(label_dir, "*_sets"))):
            stem = os.path.basename(sets_dir)[: -len("_sets")]
            found.append({
                "plot": plot_name,
                "stem": stem,
                "input_plot_dir": input_plot_dir,
                "label_dir": label_dir,
                "result_dir": os.path.join(cfg.dataset.result_dir_masks, plot_name,
                                           get_method_name(cfg), cfg.mask_gen_experiment),
            })
    return found


# ----------------------------------------------------------------------------- loading

def load_gt_instances(label_dir, stem):
    """Load one image's GT instance map through the sets manifest (the authoritative path).
    Returns (instance_map int32 HxW, gt_ids, areas_by_id) or (None, None, None) if there's no GT."""
    sets_dir = os.path.join(label_dir, f"{stem}_sets")
    man_path = os.path.join(sets_dir, "manifest.json")
    if not os.path.exists(man_path):
        return None, None, None
    man = json.load(open(man_path))
    entry = next((e for e in man.get("sets", []) if e["name"] == man.get("active")), None)
    if entry is None:                                   # active name missing -> fall back to the first set
        if not man.get("sets"):
            return None, None, None
        entry = man["sets"][0]
        print(f"    WARNING: manifest 'active' set not found for {stem}; using {entry['name']}")
    png = os.path.join(sets_dir, f"{entry['file']}_instances.png")
    m = cv2.imread(png, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None, None, None
    if m.ndim == 3:                                     # be tolerant if it ever got saved as 3-channel
        m = m[..., 0]
    m = m.astype(np.int32)
    areas = np.bincount(m.ravel())
    ids = np.nonzero(areas)[0]
    ids = ids[ids != 0]                                 # drop background
    return m, ids, areas


def load_pred_masks(masks_dir, stem, shape):
    """Load the predicted per-head masks as SPARSE {bbox, sub, area}.
    Each SAM mask is a FULL-RES png (~12 MB) and there can be ~600 per image (~7 GB if held dense), so
    each one is cropped to its own bbox right after decode and the full frame dropped."""
    pat = re.compile(re.escape(stem) + r"_\d+\.png$")   # strict: skips _union.png + a longer stem's files
    paths = sorted(p for p in glob.glob(os.path.join(masks_dir, f"{stem}_*.png"))
                   if pat.search(os.path.basename(p)))
    preds = []
    for p in paths:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != shape:                            # GT and preds must live on the same frame
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        b = m > 127
        if not b.any():
            continue
        ys, xs = np.nonzero(b)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        preds.append({"bbox": (x0, y0, x1, y1),
                      "sub": np.ascontiguousarray(b[y0:y1, x0:x1]),
                      "area": int(b.sum())})
    return preds


# ----------------------------------------------------------------------------- the core

def compute_iou_matrix(gt_map, gt_ids, gt_areas, preds):
    """IoU of every prediction against every GT head — the instance-map + bincount trick.

    For each prediction we look up the GT ids lying under its pixels and bincount them: because every
    pixel belongs to exactly ONE GT head, that single pass gives the intersection with EVERY head at once.
    Cost is O(sum of prediction areas) instead of O(n_pred * n_gt * H * W) — ~600x600 full-frame IoUs
    would be hopeless. Returns (iou, inter), both (n_pred, n_gt) and aligned to gt_ids."""
    n_pred, n_gt = len(preds), len(gt_ids)
    inter = np.zeros((n_pred, n_gt), np.int64)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), np.float64), inter
    minlen = int(gt_map.max()) + 1
    for i, p in enumerate(preds):
        x0, y0, x1, y1 = p["bbox"]
        ids_under = gt_map[y0:y1, x0:x1][p["sub"]]      # the GT id at each pixel of this prediction
        cnt = np.bincount(ids_under, minlength=minlen)  # cnt[id] = overlap area with that head
        inter[i] = cnt[gt_ids]
    pred_areas = np.array([p["area"] for p in preds], np.int64)
    gt_a = gt_areas[gt_ids].astype(np.int64)
    union = pred_areas[:, None] + gt_a[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return iou, inter


def match_instances(iou, thr):
    """One-to-one matching of predictions to GT heads (Hungarian, maximising total IoU), keeping only
    pairs at/above the threshold. Above IoU 0.5 a match is unique anyway; Hungarian keeps the lower
    thresholds honest too."""
    if iou.size == 0:
        return []
    rows, cols = linear_sum_assignment(-iou)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if iou[i, j] >= thr]


def metrics_at(iou, thr, n_pred, n_gt):
    """precision / recall / F1 / mean matched IoU / panoptic quality at one IoU threshold."""
    pairs = match_instances(iou, thr)
    tp = len(pairs)
    fp, fn = n_pred - tp, n_gt - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    ious = [float(iou[i, j]) for i, j in pairs]
    sq = float(np.mean(ious)) if ious else 0.0          # segmentation quality = mean IoU of the matches
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = float(sum(ious) / denom) if denom > 0 else 0.0  # panoptic quality = SQ x RQ in one number
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
            "mean_iou_matched": sq, "pq": pq}, pairs


def matched_iou_values(iou, thr):
    """The per-pair IoU of every matched prediction<->GT head at threshold `thr`. This is the raw
    distribution that mean_iou_matched collapses to a single number — histogrammed to see if the
    matched masks cluster tight (~0.7) or spread out."""
    return [float(iou[i, j]) for i, j in match_instances(iou, thr)]


def iou_curve(iou, thresholds, n_pred, n_gt):
    """P/R/F1 at each of a fine IoU sweep — the data behind the F1-vs-IoU curve figure.
    Just re-runs the (cheap) Hungarian match per threshold on the already-built IoU matrix."""
    curve = []
    for t in thresholds:
        m, _ = metrics_at(iou, float(t), n_pred, n_gt)
        curve.append({"iou": round(float(t), 4), "precision": m["precision"],
                      "recall": m["recall"], "f1": m["f1"], "tp": m["tp"]})
    return curve


def count_merges_splits(inter, preds, gt_areas, gt_ids, merge_frac, split_frac):
    """The failures a UNION mask cannot show.
      merge = ONE prediction swallowing >=2 GT heads (each with >= merge_frac of its area inside it)
      split = ONE GT head broken across >=2 predictions (each with >= split_frac of its OWN area inside)
    Returns (counts, merge_pred_indices, split_gt_indices) — the indices let the viz colour them."""
    if inter.size == 0:
        return {"merge_preds": 0, "gt_heads_merged": 0, "split_gts": 0}, [], []
    pred_areas = np.array([p["area"] for p in preds], np.int64)
    gt_a = gt_areas[gt_ids].astype(np.int64)
    frac_of_gt = inter / np.maximum(gt_a[None, :], 1)          # how much of each GT head is in each pred
    frac_of_pred = inter / np.maximum(pred_areas[:, None], 1)  # how much of each pred is in each GT head
    merge_hits = frac_of_gt >= merge_frac
    merge_idx = np.nonzero(merge_hits.sum(axis=1) >= 2)[0]
    gt_merged = int(merge_hits[merge_idx].any(axis=0).sum()) if len(merge_idx) else 0
    split_hits = frac_of_pred >= split_frac
    split_idx = np.nonzero(split_hits.sum(axis=0) >= 2)[0]
    counts = {"merge_preds": int(len(merge_idx)), "gt_heads_merged": gt_merged,
              "split_gts": int(len(split_idx))}
    return counts, merge_idx.tolist(), split_idx.tolist()


def semantic_ious(gt_map, preds):
    """Binary SEMANTIC-segmentation IoUs — every head melted into ONE foreground class, no per-head identity.
    Returns (foreground_iou, background_iou, miou).

    - foreground_iou = the classic 'union pixel IoU' (aka foreground IoU) that score_sam_masks.py /
      eval_seg_2d.py report. Kept for CONTRAST: it's BLIND TO MERGES (fused heads still union perfectly), so
      a big gap between it and the instance F1 IS the merge problem, quantified.
    - background_iou = IoU of the not-a-head class. Background is huge and easy, so this sits near 1.0.
    - miou = mean of the two classes = the standard semantic-seg metric. Because background inflates it, mIoU
      flatters; we report it for completeness but foreground IoU is the honest single number here."""
    gt_fg = gt_map > 0
    pred_fg = np.zeros(gt_map.shape, bool)
    for p in preds:
        x0, y0, x1, y1 = p["bbox"]
        pred_fg[y0:y1, x0:x1] |= p["sub"]

    def _iou(a, b):
        inter = int(np.logical_and(a, b).sum())
        uni = int(np.logical_or(a, b).sum())
        return inter / uni if uni else 0.0

    fg = _iou(gt_fg, pred_fg)
    bg = _iou(~gt_fg, ~pred_fg)
    return fg, bg, 0.5 * (fg + bg)


# ----------------------------------------------------------------------------- boundary (edge) metrics

def _erode(m, k):
    """Binary erosion by a (2k+1) square — used to peel a boundary ring off a mask."""
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 2 * k + 1))
    return cv2.erode(m.astype(np.uint8), ker)


def _boundary_band(m, d):
    """A d-px-wide ring just inside a mask's edge: mask minus its erosion. This is what boundary-IoU scores
    instead of the whole area, so a big head with a ragged edge no longer hides its edge error."""
    return (m.astype(np.uint8) & (1 - _erode(m, d))).astype(np.uint8)


def _contour(m):
    """The 1-px outline of a mask (mask minus a 1-px erosion)."""
    return (m.astype(np.uint8) & (1 - _erode(m, 1))).astype(np.uint8)


def boundary_iou(pm, gm, d):
    """IoU of just the two boundary rings (edge agreement, area-independent)."""
    pb, gb = _boundary_band(pm, d), _boundary_band(gm, d)
    inter = int(np.logical_and(pb, gb).sum())
    uni = int(np.logical_or(pb, gb).sum())
    return inter / uni if uni else 0.0


def boundary_f(pm, gm, tol):
    """DAVIS-style boundary F: precision = fraction of predicted contour pixels within `tol` of a GT contour
    pixel, recall = the reverse, F = their harmonic mean. Distances come from a distance transform (distance
    to the nearest contour pixel), so `tol` is a pixel tolerance on how well the outlines trace each other."""
    pc, gc = _contour(pm), _contour(gm)
    ps, gs = int(pc.sum()), int(gc.sum())
    if ps == 0 and gs == 0:
        return 1.0
    if ps == 0 or gs == 0:
        return 0.0
    dt_g = cv2.distanceTransform(1 - gc, cv2.DIST_L2, 3)     # each pixel -> distance to nearest GT contour
    dt_p = cv2.distanceTransform(1 - pc, cv2.DIST_L2, 3)
    prec = float((dt_g[pc > 0] <= tol).mean())
    rec = float((dt_p[gc > 0] <= tol).mean())
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _dyn_band(gt_area, k, min_px):
    """Size-proportional boundary width for ONE head: max(min_px, round(k*sqrt(area))). Scales the band
    with the head's apparent size (Boundary IoU paper, Cheng 2021, which scales by object diagonal; sqrt(area)
    is the same idea, cheap). Big near heads get a wider band, tiny far heads stay at the floor."""
    return int(max(min_px, round(k * np.sqrt(max(gt_area, 1)))))


def boundary_metrics_over_pairs(preds, pairs, gt_map, gt_ids, gt_slices, band_d, tol, dyn_k, dyn_min_px):
    """Mean boundary-IoU and boundary-F over the matched pairs, computed TWO ways in one pass over the same
    per-head canvas (cost stays per-head, not per-frame):
      - FIXED  : a constant band_d / tol (px at full res) — the original metric.
      - DYNAMIC: a per-head band scaled by that GT head's size (_dyn_band), so it's scale-invariant across
        near/far heads. The tolerance for the F-score scales the same way.
    Returns (fixed_dict, dynamic_dict), each {boundary_iou, boundary_f, n_pairs}."""
    fx_iou, fx_f, dyn_iou, dyn_f = [], [], [], []
    for i, j in pairs:
        gid = int(gt_ids[j])
        sl = gt_slices[gid - 1] if 0 <= gid - 1 < len(gt_slices) else None
        if sl is None:
            continue
        gy0, gy1, gx0, gx1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        px0, py0, px1, py1 = preds[i]["bbox"]
        X0, Y0 = min(px0, gx0), min(py0, gy0)
        X1, Y1 = max(px1, gx1), max(py1, gy1)
        pm = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
        pm[py0 - Y0:py1 - Y0, px0 - X0:px1 - X0] = preds[i]["sub"]
        gm = (gt_map[Y0:Y1, X0:X1] == gid).astype(np.uint8)
        fx_iou.append(boundary_iou(pm, gm, band_d))
        fx_f.append(boundary_f(pm, gm, tol))
        d = _dyn_band(int(gm.sum()), dyn_k, dyn_min_px)     # band from THIS head's area
        dyn_iou.append(boundary_iou(pm, gm, d))
        dyn_f.append(boundary_f(pm, gm, d))                 # tolerance scales with the head too
    _mean = lambda v: float(np.mean(v)) if v else float("nan")
    fixed = {"boundary_iou": _mean(fx_iou), "boundary_f": _mean(fx_f), "n_pairs": len(fx_iou)}
    dynamic = {"boundary_iou": _mean(dyn_iou), "boundary_f": _mean(dyn_f), "n_pairs": len(dyn_iou)}
    return fixed, dynamic


# ----------------------------------------------------------------------------- viz

def save_viz(img_path, gt_map, gt_ids, preds, pairs, merge_idx, split_idx, out_path):
    """Colour-coded overlay so the failures are visible, not just tabulated:
    green=TP, red=FP (pred with no GT), blue=missed GT, magenta=MERGE, orange=SPLIT."""
    img = cv2.imread(img_path)
    if img is None:
        img = np.zeros((*gt_map.shape, 3), np.uint8)
    if img.shape[:2] != gt_map.shape:
        img = cv2.resize(img, (gt_map.shape[1], gt_map.shape[0]))
    over = img.copy()
    matched_pred = {i for i, _ in pairs}
    matched_gt = {j for _, j in pairs}
    merge_set, split_set = set(merge_idx), set(split_idx)
    C_TP, C_FP, C_MERGE = (0, 200, 0), (0, 0, 255), (255, 0, 255)
    C_FN, C_SPLIT = (255, 60, 60), (0, 165, 255)

    for i, p in enumerate(preds):                       # filled predictions
        col = C_MERGE if i in merge_set else (C_TP if i in matched_pred else C_FP)
        x0, y0, x1, y1 = p["bbox"]
        over[y0:y1, x0:x1][p["sub"]] = col

    # GT heads that were missed or split — outline them. find_objects gives each id's slice in ONE pass
    # (scanning the full map per head would be O(H*W*n_gt) and far too slow).
    slices = find_objects(gt_map)
    for j, gid in enumerate(gt_ids):
        if j in matched_gt and j not in split_set:
            continue
        sl = slices[gid - 1] if gid - 1 < len(slices) else None
        if sl is None:
            continue
        sub = (gt_map[sl] == gid).astype(np.uint8)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c + [sl[1].start, sl[0].start] for c in cnts]    # slice-local -> full-frame coords
        cv2.drawContours(over, cnts, -1, C_SPLIT if j in split_set else C_FN, 2)

    out = cv2.addWeighted(over, 0.55, img, 0.45, 0)
    legend = [("TP", C_TP), ("FP", C_FP), ("missed GT", C_FN), ("MERGE", C_MERGE), ("SPLIT", C_SPLIT)]
    y = 34
    for name, col in legend:
        cv2.rectangle(out, (14, y - 18), (44, y + 2), col, -1)
        cv2.putText(out, name, (52, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 30
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out)


# ----------------------------------------------------------------------------- reporting

def print_single(rec, thr):
    """One image's headline numbers."""
    m = rec["at_threshold"]
    print(f"  {rec['plot']}/{rec['stem']}")
    print(f"    GT heads {rec['n_gt']:4d} | pred {rec['n_pred']:4d} | count err {rec['count_error_ratio']:+.1%}")
    print(f"    IoU@{thr}: P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f} "
          f"| mean IoU(matched) {m['mean_iou_matched']:.3f} | PQ {m['pq']:.3f}")
    print(f"    TP {m['tp']:4d}  FP {m['fp']:4d}  FN {m['fn']:4d}")
    ms = rec["merge_split"]
    print(f"    MERGES: {ms['merge_preds']} preds swallowing {ms['gt_heads_merged']} GT heads "
          f"| SPLITS: {ms['split_gts']} GT heads")
    if "boundary" in rec:
        b = rec["boundary"]
        print(f"    boundary IoU {b['boundary_iou']:.3f} | boundary F {b['boundary_f']:.3f} "
              f"(fixed band, edge fidelity over {b['n_pairs']} matched pairs)")
    if "boundary_dynamic" in rec:
        d = rec["boundary_dynamic"]
        print(f"    boundary IoU {d['boundary_iou']:.3f} | boundary F {d['boundary_f']:.3f} "
              f"(dynamic size-scaled band)")
    print(f"    foreground IoU (union pixel IoU) {rec['foreground_iou']:.3f}  (semantic, blind to merges — "
          f"compare with F1 above) | background IoU {rec['background_iou']:.3f} | mIoU {rec['miou']:.3f}")


def print_aggregate(records, cfg, thr):
    """Mean +/- std across images, plus the totals that actually matter for merges."""
    if not records:
        return
    def ms(key, sub="at_threshold"):
        v = [r[sub][key] for r in records]
        return float(np.mean(v)), float(np.std(v))
    print("\n" + "=" * 78)
    print(f" AGGREGATE over {len(records)} image(s) — {get_method_name(cfg)} / {cfg.mask_gen_experiment}")
    print("=" * 78)
    for k in ("precision", "recall", "f1", "mean_iou_matched", "pq"):
        mean, std = ms(k)
        print(f"   {k:18s} {mean:.3f} ± {std:.3f}")
    if all("boundary" in r for r in records):               # edge fidelity (matched pairs only)
        for k in ("boundary_iou", "boundary_f"):
            v = [r["boundary"][k] for r in records if not np.isnan(r["boundary"][k])]
            if v:
                print(f"   {k:18s} {float(np.mean(v)):.3f} ± {float(np.std(v)):.3f}   <- fixed band")
    if all("boundary_dynamic" in r for r in records):       # size-proportional band, side by side
        for k in ("boundary_iou", "boundary_f"):
            v = [r["boundary_dynamic"][k] for r in records if not np.isnan(r["boundary_dynamic"][k])]
            if v:
                print(f"   {k + '(dyn)':18s} {float(np.mean(v)):.3f} ± {float(np.std(v)):.3f}   <- dynamic band")
    tot_gt = sum(r["n_gt"] for r in records)
    tot_pred = sum(r["n_pred"] for r in records)
    tot_merge = sum(r["merge_split"]["merge_preds"] for r in records)
    tot_merged_gt = sum(r["merge_split"]["gt_heads_merged"] for r in records)
    tot_split = sum(r["merge_split"]["split_gts"] for r in records)
    def top(key):
        v = [r[key] for r in records]
        return float(np.mean(v)), float(np.std(v))
    fg_m, fg_s = top("foreground_iou")
    bg_m, bg_s = top("background_iou")
    mi_m, mi_s = top("miou")
    print(f"   {'foreground IoU':18s} {fg_m:.3f} ± {fg_s:.3f}   <- old union-only metric (== union pixel IoU), blind to merges")
    print(f"   {'background IoU':18s} {bg_m:.3f} ± {bg_s:.3f}   <- huge easy class (near 1) — inflates mIoU")
    print(f"   {'mIoU (fg+bg)':18s} {mi_m:.3f} ± {mi_s:.3f}   <- standard semantic-seg metric; optimistic here")
    print(f"\n   totals: GT {tot_gt} | pred {tot_pred}")
    print(f"   MERGES: {tot_merge} predictions swallowed {tot_merged_gt} GT heads "
          f"({tot_merged_gt / tot_gt:.1%} of all GT)" if tot_gt else "")
    print(f"   SPLITS: {tot_split} GT heads broken across >1 prediction "
          f"({tot_split / tot_gt:.1%} of all GT)" if tot_gt else "")
    print("=" * 78)


# ----------------------------------------------------------------------------- curve + histogram outputs

def _mean_curve(records):
    """Average the per-image F1-vs-IoU curves into one curve (mean P/R/F1 per threshold, over images
    that have that threshold). Returns a list aligned to the thresholds of the first record's curve."""
    curves = [r["f1_curve"] for r in records if r.get("f1_curve")]
    if not curves:
        return []
    thrs = [pt["iou"] for pt in curves[0]]
    out = []
    for k, t in enumerate(thrs):
        p = [c[k]["precision"] for c in curves if k < len(c)]
        r = [c[k]["recall"] for c in curves if k < len(c)]
        f = [c[k]["f1"] for c in curves if k < len(c)]
        out.append({"iou": t, "precision": float(np.mean(p)), "recall": float(np.mean(r)),
                    "f1": float(np.mean(f))})
    return out


def write_curve_and_hist(records, cfg, eval_dir, thr):
    """Write the F1-vs-IoU curve (CSV + optional PNG) and the matched-pair IoU histogram (CSV + optional
    PNG), aggregated across all scored images. Returns a small dict folded into the JSON summary."""
    tag = f"{get_method_name(cfg)} / {cfg.mask_gen_experiment}"

    # --- F1-vs-IoU curve (mean over images) ---
    mean_curve = _mean_curve(records)
    curve_csv = os.path.join(eval_dir, "f1_vs_iou.csv")
    with open(curve_csv, "w") as f:
        f.write("iou,precision,recall,f1\n")
        for pt in mean_curve:
            f.write(f"{pt['iou']:.2f},{pt['precision']:.4f},{pt['recall']:.4f},{pt['f1']:.4f}\n")

    # --- matched-pair IoU histogram (pooled over all images, matched at hist_iou_threshold) ---
    hist_thr = float(cfg.get("hist_iou_threshold", 0.25))
    nbins = int(cfg.get("hist_bins", 20))
    pooled = [v for r in records for v in r.get("matched_ious", [])]
    edges = np.linspace(hist_thr, 1.0, nbins + 1)
    counts, _ = np.histogram(pooled, bins=edges) if pooled else (np.zeros(nbins, int), edges)
    hist_csv = os.path.join(eval_dir, "matched_iou_hist.csv")
    with open(hist_csv, "w") as f:
        f.write("bin_lo,bin_hi,count\n")
        for b in range(nbins):
            f.write(f"{edges[b]:.4f},{edges[b + 1]:.4f},{int(counts[b])}\n")
    stats = {}
    if pooled:
        arr = np.array(pooled)
        stats = {"n": int(arr.size), "mean": float(arr.mean()), "median": float(np.median(arr)),
                 "p25": float(np.percentile(arr, 25)), "p75": float(np.percentile(arr, 75)),
                 "min": float(arr.min()), "max": float(arr.max())}

    # console: compact ASCII so you see it without opening the PNG
    print("\n" + "-" * 78)
    print(f" F1-vs-IoU curve (mean over {len(records)} img) — {tag}")
    print("   IoU   P      R      F1")
    for pt in mean_curve:
        print(f"   {pt['iou']:.2f}  {pt['precision']:.3f}  {pt['recall']:.3f}  {pt['f1']:.3f}")
    if pooled:
        print(f" matched-pair IoU (matched@{hist_thr}, n={stats['n']}): "
              f"median {stats['median']:.3f} | IQR [{stats['p25']:.3f}, {stats['p75']:.3f}] "
              f"| mean {stats['mean']:.3f}")
        peak = int(np.argmax(counts))
        print(f"   histogram peak bin: [{edges[peak]:.2f}, {edges[peak + 1]:.2f}] ({int(counts[peak])} pairs)")
    print(f" CSVs -> {os.path.basename(curve_csv)}, {os.path.basename(hist_csv)}")
    print("-" * 78)

    if cfg.get("save_plots", True):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if mean_curve:
                xs = [pt["iou"] for pt in mean_curve]
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(xs, [pt["f1"] for pt in mean_curve], "-o", label="F1", color="tab:blue")
                ax.plot(xs, [pt["precision"] for pt in mean_curve], "--", label="precision", color="tab:green")
                ax.plot(xs, [pt["recall"] for pt in mean_curve], "--", label="recall", color="tab:red")
                ax.axvline(thr, color="grey", ls=":", lw=1, label=f"table thr {thr}")
                ax.set_xlabel("matching IoU threshold"); ax.set_ylabel("score")
                ax.set_ylim(0, 1); ax.set_title(f"F1 vs IoU — {tag}")
                ax.grid(alpha=0.3); ax.legend(fontsize=8)
                fig.tight_layout(); fig.savefig(os.path.join(eval_dir, "f1_vs_iou.png"), dpi=130)
                plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(0.5 * (edges[:-1] + edges[1:]), counts, width=(edges[1] - edges[0]) * 0.9,
                   color="tab:purple", edgecolor="black", linewidth=0.4)
            if pooled:
                ax.axvline(stats["median"], color="black", ls="--", lw=1,
                           label=f"median {stats['median']:.3f}")
                ax.legend(fontsize=8)
            ax.set_xlabel(f"matched-pair IoU (matched@{hist_thr})"); ax.set_ylabel("count")
            ax.set_title(f"Matched-pair IoU distribution — {tag}")
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout(); fig.savefig(os.path.join(eval_dir, "matched_iou_hist.png"), dpi=130)
            plt.close(fig)
        except ImportError:
            print("   (matplotlib not available — wrote CSVs only)")

    return {"f1_curve_mean": mean_curve, "matched_iou_stats": stats,
            "matched_iou_hist": {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}}


# ----------------------------------------------------------------------------- driver

def evaluate_all(cfg):
    """Score every GT-labeled image and write the JSON + viz."""
    eval_dir = os.path.join(cfg.dataset.result_dir_masks, "evaluation", get_method_name(cfg),
                            "masks_instance", get_eval_experiment(cfg))
    viz_dir = os.path.join(eval_dir, "viz")
    os.makedirs(eval_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(eval_dir, "config.yaml"))

    thr = float(cfg.matching_iou_threshold)
    items = find_labeled_images(cfg)
    print("=" * 78)
    print(f" Instance-level mask eval — method={get_method_name(cfg)}  run={cfg.mask_gen_experiment}")
    print(f" GT images found: {len(items)} | matching IoU: {thr}")
    print(f" Output: {eval_dir}")
    print("=" * 78)
    if not items:
        print("No GT found (looked for manual_label/*_sets/). Nothing to do.")
        return

    records = []
    for it in items:
        gt_map, gt_ids, gt_areas = load_gt_instances(it["label_dir"], it["stem"])
        if gt_map is None:
            print(f"  {it['plot']}/{it['stem']}: no readable GT instance map — skipping")
            continue
        masks_dir = os.path.join(it["result_dir"], "masks")
        if not os.path.isdir(masks_dir):
            print(f"  {it['plot']}/{it['stem']}: no masks/ at {masks_dir} — skipping")
            continue
        preds = load_pred_masks(masks_dir, it["stem"], gt_map.shape)
        if not preds:
            print(f"  {it['plot']}/{it['stem']}: no predicted masks for this stem — skipping")
            continue

        iou, inter = compute_iou_matrix(gt_map, gt_ids, gt_areas, preds)
        at_thr, pairs = metrics_at(iou, thr, len(preds), len(gt_ids))
        ms_counts, merge_idx, split_idx = count_merges_splits(
            inter, preds, gt_areas, gt_ids, float(cfg.merge_frac), float(cfg.split_frac))
        others = {}
        for t in cfg.extra_iou_thresholds:
            m, _ = metrics_at(iou, float(t), len(preds), len(gt_ids))
            others[str(t)] = m

        # fine IoU sweep for the F1-vs-IoU curve figure, and the matched-pair IoU distribution
        curve = iou_curve(iou, list(cfg.get("curve_iou_thresholds", [])), len(preds), len(gt_ids))
        matched_ious = matched_iou_values(iou, float(cfg.get("hist_iou_threshold", 0.25)))

        rec = {
            "plot": it["plot"], "stem": it["stem"],
            "n_gt": int(len(gt_ids)), "n_pred": int(len(preds)),
            "count_error_ratio": float((len(preds) - len(gt_ids)) / max(len(gt_ids), 1)),
            "at_threshold": at_thr,
            "by_iou_threshold": others,
            "f1_curve": curve,
            "matched_ious": matched_ious,
            "merge_split": ms_counts,
            "foreground_iou": None, "background_iou": None, "miou": None,   # filled just below
            "union_pixel_iou": None,   # == foreground_iou; kept for back-compat with older readers
        }

        # semantic (foreground/background) IoUs — heads as ONE class; foreground_iou is the old union metric
        fg_iou, bg_iou, miou = semantic_ious(gt_map, preds)
        rec["foreground_iou"] = float(fg_iou)
        rec["background_iou"] = float(bg_iou)
        rec["miou"] = float(miou)
        rec["union_pixel_iou"] = float(fg_iou)     # alias

        # boundary (edge-fidelity) metrics on the matched pairs — what area-IoU can't see at the outline
        if cfg.get("compute_boundary", True):
            gt_slices = find_objects(gt_map)
            fixed_b, dyn_b = boundary_metrics_over_pairs(
                preds, pairs, gt_map, gt_ids, gt_slices,
                int(cfg.get("boundary_band_px", 2)), float(cfg.get("boundary_tol_px", 2)),
                float(cfg.get("boundary_dyn_k", 0.05)), int(cfg.get("boundary_dyn_min_px", 2)))
            rec["boundary"] = fixed_b               # fixed-band block (back-compat name)
            rec["boundary_dynamic"] = dyn_b         # size-proportional band (parallel block)

        records.append(rec)
        print_single(rec, thr)

        if cfg.save_viz:
            hits = glob.glob(os.path.join(it["input_plot_dir"], "images", it["stem"] + ".*"))
            save_viz(hits[0] if hits else "", gt_map, gt_ids, preds, pairs, merge_idx, split_idx,
                     os.path.join(viz_dir, f"{it['stem']}_instance_eval.png"))

    print_aggregate(records, cfg, thr)

    curve_hist = {}
    if records:
        curve_hist = write_curve_and_hist(records, cfg, eval_dir, thr)

    out = {"method": get_method_name(cfg), "mask_gen_experiment": cfg.mask_gen_experiment,
           "matching_iou_threshold": thr, "merge_frac": float(cfg.merge_frac),
           "split_frac": float(cfg.split_frac), "n_images": len(records),
           "hist_iou_threshold": float(cfg.get("hist_iou_threshold", 0.25)),
           "aggregate_curve_hist": curve_hist, "images": records}
    with open(os.path.join(eval_dir, "eval_masks_instance.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nJSON -> {os.path.join(eval_dir, 'eval_masks_instance.json')}")
    if cfg.save_viz:
        print(f"viz  -> {viz_dir}")


@hydra.main(version_base=None, config_path="../../../configs/mask_generation",
            config_name="eval_masks_instance")
def main(cfg: DictConfig):
    evaluate_all(cfg)


if __name__ == "__main__":
    main()
