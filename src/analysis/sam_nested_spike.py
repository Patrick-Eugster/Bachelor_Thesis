"""
sam_nested_spike.py — throwaway spike (NOT pipeline code).

ONE question before we build mask-based dedup: when two real wheat heads have NESTED boxes (a small
head sitting inside a big diagonal head's axis-aligned box), does SAM return TWO distinct masks, or
does it blob them into one? If distinct -> mask-overlap can separate them -> mask-dedup will work.
If blobbed -> mask-dedup can't fix it either and we stop.

It uses the GROUND-TRUTH boxes (input_plots/fip/plot_*/manual_label/*.txt, YOLO format), because GT
has a separate box per head, so a nested GT pair = a correct prompt pair to test SAM with. For each
nested pair it prompts SAM with the big box and the small box separately, then compares the two MASKS:
  - box IoS high + mask IoS LOW  -> SAM separated the heads (GREEN, build it)
  - mask IoS still HIGH          -> SAM blobbed them (RED, mask-dedup won't help)

Run:  python src/analysis/sam_nested_spike.py
Out:  docs/analysis_results/sam_nested_spike/  (side-by-side viz per pair) + a printed verdict table.
"""

import os
import glob
import numpy as np
import torch
from PIL import Image, ImageDraw
from segment_anything import sam_model_registry, SamPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
FIP_DIR = os.path.join(ROOT, "input_plots", "fip")
SAM_CKPT = os.path.join(ROOT, "src", "mask_generation", "weights", "sam_vit_h_4b8939.pth")
OUT_DIR = os.path.join(ROOT, "docs", "analysis_results", "sam_nested_spike")

# how nested a pair must be to count: smaller box mostly inside bigger (IoS high) but not the same box (IoU low)
IOS_MIN = 0.6
IOU_MAX = 0.5
MAX_PAIRS = 10   # cap how many pairs we SAM (this is a quick spike)


def load_gt_boxes(txt_path, W, H):
    """Read YOLO-format GT (class cx cy w h, normalized) -> pixel xyxy array [N,4]."""
    boxes = []
    with open(txt_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, w, h = (float(v) for v in p[1:5])
            x1 = (cx - w / 2) * W; y1 = (cy - h / 2) * H
            x2 = (cx + w / 2) * W; y2 = (cy + h / 2) * H
            boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def box_iou_ios(a, b):
    """IoU and IoS (intersection-over-smaller) of two xyxy boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1]); ab = (b[2] - b[0]) * (b[3] - b[1])
    if inter <= 0:
        return 0.0, 0.0
    return inter / (aa + ab - inter), inter / min(aa, ab)


def find_nested_pairs(boxes):
    """Return (big_idx, small_idx, box_iou, box_ios) for pairs where the smaller box is mostly inside
    the bigger one (high IoS) but they're clearly not the same box (low IoU) -> the nested case."""
    pairs = []
    n = len(boxes)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for i in range(n):
        for j in range(i + 1, n):
            iou, ios = box_iou_ios(boxes[i], boxes[j])
            if ios >= IOS_MIN and iou <= IOU_MAX:
                big, small = (i, j) if areas[i] >= areas[j] else (j, i)
                pairs.append((big, small, iou, ios))
    return pairs


def mask_iou_ios(m1, m2):
    """IoU and IoS of two boolean masks."""
    inter = np.logical_and(m1, m2).sum()
    a1, a2 = m1.sum(), m2.sum()
    if inter == 0 or min(a1, a2) == 0:
        return 0.0, 0.0
    return inter / (a1 + a2 - inter), inter / min(a1, a2)


def box_center(b):
    """Center point (x,y) of an xyxy box — for a diagonal head this lands ON the head (the head
    crosses the rectangle's middle), while the neighbour sits in an empty corner."""
    return [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]


def sam_mask(predictor, box, pos_pt=None, neg_pt=None):
    """One SAM mask. Three prompt modes:
      - box only (pos_pt=None): 'segment what's in this rectangle' — can grab a neighbour in the corner
      - center point (pos_pt set): 'segment the head AT this spot' — ignores the neighbour
      - center point + negative (neg_pt set too): also tells SAM 'NOT that other head'."""
    if pos_pt is None:
        masks, _, _ = predictor.predict(box=np.array(box, dtype=np.float32), multimask_output=False)
        return masks[0].astype(bool)
    pts, labs = [pos_pt], [1]
    if neg_pt is not None:
        pts.append(neg_pt); labs.append(0)
    masks, _, _ = predictor.predict(point_coords=np.array(pts, dtype=np.float32),
                                    point_labels=np.array(labs, dtype=np.int32),
                                    multimask_output=False)
    return masks[0].astype(bool)


def save_viz(image, big, small, m_big, m_small, out_path):
    """Side-by-side: big-box+its mask (red), small-box+its mask (green), so you can SEE if SAM split them."""
    base = image.copy()
    over = np.array(base).astype(np.float32)
    over[m_big] = over[m_big] * 0.5 + np.array([255, 0, 0]) * 0.5      # big head mask = red
    over[m_small] = over[m_small] * 0.5 + np.array([0, 255, 0]) * 0.5  # small head mask = green
    img = Image.fromarray(over.astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle(list(big), outline=(255, 80, 80), width=3)
    d.rectangle(list(small), outline=(80, 255, 80), width=3)
    img.save(out_path, quality=92)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # collect labeled FIP images (those with a manual_label/*.txt)
    jobs = []
    for plot_dir in sorted(glob.glob(os.path.join(FIP_DIR, "plot_*"))):
        for txt in glob.glob(os.path.join(plot_dir, "manual_label", "*.txt")):
            stem = os.path.splitext(os.path.basename(txt))[0]
            for ext in (".png", ".jpg"):
                img_path = os.path.join(plot_dir, "images", stem + ext)
                if os.path.exists(img_path):
                    jobs.append((os.path.basename(plot_dir), stem, img_path, txt))
                    break
    print(f"Found {len(jobs)} labeled FIP images. Loading SAM on {DEVICE} ...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT).to(device=DEVICE)
    predictor = SamPredictor(sam)

    rows = []
    done = 0
    for plot, stem, img_path, txt in jobs:
        if done >= MAX_PAIRS:
            break
        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        boxes = load_gt_boxes(txt, W, H)
        pairs = find_nested_pairs(boxes)
        if not pairs:
            continue
        predictor.set_image(np.array(image))   # expensive encode, once per image
        for (bi, si, b_iou, b_ios) in pairs:
            if done >= MAX_PAIRS:
                break
            big, small = boxes[bi], boxes[si]
            cb, cs = box_center(big), box_center(small)
            # mask IoS under each prompt mode (lower = better separation)
            ios_box  = mask_iou_ios(sam_mask(predictor, big), sam_mask(predictor, small))[1]
            ios_pt   = mask_iou_ios(sam_mask(predictor, big, cb), sam_mask(predictor, small, cs))[1]
            mb_neg   = sam_mask(predictor, big,   cb, neg_pt=cs)   # big: 'this head, NOT the small one'
            ms_neg   = sam_mask(predictor, small, cs, neg_pt=cb)   # small: 'this head, NOT the big one'
            ios_neg  = mask_iou_ios(mb_neg, ms_neg)[1]
            # viz uses the strongest mode (point + negative neighbour)
            save_viz(image, big, small, mb_neg, ms_neg, os.path.join(OUT_DIR, f"{plot}_{stem}_{bi}_{si}.jpg"))
            rows.append((plot, ios_box, ios_pt, ios_neg))
            done += 1

    def sep(v): return "sep✓" if v < 0.5 else "BLOB"
    print(f"\n{'plot':<12}{'box':>7} {'  ':<4}{'point':>7} {'  ':<4}{'pt+neg':>7} {'  ':<4}  (mask IoS, lower=better)")
    print("-" * 64)
    for plot, ib, ip, ineg in rows:
        print(f"{plot:<12}{ib:>7.2f} {sep(ib):<4}{ip:>7.2f} {sep(ip):<4}{ineg:>7.2f} {sep(ineg):<4}")
    if rows:
        n = len(rows)
        for label, k in [("box", 1), ("point", 2), ("point+neg", 3)]:
            s = sum(1 for r in rows if r[k] < 0.5)
            print(f"  {label:<10}: {s}/{n} separated")
        print("GREEN LIGHT if point / point+neg separates clearly more than box.")
        print(f"Viz (point+neg masks) saved to {OUT_DIR}")
    else:
        print("No nested GT pairs found — loosen IOS_MIN/IOU_MAX or check labels.")


if __name__ == "__main__":
    main()
