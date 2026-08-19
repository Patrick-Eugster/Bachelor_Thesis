"""Empirically test whether per_tile SAM gives the SAME masks via "low conf floor then drop" vs
"run directly at conf t" — i.e. is the conf a pure post-filter, or does the tile crop (which is grown to
contain the boxes in a tile group) change the kept boxes' masks when low-conf boxes are also present?

For each image and threshold t it runs the REAL _infer_masks_per_tile twice:
  A = segment ALL boxes (low floor), keep the rows for boxes with conf >= t
  B = segment ONLY the boxes with conf >= t
and compares A vs B box-for-box (exact-equal + IoU). masks_full is indexed by input-box order, so the
kept rows of A line up 1:1 with B — the comparison is exact, no filename matching.

  identical  -> per_tile low-floor == direct  (the conf curve from ONE low-floor run is valid)
  differs    -> the tile-crop coupling bites; per_tile must be run at the keep line (report shows by how much)

Run:  python src/analysis/check_pertile_conf_equivalence.py \
        --pairs img1.jpg=bboxes_with_conf/img1.pt img2.jpg=bboxes_with_conf/img2.pt \
        --backend sam1 --thresholds 0.25 0.35 0.45
"""
import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mask_generation"))
from sam_v1.sam_v1_pipelined import _build_sam_backend, _infer_masks_dispatch  # noqa: E402


def _iou(a, b):
    """IoU of two binary masks (1.0 if both empty)."""
    a = a > 0; b = b > 0
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum()) / float(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="image.jpg=bboxes_with_conf.pt entries")
    ap.add_argument("--backend", default="sam1", choices=["sam1", "sam2"])
    ap.add_argument("--mode", default="per_tile", choices=["full_frame", "per_tile", "per_head"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.25, 0.35, 0.45])
    ap.add_argument("--weights_dir", default="src/mask_generation/weights")
    ap.add_argument("--sam_checkpoint", default="sam_vit_h_4b8939.pth")
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.2)
    args = ap.parse_args()

    cfg = OmegaConf.create({"method": {
        "sam_backend": args.backend, "sam_checkpoint": args.sam_checkpoint,
        "sam_crop_mode": args.mode, "sam1_decode_batch": 1,
        "sam_ul_decode_batch": 16, "sam_ul_chunk_on_oom": 64,
        "sam_tile_size": args.tile, "sam_tile_overlap": args.overlap,
        "sam_head_margin_frac": 0.4, "sam_head_min_pad": 16,
    }})
    print(f"Building SAM backend '{args.backend}' ...")
    backend, state = _build_sam_backend(cfg, args.weights_dir)

    worst = 0  # count of (image,threshold,box) rows that differ at all
    for pair in args.pairs:
        img_path, bbox_path = pair.split("=", 1)
        sam_image = np.array(Image.open(img_path).convert("RGB"))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        bwc = torch.load(bbox_path, weights_only=True)          # (N,5) x1 y1 x2 y2 conf
        boxes = bwc[:, :4].float().to(dev)                      # full_frame path feeds these to the GPU predictor
        conf = bwc[:, 4].float().cpu().numpy()
        # segment ALL boxes ONCE (this is the low-floor run)
        masks_all = _infer_masks_dispatch(backend, state, sam_image, boxes, cfg)[0]
        print(f"\n{os.path.basename(img_path)} [{args.mode}]: {len(boxes)} boxes (conf {conf.min():.3f}..{conf.max():.3f})")
        for t in args.thresholds:
            keep = conf >= t
            if keep.sum() == 0:
                print(f"  t={t:.2f}: no boxes >= t"); continue
            A = masks_all[keep]                                  # kept rows of the low-floor run
            keep_t = torch.from_numpy(keep).to(boxes.device)
            B = _infer_masks_dispatch(backend, state, sam_image, boxes[keep_t], cfg)[0]  # direct-at-t
            n = len(A)
            exact = sum(np.array_equal(A[i], B[i]) for i in range(n))
            ious = np.array([_iou(A[i], B[i]) for i in range(n)])
            diff = n - exact
            worst += diff
            flag = "IDENTICAL" if diff == 0 else f"DIFFERS ({diff}/{n})"
            print(f"  t={t:.2f}: kept={n:4d}  exact-equal={exact:4d}  "
                  f"min-IoU={ious.min():.4f} mean-IoU={ious.mean():.5f}  -> {flag}")

    print("\n" + "=" * 60)
    if worst == 0:
        print("RESULT: per_tile low-floor == direct  (identical masks) — conf is a safe post-filter.")
    else:
        print(f"RESULT: per_tile DIFFERS in {worst} box-rows total — the tile-crop coupling is real.")
        print("        -> build the per_tile conf curve at the keep line, not from one low-floor run.")


if __name__ == "__main__":
    main()
