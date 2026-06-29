"""
viz_sahi_tiles.py — draw the SAHI tile grid on an image so you can see, by eye, how many
tiles there are and how they overlap. Diagnostic only — touches nothing in the pipeline.

Each tile is drawn as a coloured outline + faint fill + an index label; overlap bands show up
where the faint fills stack (darker) and where outlines cross. Also prints the tile count.
"""

import os
import argparse
import numpy as np
import cv2

from mask_generation.sahi_yolo_sam.sahi_yolo_pipelined import compute_tile_boxes, compute_tile_boxes_dynamic

# distinct BGR colours cycled per tile
_COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0),
           (255, 0, 0), (255, 0, 255), (128, 0, 255), (0, 128, 255), (128, 255, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--slice", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.3)
    ap.add_argument("--dynamic", action="store_true", help="use resolution-adaptive tiling (sizes from longer side)")
    ap.add_argument("--target", type=int, default=1280, help="target tile size for --dynamic")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    h, w = img.shape[:2]
    if args.dynamic:
        tiles, args.slice = compute_tile_boxes_dynamic(w, h, args.target, args.overlap)
    else:
        tiles = compute_tile_boxes(w, h, args.slice, args.overlap)

    # 1) faint fills first, so the stacked overlap bands get visibly darker/more saturated
    fill = img.copy()
    for i, (x0, y0, x1, y1) in enumerate(tiles):
        c = _COLORS[i % len(_COLORS)]
        sub = fill[y0:y1, x0:x1]
        tint = np.full_like(sub, c)
        fill[y0:y1, x0:x1] = cv2.addWeighted(sub, 0.82, tint, 0.18, 0)
    vis = cv2.addWeighted(img, 0.4, fill, 0.6, 0)

    # 2) outlines + index labels on top
    for i, (x0, y0, x1, y1) in enumerate(tiles):
        c = _COLORS[i % len(_COLORS)]
        # nudge each rect inward by a couple px per index so coincident edges don't perfectly overlap
        p = (i % 4) * 4
        cv2.rectangle(vis, (x0 + p, y0 + p), (x1 - 1 - p, y1 - 1 - p), c, 5)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        cv2.putText(vis, str(i), (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 8, cv2.LINE_AA)
        cv2.putText(vis, str(i), (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 2.0, c, 4, cv2.LINE_AA)

    label = f"slice={args.slice}px  overlap={args.overlap}  ->  {len(tiles)} tiles  (image {w}x{h})"
    cv2.rectangle(vis, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(vis, label, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, vis)
    print(f"{len(tiles)} tiles -> {args.out}")


if __name__ == "__main__":
    main()
