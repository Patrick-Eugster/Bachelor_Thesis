"""SAM wrapper for the point-GT tool.

Two jobs:
  - seed_boxes: encode the full image ONCE and decode all YOLO/SAHI seed boxes in batches
    (fast draft for ~200 heads — same bounded-VRAM trick as sam_v1_pipelined's batched path).
  - refine_points: run SAM on a tight CROP around one head with positive/negative point prompts
    (the per-head crop is the resolution lever — a ~19px head fills the ~1024 encode when cropped).

Uses ultralytics SAM (sam2.1_l by default). Kept separate from the pipeline's sam_v1_pipelined.py so
the interactive tool can evolve without touching the production mask-gen code.
"""

import numpy as np
import cv2


class SamEngine:
    """Loads an ultralytics SAM model once and answers box-seed + point-refine requests."""

    def __init__(self, weight, decode_batch=8):
        """weight = path to a SAM checkpoint (e.g. sam2.1_l.pt); decode_batch bounds the seed-pass VRAM."""
        from ultralytics import SAM  # lazy: only import when the tool actually runs
        self.model = SAM(weight)
        self.decode_batch = int(decode_batch)   # kept for API compat; per-head crop path doesn't batch

    def seed_boxes(self, img_rgb, boxes_xyxy):
        """Seed a mask per box, decoding each on a TIGHT CROP around the box so the head fills the ~1024
        encode (precise) instead of being ~20 px on the full 4032 frame (bloated). One encode per head —
        slower than the full-image path but far tighter, and the tool caches it so it's a one-time cost.
        Returns a list of bool HxW masks (one per box, aligned to the full image)."""
        import torch
        H, W = img_rgb.shape[:2]
        boxes_np = np.asarray(boxes_xyxy, dtype=np.float32)
        if len(boxes_np) == 0:
            return []
        masks = []
        for bx in boxes_np:
            masks.append(self._crop_decode_box(img_rgb, bx, H, W))
        torch.cuda.empty_cache()
        return masks

    def _crop_decode_box(self, img_rgb, box, H, W, margin_frac=0.4, min_pad=16):
        """Crop a padded window around one box and segment the box on that crop, then paste the mask back
        into full-image coords. Uses the high-level model.predict (it scales the box to the crop correctly;
        the low-level prompt_inference mis-scales boxes on small crops → degenerate masks). The head filling
        the crop is what makes the mask tight."""
        x0b, y0b, x1b, y1b = box
        pad = max(min_pad, int(margin_frac * max(x1b - x0b, y1b - y0b)))
        x0 = max(0, int(x0b - pad)); y0 = max(0, int(y0b - pad))
        x1 = min(W, int(x1b + pad)); y1 = min(H, int(y1b + pad))
        crop_bgr = np.ascontiguousarray(img_rgb[y0:y1, x0:x1, ::-1])
        box_local = np.array([[x0b - x0, y0b - y0, x1b - x0, y1b - y0]], dtype=np.float32)
        r = self.model.predict(crop_bgr, bboxes=box_local, verbose=False, save=False)
        md = r[0].masks
        full = np.zeros((H, W), dtype=bool)
        if md is not None and md.data is not None and len(md.data):
            m = (md.data[0].detach().cpu().numpy() > 0.5)
            ch, cw = y1 - y0, x1 - x0
            if m.shape != (ch, cw):
                m = cv2.resize(m.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST).astype(bool)
            full[y0:y1, x0:x1] = m
        return full

    def refine_points(self, img_rgb, points_xy, labels, box_hint=None, margin_frac=0.35, min_pad=24):
        """Segment ONE head from positive/negative points, on a tight crop around it.
        points_xy = (N,2) full-image pixel coords, labels = (N,) with 1=positive / 0=negative.
        box_hint = optional [x0,y0,x1,y1] to anchor the crop (the seed box). Returns a bool HxW mask."""
        import torch
        H, W = img_rgb.shape[:2]
        pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        lbl = np.asarray(labels, dtype=np.int32).reshape(-1)
        if len(pts) == 0:
            return np.zeros((H, W), dtype=bool)

        # crop = bounding box of (box_hint ∪ all points) + a margin, clamped to the image
        xs = list(pts[:, 0]); ys = list(pts[:, 1])
        if box_hint is not None:
            xs += [box_hint[0], box_hint[2]]; ys += [box_hint[1], box_hint[3]]
        else:
            min_pad = max(min_pad, 130)   # no seed box (new head): a single point needs a crop big
                                          # enough to actually contain a wheat head, else SAM sees a sliver
        bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
        pad = max(min_pad, int(margin_frac * max(bx1 - bx0, by1 - by0)))
        x0 = max(0, int(bx0 - pad)); y0 = max(0, int(by0 - pad))
        x1 = min(W, int(bx1 + pad)); y1 = min(H, int(by1 + pad))
        crop_bgr = np.ascontiguousarray(img_rgb[y0:y1, x0:x1, ::-1])   # -> BGR
        # high-level predict scales the prompts to the crop correctly (the low-level prompt_inference
        # mis-scales on small crops → degenerate masks). Nested shape (1,N,*) = ONE object, N points.
        P = (pts - np.array([x0, y0], dtype=np.float32))[None, ...]    # (1,N,2)
        L = lbl[None, ...]                                            # (1,N)
        r = self.model.predict(crop_bgr, points=P, labels=L, verbose=False, save=False)
        md = r[0].masks
        full = np.zeros((H, W), dtype=bool)
        if md is not None and md.data is not None and len(md.data):
            m = (md.data[0].detach().cpu().numpy() > 0.5)
            ch, cw = y1 - y0, x1 - x0
            if m.shape != (ch, cw):
                m = cv2.resize(m.astype(np.uint8), (cw, ch),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            full[y0:y1, x0:x1] = m
        torch.cuda.empty_cache()
        return full
