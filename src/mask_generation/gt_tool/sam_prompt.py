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
        """Segment ONE head and return the auto-picked best mask (bool HxW). Thin wrapper over
        refine_points_all — kept for callers that just want the single best mask."""
        masks, idx = self.refine_points_all(img_rgb, points_xy, labels, box_hint, margin_frac, min_pad)
        if not masks:
            H, W = img_rgb.shape[:2]
            return np.zeros((H, W), dtype=bool)
        return masks[idx]

    def refine_points_all(self, img_rgb, points_xy, labels, box_hint=None, margin_frac=0.35, min_pad=24):
        """Segment ONE head from positive/negative points on a tight crop, and return SAM2's 3 candidate
        masks (each a bool HxW aligned to the full image) PLUS the index of the auto-picked best one —
        fewest ⊖ negatives inside, then fewest ⊕ positives outside, then highest SAM score. This lets the
        UI show/cycle the candidates instead of only the auto pick. Empty points → ([], 0). Falls back to a
        single-mask predict (a 1-element list) if the low-level multimask path ever errors.
        points_xy = (N,2) full-image pixel coords, labels = (N,) with 1=positive / 0=negative;
        box_hint = optional [x0,y0,x1,y1] to anchor the crop (the seed box)."""
        import torch
        H, W = img_rgb.shape[:2]
        pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        lbl = np.asarray(labels, dtype=np.int32).reshape(-1)
        if len(pts) == 0:
            return [], 0

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
        # Nested shape (1,N,*) = ONE object, N points. Points shifted into crop-local coords.
        pts_local = pts - np.array([x0, y0], dtype=np.float32)
        P = pts_local[None, ...]                                       # (1,N,2)
        L = lbl[None, ...]                                             # (1,N)
        cand_masks, idx = self._multimask_all(crop_bgr, P, L, pts_local, lbl)   # crop-space bool masks
        ch, cw = y1 - y0, x1 - x0
        full_list = []
        for m in cand_masks:                                          # paste each candidate back full-size
            full = np.zeros((H, W), dtype=bool)
            if m is not None:
                if m.shape != (ch, cw):
                    m = cv2.resize(m.astype(np.uint8), (cw, ch),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                full[y0:y1, x0:x1] = m
            full_list.append(full)
        torch.cuda.empty_cache()
        return full_list, idx

    def _predictor(self):
        """Return the ultralytics predictor, priming it with a tiny predict on first use (the predictor
        object only exists after the first model.predict call)."""
        if self.model.predictor is None:
            self.model.predict(np.zeros((64, 64, 3), np.uint8),
                               points=np.array([[[32, 32]]], np.float32),
                               labels=np.array([[1]], np.int32), verbose=False, save=False)
        return self.model.predictor

    def _multimask_all(self, crop_bgr, P, L, pts_local, lbl):
        """Run SAM2 on the crop with multimask_output=True and return (list of candidate bool masks in crop
        coords, index of the best-respecting one). Mirrors ultralytics' own preprocess + prompt-scaling +
        scale_masks so the masks land in crop coords exactly like the high-level predict (verified:
        candidate[0] area == high-level single-mask area). Any failure → a 1-element single-mask fallback
        so the tool never breaks."""
        try:
            import ultralytics.utils.ops as ops
            pred = self._predictor()
            pred.setup_source(crop_bgr)                       # configures imgsz + letterbox for this crop
            im = None
            for batch in pred.dataset:
                im = pred.preprocess(batch[1]); pred.batch = batch; break
            feats = pred.get_im_features(im)
            # SAM2 _prepare_prompts scales the prompts to the model input using the crop's real shape
            prompts = pred._prepare_prompts(im.shape[2:], pred.batch[1][0].shape[:2], points=P, labels=L)
            pm, ps = pred._inference_features(feats, *prompts, multimask_output=True)   # (3,256,256),(3,)
            ch, cw = crop_bgr.shape[:2]
            sm = ops.scale_masks(pm[None].float(), (ch, cw), padding=False)[0] > pred.model.mask_threshold
            sm = sm.detach().cpu().numpy()
            idx = self._select_candidate(sm, ps.flatten().detach().cpu().numpy(), pts_local, lbl)
            return [sm[i] for i in range(sm.shape[0])], idx
        except Exception as e:
            print(f"[gt_tool] multimask refine failed ({type(e).__name__}: {e}); single-mask fallback")
            r = self.model.predict(crop_bgr, points=P, labels=L, verbose=False, save=False)
            md = r[0].masks
            if md is not None and md.data is not None and len(md.data):
                return [md.data[0].detach().cpu().numpy() > 0.5], 0
            return [], 0

    @staticmethod
    def _select_candidate(masks_bool, scores, pts_local, lbl):
        """Pick the candidate that best honours the prompts: fewest NEGATIVE points inside it, then fewest
        POSITIVE points outside it, then highest SAM quality score. This is what makes ⊖ points actually
        carve a region out — the tightest candidate that excludes the neighbour wins over the whole-blob one."""
        C, ch, cw = masks_bool.shape
        pos = [(int(round(x)), int(round(y))) for (x, y), l in zip(pts_local, lbl) if l == 1]
        neg = [(int(round(x)), int(round(y))) for (x, y), l in zip(pts_local, lbl) if l == 0]

        def inside(mk, x, y):
            return 0 <= y < ch and 0 <= x < cw and bool(mk[y, x])

        best, best_key = 0, None
        for c in range(C):
            mk = masks_bool[c]
            neg_hits = sum(inside(mk, x, y) for x, y in neg)
            pos_miss = sum(not inside(mk, x, y) for x, y in pos)
            key = (neg_hits, pos_miss, -float(scores[c]))     # lexicographic: respect negs > cover pos > score
            if best_key is None or key < best_key:
                best_key, best = key, c
        return best
