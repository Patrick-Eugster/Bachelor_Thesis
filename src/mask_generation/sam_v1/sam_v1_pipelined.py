"""
sam_v1_pipelined.py — pipelined SAM inference (upgraded from sam_v1.py)

HOW IT WORKS:
Each image goes through 3 stages:

    Stage 1 — Pre-Processing (CPU):  load image + bbox tensor from disk
    Stage 2 — GPU Inference:         SAM image encoding (set_image) + mask decoding (predict_torch)
    Stage 3 — Post-Processing (CPU): save individual mask PNGs + overlay visualization

In sam_v1 these 3 stages ran sequentially per image. In the pipelined version:

    while GPU encodes image N   →  CPU loads image N+1      (background thread)
                                →  CPU saves masks N-1      (background thread)

This keeps the GPU continuously busy and hides most of the disk I/O behind the SAM encoder,
which is the bottleneck (~1-2s per image for ViT-H).

CONCURRENCY STRUCTURE:
- One outer ThreadPoolExecutor (max_workers=2): one slot for the load future, one for the save future.
- The save task spawns its own inner pool (MAX_THREADS threads) for parallel mask PNG writing.
- The main thread runs GPU inference (set_image + predict_torch).
- torch.cuda.synchronize() blocks the main thread but background threads keep running — they are
  independent OS threads unaffected by the main thread waiting for the GPU.
"""

import os
import glob
import time
import concurrent.futures
import gc
import json
import resource
import numpy as np
import cv2
from PIL import Image
import torch
import colorsys
import shutil
import wandb
from segment_anything import sam_model_registry, SamPredictor

from wheat_utils.path_utils import get_mask_generation_result_path
from mask_generation.roi_mask import apply_roi, roi_crop_box

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =====================================================================
#-------- HELPER FUNCTIONS FOR SAM --------
# =====================================================================

def reset_folder(folder_path):
    """Deletes all contents of a folder and recreates it,
    since thats cheaper than to go through the folder and delete every single item."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Deletes the folder and everything inside
    os.makedirs(folder_path, exist_ok=True)  # Recreates the empty folder


def id2rgb(id, max_num_obj=65535):
    """Color Generator, turns a single number into a specific rgb color,
    dynamically handles any ID number with a high maximum limit."""
    if id == 0:  # invalid region / background
        return np.zeros((3, ), dtype=np.uint8)
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")
    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)  # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5  # Alternate between 0.5 and 1.0
    l = 0.5
    # Use colorsys to convert HSL to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return np.array([int(r*255), int(g*255), int(b*255)], dtype=np.uint8)


def visualize_obj(objects):
    """For visualize color mask, basically every wheat head another color."""
    assert len(objects.shape) == 2
    # Create the blank RGB canvas
    rgb_mask = np.zeros((*objects.shape, 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)  # Get all unique IDs present in the image
    for id in all_obj_ids:
        if id == 0:
            continue  # Skip the background
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask


def print_sam_step_report(idx, total_imgs, name, n_heads, t_embed, t_pred):
    """Print a per-image timing line. t_save is excluded — it runs async in the background."""
    print(f"  [{idx+1}/{total_imgs}] {name:<20} | "
          f"Embed: {t_embed:>5.2f}s | Pred: {t_pred:>5.2f}s | "
          f"Heads: {n_heads:>3}")


def print_sam_plot_summary(num_images, total_time):
    """Prints the final summary for the entire plot's SAM processing."""
    print(f"\n" + "="*45)
    print(f"      SAM PLOT SUMMARY ({num_images} images)")
    print("="*45)
    print(f"{'Total SAM Time (pipelined):':<25} {total_time:>8.2f}s")
    print(f"{'Average Time Per Image:':<25} {total_time/num_images:>8.2f}s")
    print("="*45 + "\n")


# =====================================================================
# -------- PIPELINE HELPERS (new in pipelined version) --------
# =====================================================================

def _load_image_and_bbox(image_file, bbox_folder, cfg):
    """Load one image from disk and its corresponding bbox .pt tensor.
    Returns (image_name, save_name, image_rgb, bbox_or_None, crop_or_None) — bbox None signals a
    missing bbox file; crop is the (x0,y0,x1,y1) ROI-crop rect (None unless roi.crop_before_inference)."""
    image_name = os.path.basename(image_file)
    save_name = os.path.splitext(image_name)[0]
    # Load with PIL (not cv2.imread): both use libjpeg-turbo so pixels are equivalent, but PIL decodes
    # the phone JPEGs silently whereas cv2 prints "Invalid SOS parameters for sequential JPEG" on the
    # ~half that carry a non-standard SOS header. PIL also returns RGB directly (no BGR->RGB needed) and
    # matches how YOLO/SAHI load their images. We do NOT exif_transpose, so orientation stays as before.
    image = np.array(Image.open(image_file).convert("RGB"))
    # ROI: grey-out outside the plot polygon (buffered, no-op unless roi.enabled) so SAM masks
    # don't bleed into neighbour-plot wheat. The boxes were already ROI-filtered in the YOLO/SAHI phase.
    image = apply_roi(image, image_file, cfg)
    # ROI-crop: rect to crop the frame to the plot so SAM sees it at higher res (no-op unless
    # roi.crop_before_inference). Boxes are offset into crop coords in the GPU step; masks pasted back.
    h, w = image.shape[:2]
    crop = roi_crop_box(image_file, cfg, w, h)
    bbox_path = os.path.join(bbox_folder, save_name + ".pt")
    if not os.path.exists(bbox_path):
        return image_name, save_name, image, None, crop  # None signals missing bbox
    bbox = torch.load(bbox_path)
    return image_name, save_name, image, bbox, crop


def _save_image_results(masks_np, image, save_name, image_name, mask_folder, sam_vis_folder,
                        crop, max_threads, save_masks=True, save_union=False):
    """Save all mask PNGs (unless save_masks=False) and the overlay visualization for one image.
    Returns t_save. With an ROI crop, masks_np are in CROP coords → each is pasted into a full-frame
    canvas before writing so the saved PNGs + overlay stay aligned to the original image (the 3D
    segmentation reads full-image masks). Building the label map on the full frame + expanding masks
    one-at-a-time in the save threads avoids ever holding N full-size masks in RAM at once.
    save_union=True writes ONE binary foreground union PNG per image (<save_name>_union.png) instead of
    thousands of per-head PNGs — enough for the SAM-backend A/B (binary IoU) and avoids the page-cache
    balloon from writing ~8k full-res masks per backend."""
    t_start_save = time.perf_counter()
    H, W = image.shape[:2]
    if crop is not None:
        cx0, cy0, cx1, cy1 = crop

    def _expand(m):
        """Crop mask -> full-frame canvas (identity when there's no crop)."""
        if crop is None:
            return m
        full = np.zeros((H, W), dtype=m.dtype)
        full[cy0:cy1, cx0:cx1] = m
        return full

    # 4. Saving & Visualization (Parallel CPU Saving) — label map is always full-frame
    objects = np.zeros((H, W), dtype=np.uint16)
    save_tasks = []

    for idx, mask_np in enumerate(masks_np):
        if crop is None:
            objects[mask_np > 0] = idx + 1
        else:
            objects[cy0:cy1, cx0:cx1][mask_np > 0] = idx + 1  # view assignment writes back into objects
        if save_masks:
            out_path = os.path.join(mask_folder, f"{save_name}_{idx:03}.png")
            save_tasks.append((out_path, mask_np))

    if save_tasks:  # skipped entirely when save_masks=False (fast A/B: overlay + counts only)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            list(executor.map(lambda arg: cv2.imwrite(arg[0], _expand(arg[1])), save_tasks))

    # one binary union foreground PNG per image (for the SAM-backend A/B scorer) — cheap, avoids the
    # thousands-of-per-head-PNGs page-cache balloon. `_union` suffix won't match the per-head glob.
    if save_union:
        cv2.imwrite(os.path.join(mask_folder, f"{save_name}_union.png"),
                    ((objects > 0).astype(np.uint8) * 255))

    # Reverted to your highly efficient sparse NumPy math
    color_mask = visualize_obj(objects) / 255.0
    color_img = image / 255.0
    non_black_pixels = np.any(color_mask > 0, axis=-1)

    overlayed_img = color_img.copy()
    alpha = 0.5
    overlayed_img[non_black_pixels, :] = (alpha * color_mask[non_black_pixels, :] +
                                          (1 - alpha) * color_img[non_black_pixels, :])

    cv2.imwrite(
        os.path.join(sam_vis_folder, image_name.replace(".png", ".jpg")),
        (overlayed_img * 255).astype(np.uint8)[:, :, ::-1]
    )

    return time.perf_counter() - t_start_save


# =====================================================================
# -------- SAM BACKEND SWITCH (SAM1 / SAM2 / SAM3) --------
# =====================================================================

def _build_sam_backend(cfg, weights_dir):
    """Build the chosen SAM backend and return (backend_name, state_dict).
    'sam1' (default) = the original ViT-H SamPredictor — byte-identical to before.
    'sam2'/'sam3'    = Meta SAM2/SAM3 loaded through ultralytics (already installed; NO dependency
                       change — the classes ship in our ultralytics build). Only the segmenter
                       changes; the box prompts + images + eval are held fixed so the A/B is fair."""
    backend = cfg.method.get("sam_backend", "sam1")
    if backend == "sam1":
        ckpt = os.path.join(weights_dir, cfg.method.sam_checkpoint)
        sam = sam_model_registry["vit_h"](checkpoint=ckpt).to(device=DEVICE)
        sam = torch.compile(sam)
        predictor = SamPredictor(sam)
        torch.cuda.synchronize()
        return backend, {"sam": sam, "predictor": predictor}
    if backend in ("sam2", "sam3"):
        from ultralytics import SAM  # lazy import: only when actually used (default sam1 never imports it)
        key        = "sam_sam2_weight" if backend == "sam2" else "sam_sam3_weight"
        default_w  = "sam2.1_l.pt"     if backend == "sam2" else "sam3.pt"   # large by default (best quality)
        weight     = cfg.method.get(key, default_w)
        wpath      = os.path.join(weights_dir, weight)
        # if the weight sits in our weights/ dir use it; else hand the bare name to ultralytics
        # (sam2 auto-downloads; sam3 is GATED so it must already be on disk — see the run doc)
        model = SAM(wpath if os.path.exists(wpath) else weight)
        chunk = int(cfg.method.get("sam_ul_chunk_on_oom", 64))  # box-chunk size for the OOM fallback
        # 0 = original all-at-once decode (default, unchanged); >0 = encode-once + decode in batches of
        # this many boxes (bounds VRAM — fixes SAM2's spill; does not help SAM3, encoder alone ~26GB).
        decode_batch = int(cfg.method.get("sam_ul_decode_batch", 0))
        return backend, {"model": model, "chunk": chunk, "decode_batch": decode_batch}
    raise ValueError(f"unknown sam_backend '{backend}' (expected sam1 / sam2 / sam3)")


def _ultra_predict_boxes(model, img_bgr, boxes_np, hw, chunk_on_oom=64):
    """ultralytics SAM (sam2/sam3) box-prompt inference for ONE image, OOM-safe.
    Tries ALL boxes in one predict call (encodes once → fast + fair timing). On CUDA OOM — FIP has up
    to ~300 heads on a 4k frame, and ultralytics decodes every box's full-res mask at once, which can
    exceed a 16 GB GPU — it falls back to decoding the boxes in chunks (re-encodes per chunk, slower)
    so the run COMPLETES instead of crashing. Returns (masks_np [N,H,W] uint8 0/255, chunked_flag)."""
    Hc, Wc = hw

    def _decode(res):
        """One predict Result -> (N,H,W) uint8 masks aligned to the input image."""
        r = res[0]
        if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
            return np.zeros((0, Hc, Wc), dtype=np.uint8)
        m = r.masks.data.detach().cpu().numpy()          # (N, h, w), box-prompt order
        mb = (m > 0.5).astype(np.uint8) * 255
        if mb.shape[1:] != (Hc, Wc) and len(mb):         # upsample to input size if ultralytics didn't
            mb = np.stack([cv2.resize(x, (Wc, Hc), interpolation=cv2.INTER_NEAREST) for x in mb])
        return mb

    if len(boxes_np) == 0:
        return np.zeros((0, Hc, Wc), dtype=np.uint8), False
    try:
        return _decode(model.predict(img_bgr, bboxes=boxes_np, verbose=False, save=False)), False
    except torch.cuda.OutOfMemoryError:
        # bounded-VRAM fallback: decode the boxes in chunks (each predict re-encodes the image)
        torch.cuda.empty_cache()
        gc.collect()
        parts = []
        for i in range(0, len(boxes_np), chunk_on_oom):
            parts.append(_decode(model.predict(img_bgr, bboxes=boxes_np[i:i + chunk_on_oom],
                                               verbose=False, save=False)))
            torch.cuda.empty_cache()
        masks_np = np.concatenate(parts, axis=0) if parts else np.zeros((0, Hc, Wc), dtype=np.uint8)
        print(f"    [OOM-safe] decoded {len(boxes_np)} boxes in chunks of {chunk_on_oom} "
              f"(all-at-once OOMed) — timing for this image includes extra re-encodes")
        return masks_np, True


def _ultra_predict_boxes_batched(model, img_bgr, boxes_np, hw, batch=32):
    """SEPARATE opt-in path (sam_ul_decode_batch>0): ENCODE the image ONCE, then decode the boxes in
    batches — instead of the default all-at-once `model.predict` that upsamples every box's full-res
    mask together (the 23 GB spike). Bounds VRAM to encoder + one batch of masks.
      Measured: SAM2 encoder ~5 GB → this keeps SAM2 at ~7 GB (fits 16 GB, no WSL spill = fast).
      SAM3 encoder alone is ~26 GB, so this does NOT rescue SAM3 (a hardware wall, not a code issue).
    The default all-at-once path (_ultra_predict_boxes) is left untouched — this only runs when the
    sam_ul_decode_batch knob is set."""
    Hc, Wc = hw
    if len(boxes_np) == 0:
        return np.zeros((0, Hc, Wc), dtype=np.uint8)
    # lazily init the predictor (one tiny predict creates model.predictor + loads the model)
    if getattr(model, "predictor", None) is None:
        model.predict(img_bgr, bboxes=boxes_np[:1], verbose=False, save=False)
    pred = model.predictor
    pred.set_image(img_bgr)                                   # ENCODE ONCE (features cached on pred)
    im = pred.preprocess(pred.batch[1])                       # preprocessed tensor for decode/postprocess
    parts = []
    for i in range(0, len(boxes_np), batch):                 # decode a batch at a time (reuses features)
        preds = pred.prompt_inference(im, bboxes=boxes_np[i:i + batch], multimask_output=False)
        results = pred.postprocess(preds, im, [img_bgr])      # upsample this batch to full res
        md = results[0].masks
        if md is not None and md.data is not None and len(md.data):
            m = md.data.detach().cpu().numpy()
            mb = (m > 0.5).astype(np.uint8) * 255
            if mb.shape[1:] != (Hc, Wc) and len(mb):
                mb = np.stack([cv2.resize(x, (Wc, Hc), interpolation=cv2.INTER_NEAREST) for x in mb])
            parts.append(mb)
        del preds, results, md
        torch.cuda.empty_cache()
    pred.reset_image()
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, Hc, Wc), dtype=np.uint8)


def _infer_masks(backend, state, sam_image, bbox_in):
    """Run one image's boxes through the chosen SAM backend. Returns
    (masks_np [N,H,W] uint8 0/255, t_embed, t_pred), masks aligned to sam_image. The image + boxes are
    identical across backends (fair swap) — only the model differs. SAM2/3 report t_embed=0 because
    ultralytics encodes+decodes in one call (per-image total time t_embed+t_pred stays comparable)."""
    Hc, Wc = sam_image.shape[:2]

    if backend == "sam1":
        predictor = state["predictor"]
        # 2. Image Embedding (heavy part). The encoder always takes one image at a time.
        t0 = time.perf_counter()
        predictor.set_image(sam_image)
        torch.cuda.synchronize()
        t_embed = time.perf_counter() - t0

        # 3. Predict Masks — one box at a time through the decoder. The decoder *can* batch boxes, but
        # with hundreds of heads on a full-res image that spikes VRAM (each box -> a full H x W mask),
        # so we keep it one-at-a-time which is the safe, proven default.
        t1 = time.perf_counter()
        transformed_boxes = predictor.transform.apply_boxes_torch(bbox_in, sam_image.shape[:2])
        all_masks_np = []
        with torch.no_grad():
            for b_idx in range(len(transformed_boxes)):
                single_box = transformed_boxes[b_idx : b_idx + 1]  # keep the [1,4] shape
                masks, _, _ = predictor.predict_torch(
                    point_coords=None, point_labels=None,
                    boxes=single_box, multimask_output=False)
                # squeeze(1) -> [1, H, W]; move to CPU + uint8 immediately to free VRAM
                all_masks_np.append((masks.squeeze(1).cpu().numpy() * 255).astype(np.uint8))
        masks_np = (np.concatenate(all_masks_np, axis=0) if all_masks_np
                    else np.zeros((0, Hc, Wc), dtype=np.uint8))
        torch.cuda.synchronize()
        t_pred = time.perf_counter() - t1
        predictor.reset_image()
        return masks_np, t_embed, t_pred

    # --- sam2 / sam3 via ultralytics ---
    model = state["model"]
    boxes_np = bbox_in.detach().cpu().numpy()
    # ultralytics takes numpy images as BGR (cv2 convention); our sam_image is RGB -> convert.
    img_bgr = np.ascontiguousarray(sam_image[:, :, ::-1])
    t0 = time.perf_counter()
    if state.get("decode_batch", 0) > 0:
        # opt-in encode-once + batched decode (bounds VRAM — the SAM2 fix). Default path untouched below.
        masks_np = _ultra_predict_boxes_batched(model, img_bgr, boxes_np, (Hc, Wc),
                                                batch=state["decode_batch"])
    else:
        # DEFAULT (original): one call encodes + decodes all boxes at once (OOM-safe chunk fallback).
        masks_np, _chunked = _ultra_predict_boxes(model, img_bgr, boxes_np, (Hc, Wc),
                                                  chunk_on_oom=state.get("chunk", 64))
    torch.cuda.synchronize()
    t_pred = time.perf_counter() - t0
    return masks_np, 0.0, t_pred


# =====================================================================
# -------- SAM GRANULARITY (full_frame / per_tile / per_head) --------
# how much of the frame SAM ENCODES per head → the head's resolution at SAM's fixed ~1024 encode.
#   full_frame : encode the whole frame ONCE, decode all boxes   (default, byte-identical to before)
#   per_tile   : tile the frame, encode each tile, decode its boxes  (~tile-res heads)
#   per_head   : encode a tight crop PER head, decode its box        (~1024px heads — quality ceiling)
# Both new modes reuse the SAME backend decoders as full_frame, so SAM1/SAM2/SAM3 all work unchanged.
# =====================================================================

def _backend_masks_on_crop(backend, state, crop_rgb, boxes_local):
    """Run the chosen SAM backend on ONE crop with boxes already in crop-local xyxy px. Returns
    [K, ch, cw] uint8 0/255 (aligned to the crop). SAM always encodes at ~1024 internally, so a tight
    crop makes the head fill the encode = the resolution lever."""
    ch, cw = crop_rgb.shape[:2]
    boxes_local = np.asarray(boxes_local, dtype=np.float32).reshape(-1, 4)
    if len(boxes_local) == 0:
        return np.zeros((0, ch, cw), dtype=np.uint8)
    if backend == "sam1":
        predictor = state["predictor"]
        predictor.set_image(crop_rgb)                        # encode this crop (ResizeLongestSide→1024²)
        tb = predictor.transform.apply_boxes_torch(
            torch.as_tensor(boxes_local, dtype=torch.float, device=DEVICE), (ch, cw))
        out = []
        with torch.no_grad():
            for k in range(len(tb)):
                m, _, _ = predictor.predict_torch(point_coords=None, point_labels=None,
                                                  boxes=tb[k:k + 1], multimask_output=False)
                out.append((m.squeeze(1).cpu().numpy() * 255).astype(np.uint8))
        predictor.reset_image()
        return np.concatenate(out, axis=0) if out else np.zeros((0, ch, cw), dtype=np.uint8)
    # sam2 / sam3 via ultralytics — same OOM-safe decoder as full_frame, just on the crop
    model = state["model"]
    crop_bgr = np.ascontiguousarray(crop_rgb[:, :, ::-1])
    masks_np, _ = _ultra_predict_boxes(model, crop_bgr, boxes_local, (ch, cw),
                                       chunk_on_oom=state.get("chunk", 64))
    return masks_np


def _infer_masks_per_head(backend, state, sam_image, bbox_in, margin_frac=0.4, min_pad=16):
    """Encode a tight padded crop around EACH box and decode just that box, then paste the mask into a
    full-frame canvas. One encode per head (slow) but the head fills SAM's ~1024 encode → tight masks.
    Same return contract as _infer_masks: (masks_np [N,H,W] uint8, t_embed=0, t_pred)."""
    H, W = sam_image.shape[:2]
    boxes = bbox_in.detach().cpu().numpy().astype(np.float32)
    masks_full = np.zeros((len(boxes), H, W), dtype=np.uint8)
    t0 = time.perf_counter()
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        pad = max(min_pad, int(margin_frac * max(x1 - x0, y1 - y0)))
        cx0, cy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
        cx1, cy1 = min(W, int(x1 + pad)), min(H, int(y1 + pad))
        crop = sam_image[cy0:cy1, cx0:cx1]
        box_local = np.array([[x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0]], dtype=np.float32)
        m = _backend_masks_on_crop(backend, state, crop, box_local)
        if len(m):
            masks_full[i, cy0:cy1, cx0:cx1] = m[0]
    return masks_full, 0.0, time.perf_counter() - t0


def _infer_masks_per_tile(backend, state, sam_image, bbox_in, tile=1280, overlap=0.2, pad_frac=0.02):
    """Group boxes by tile cell (by centre), then encode ONE crop per non-empty tile and decode all its
    boxes. The crop = the tile cell grown to `tile` px AND expanded to fully contain its boxes (+pad), so
    no head is cut. ~tile-resolution heads for far fewer encodes than per_head. Same return contract."""
    from collections import defaultdict
    H, W = sam_image.shape[:2]
    boxes = bbox_in.detach().cpu().numpy().astype(np.float32)
    N = len(boxes)
    masks_full = np.zeros((N, H, W), dtype=np.uint8)
    if N == 0:
        return masks_full, 0.0, 0.0
    step = max(1, int(tile * (1 - overlap)))
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    groups = defaultdict(list)
    for i in range(N):
        groups[(int(cx[i] // step), int(cy[i] // step))].append(i)   # unique tile cell per box centre

    t0 = time.perf_counter()
    for (gx, gy), idxs in groups.items():
        gb = boxes[idxs]
        tx0, ty0 = gx * step, gy * step
        # crop = tile cell (grown to `tile`) ∪ this group's boxes, + a small pad, clamped to the frame
        x0 = min(tx0, gb[:, 0].min()); y0 = min(ty0, gb[:, 1].min())
        x1 = max(tx0 + tile, gb[:, 2].max()); y1 = max(ty0 + tile, gb[:, 3].max())
        pad = int(pad_frac * tile)
        cx0, cy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
        cx1, cy1 = min(W, int(x1 + pad)), min(H, int(y1 + pad))
        crop = sam_image[cy0:cy1, cx0:cx1]
        bl = gb.copy(); bl[:, [0, 2]] -= cx0; bl[:, [1, 3]] -= cy0
        m = _backend_masks_on_crop(backend, state, crop, bl)
        for j, gi in enumerate(idxs):
            if j < len(m):
                masks_full[gi, cy0:cy1, cx0:cx1] = m[j]
    return masks_full, 0.0, time.perf_counter() - t0


def _infer_masks_dispatch(backend, state, sam_image, bbox_in, cfg):
    """Pick the SAM granularity from cfg.method.sam_crop_mode (default full_frame = unchanged path)."""
    mode = cfg.method.get("sam_crop_mode", "full_frame")
    if mode == "per_head":
        return _infer_masks_per_head(backend, state, sam_image, bbox_in,
                                     margin_frac=float(cfg.method.get("sam_head_margin_frac", 0.4)),
                                     min_pad=int(cfg.method.get("sam_head_min_pad", 16)))
    if mode == "per_tile":
        return _infer_masks_per_tile(backend, state, sam_image, bbox_in,
                                     tile=int(cfg.method.get("sam_tile_size", 1280)),
                                     overlap=float(cfg.method.get("sam_tile_overlap", 0.2)),
                                     pad_frac=float(cfg.method.get("sam_tile_pad_frac", 0.02)))
    return _infer_masks(backend, state, sam_image, bbox_in)   # full_frame (default, byte-identical)


# =====================================================================
# MAIN SAM INFERENCE (ALL PLOTS)
# =====================================================================
def run_sam_phase(image_folders, cfg):
    print("\n" + "="*50)
    print(" PHASE 2: LOADING SAM AND PROCESSING ALL PLOTS")
    print("="*50)

    weights_dir = os.path.join(os.path.dirname(__file__), "..", "weights")

    if cfg.wandb_enabled:
        wandb.init(
            project="wheat3dgs-sam-v1",
            config={"device": DEVICE},
        )

    # reset the CUDA peak counter so max_memory_allocated below is THIS SAM phase's true peak VRAM
    # (includes the model load + inference). Each backend runs in its own process → clean per-backend #.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Load the chosen SAM backend ONCE (sam1 default = original ViT-H SamPredictor, byte-identical)
    print("Loading SAM (this takes a few seconds)...")
    start_sam_load = time.perf_counter()
    backend, state = _build_sam_backend(cfg, weights_dir)
    sam_load_time = time.perf_counter() - start_sam_load
    print(f"-> SAM backend '{backend}' loaded on {DEVICE} in {sam_load_time:.2f}s")

    # save_union: one binary union PNG per image (A/B scorer) instead of ~8k per-head PNGs → no page-cache
    # balloon (the RAM spike seen mid-run). Independent of save_masks (which controls the per-head PNGs).
    save_union_flag = cfg.get("save_union_mask", False)
    last_base_result_path = None

    total_sam_pure_time = 0.0
    total_sam_images = 0

    for folder in image_folders:
        # relpath gives "plot_461" for FIP, "field_A/20250618" for phone
        plot_name = os.path.relpath(os.path.dirname(folder), cfg.dataset.input_dir)
        print(f"\n[SAM Phase] Processing Plot: {plot_name}")

        base_result_path = get_mask_generation_result_path(cfg, plot_name)
        last_base_result_path = base_result_path
        bbox_folder    = os.path.join(base_result_path, "bboxes")   # read bboxes written by YOLO phase
        sam_vis_folder = os.path.join(base_result_path, "sam_vis")
        mask_folder    = os.path.join(base_result_path, "masks")
        reset_folder(sam_vis_folder)
        reset_folder(mask_folder)

        image_files = glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg'))
        if cfg.limit_images > 0:
            image_files = image_files[:cfg.limit_images]

        start_sam_plot = time.perf_counter()
        n_images = len(image_files)
        save_masks_flag = cfg.get("save_masks", True)  # false = skip per-head PNGs (fast A/B, overlay only)
        save_futures = []  # collect save futures so we can harvest t_save at the end
        # BACKPRESSURE cap: how many image-saves may sit in the queue at once. Each queued save still
        # holds that image's full-res masks (~12 MB per head -> a few GB at hundreds of phone heads),
        # so without a cap the backlog piles up across many images and RAM climbs into swap. Block on
        # the oldest save once more than this many are in flight. Tunable via method.sam_save_queue_max.
        save_queue_max = cfg.method.get("sam_save_queue_max", 2)

        # Outer executor has 2 slots: one for the load future, one for the save future.
        # The save task spawns its own inner pool (max_threads) for parallel mask PNG writing.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

            # --- PRE-PROCESSING: kick off load for image 0 immediately ---
            # It runs in the background while the loop sets up, so it's often already
            # done by the time we call .result() below.
            load_future = executor.submit(_load_image_and_bbox, image_files[0], bbox_folder, cfg)

            prev_save_data = None  # holds (masks_np, image, save_name, image_name, ...) from previous GPU step

            for i in range(n_images):
                # --- PRE-PROCESSING: collect load results for the current image ---
                # .result() blocks until the parallel load is done — usually already finished
                # since it ran during the previous GPU call.
                image_name, save_name, image, bbox, crop = load_future.result()

                if os.path.exists(os.path.join(sam_vis_folder, image_name)):
                    continue

                # --- PRE-PROCESSING: submit load for the NEXT image ---
                # This starts immediately and runs in parallel while the GPU works below.
                if i + 1 < n_images:
                    load_future = executor.submit(_load_image_and_bbox, image_files[i + 1], bbox_folder, cfg)

                # --- POST-PROCESSING: submit save for the PREVIOUS image ---
                # Also starts immediately and runs in parallel while the GPU works below.
                # _save_image_results writes all mask PNGs + the overlay visualization.
                if prev_save_data is not None:
                    sf = executor.submit(_save_image_results, *prev_save_data,
                                         cfg.method.max_threads, save_masks_flag, save_union_flag)
                    save_futures.append(sf)
                    prev_save_data = None  # drop reference so RAM can be freed once save is done
                    # backpressure: if the save queue is deeper than the cap, block on the oldest
                    # save(s) until it drains. Each drained future's t_save is accumulated here, so
                    # the end-of-plot harvest below only sums the ones still pending.
                    while len(save_futures) > save_queue_max:
                        total_sam_pure_time += save_futures.pop(0).result()

                # Handle skip cases (missing bbox file or no detections)
                if bbox is None:
                    print(f"    Warning: No boxes found for {image_name}, skipping SAM.")
                    continue
                if len(bbox) == 0:
                    print(f"    No wheat heads detected in {image_name}")
                    continue

                bbox = bbox.to(DEVICE)

                # ROI-crop: feed SAM the cropped plot (higher head resolution) and shift the boxes
                # into crop coords. masks come back at crop size → pasted back to full frame on save.
                if crop is not None:
                    cx0, cy0, cx1, cy1 = crop
                    sam_image = image[cy0:cy1, cx0:cx1]
                    bbox_in = bbox.clone()
                    bbox_in[:, [0, 2]] = (bbox_in[:, [0, 2]] - cx0).clamp(0, cx1 - cx0)
                    bbox_in[:, [1, 3]] = (bbox_in[:, [1, 3]] - cy0).clamp(0, cy1 - cy0)
                else:
                    sam_image = image
                    bbox_in = bbox

                # --- GPU INFERENCE via the chosen SAM backend + granularity ---
                # Main thread blocks here; load(N+1) and save(N-1) run on CPU in parallel.
                # Same sam_image + bbox_in for every backend → only the segmenter differs (fair swap).
                # sam_crop_mode picks full_frame (default) / per_tile / per_head — see _infer_masks_dispatch.
                masks_np, t_embed, t_pred = _infer_masks_dispatch(backend, state, sam_image, bbox_in, cfg)

                if cfg.method.show_time_sam:
                    print_sam_step_report(i, n_images, image_name, len(bbox), t_embed, t_pred)
                if cfg.wandb_enabled:
                    wandb.log({
                        "plot":      plot_name,
                        "t_embed_s": t_embed,
                        "t_pred_s":  t_pred,
                        "n_heads":   len(bbox),
                    })
                total_sam_pure_time += (t_embed + t_pred)
                total_sam_images += 1

                # store results so we can submit the save on the next loop iteration
                prev_save_data = (masks_np, image, save_name, image_name, mask_folder, sam_vis_folder, crop)

                # Cleanup loop to prevent VRAM overflow (sam1's reset_image happens inside _infer_masks)
                torch.cuda.empty_cache()
                gc.collect()

            # --- POST-PROCESSING: save the last image ---
            # No next GPU call to overlap with, but we still submit so it runs in the background
            # while the executor waits for all futures to finish on __exit__.
            if prev_save_data is not None:
                sf = executor.submit(_save_image_results, *prev_save_data,
                                     cfg.method.max_threads, save_masks_flag, save_union_flag)
                save_futures.append(sf)

            # Collect t_save from all completed save futures and add to total
            for sf in save_futures:
                total_sam_pure_time += sf.result()

        # Plot Final Summary for SAM
        sam_total_plot = time.perf_counter() - start_sam_plot
        if cfg.method.show_time_sam:
            print_sam_plot_summary(len(image_files), sam_total_plot)
        print(f"  Finished Plot: {plot_name}")

    # --- per-backend PEAK VRAM + RAM report (each backend runs in its own process → clean numbers) ---
    peak_vram_alloc = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    peak_vram_resv  = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
    peak_ram_rss    = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB->GB on Linux
    avg_s = (total_sam_pure_time / total_sam_images) if total_sam_images else 0.0
    print("\n" + "=" * 52)
    print(f"   SAM BACKEND PEAK RESOURCES — '{backend}'")
    print("=" * 52)
    print(f"{'Peak VRAM (allocated):':<26} {peak_vram_alloc:>7.2f} GB")
    print(f"{'Peak VRAM (reserved):':<26} {peak_vram_resv:>7.2f} GB")
    print(f"{'Peak RAM (RSS):':<26} {peak_ram_rss:>7.2f} GB")
    print(f"{'Avg SAM time / image:':<26} {avg_s:>7.2f} s   ({total_sam_images} imgs)")
    print("=" * 52 + "\n")
    if last_base_result_path:
        try:
            with open(os.path.join(last_base_result_path, "sam_perf.json"), "w") as f:
                json.dump({"backend": backend,
                           "peak_vram_alloc_gb": round(peak_vram_alloc, 3),
                           "peak_vram_reserved_gb": round(peak_vram_resv, 3),
                           "peak_ram_rss_gb": round(peak_ram_rss, 3),
                           "avg_sec_per_image": round(avg_s, 3),
                           "n_images": total_sam_images}, f, indent=2)
        except Exception as e:
            print(f"WARNING: could not write sam_perf.json: {e}")

    # End of SAM MAIN -> Free SAM memory (backend-aware: sam1 holds sam+predictor, sam2/3 hold model)
    state.clear()
    torch.cuda.empty_cache()
    gc.collect()

    if cfg.wandb_enabled:
        wandb.finish()

    return total_sam_pure_time, total_sam_images
