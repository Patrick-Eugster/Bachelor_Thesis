"""
sahi_yolo_pipelined.py — SAHI (tiled) YOLO detection.

Same job as yolo_v1_pipelined.run_yolo_phase (produce per-image bboxes/*.pt that
SAM consumes), but instead of letterboxing the whole image down to one 1280 pass,
it slices each image into native-resolution tiles, runs the SAME YOLO weights on
each tile, shifts the boxes back to original-image coordinates, and merges the
overlap duplicates. This recovers the small/dense/overlapping heads in phone
images that get lost when a big frame is squashed to 1280.

Why this exists + the math (tile count, edge overlap, the merge): docs/SAHI_EXPLAINED.md.
Design decisions (sahi pkg for the merge only, simple loop first): docs/SAHI_IMPLEMENTATION_PLAN.md.

One image at a time on the GPU (all its tiles go in GPU-sized batches), but the CPU
work is pipelined the same way yolo_v1_pipelined does it: while the GPU runs image N's
tiles, one background thread prefetches+slices image N+1 (load_and_slice) and another
merges+draws+saves image N-1 (_merge_and_save). The only thing kept serial is the GPU
tile inference itself — that's the part we deliberately want one-image-at-a-time so all
of an image's tiles can share a single batch.
"""

import os
import glob
import time
import gc
import concurrent.futures
import numpy as np
import cv2
import torch
from PIL import Image

from sahi.prediction import ObjectPrediction
from sahi.postprocess.combine import NMMPostprocess, NMSPostprocess

from wheat_utils.path_utils import get_mask_generation_result_path
# reuse the folder-reset helper so we don't duplicate it
from mask_generation.yolo_sam_v1.yolo_v1_pipelined import reset_folder


# =====================================================================
# -------- SLICING --------
# =====================================================================

def compute_tile_boxes(img_w, img_h, slice_size, overlap_ratio):
    """Lay a grid of overlapping slice_size tiles over the image; return (x0,y0,x1,y1) per tile.
    The right/bottom edges are clamped to the image so we never read past the border, and an
    image smaller than slice_size yields a single full-image tile (SAHI degrades to normal inference)."""
    step = max(1, int(round(slice_size * (1 - overlap_ratio))))

    def origins(total):
        # tile start positions; always include 0 and a final edge-aligned start if needed
        if total <= slice_size:
            return [0]
        starts = list(range(0, total - slice_size + 1, step))
        if starts[-1] != total - slice_size:
            starts.append(total - slice_size)
        return starts

    tiles = []
    for y0 in origins(img_h):
        for x0 in origins(img_w):
            x1 = min(x0 + slice_size, img_w)
            y1 = min(y0 + slice_size, img_h)
            tiles.append((x0, y0, x1, y1))
    # clamping on small images can create duplicates — drop them
    return list(dict.fromkeys(tiles))


# =====================================================================
# -------- CPU PREP (one image) — runs in a background thread --------
# =====================================================================

def load_and_slice(img_path, cfg):
    """CPU prep for one image (the pipeline's prefetch stage): load the RGB array, lay out the
    tile grid, and cut the native-resolution crops. Returns everything infer_tiles + the save step
    need: (save_name, img_np, h, w, crops, offsets). Runs in a worker thread so it overlaps the
    GPU work on the previous image — touches no GPU/model state, only PIL + numpy slicing."""
    save_name = os.path.splitext(os.path.basename(img_path))[0]
    img_np = np.array(Image.open(img_path).convert('RGB'))
    h, w = img_np.shape[:2]
    tiles   = compute_tile_boxes(w, h, cfg.method.sahi_slice_size, cfg.method.sahi_overlap_ratio)
    crops   = [img_np[y0:y1, x0:x1] for (x0, y0, x1, y1) in tiles]   # views into img_np, no copy
    offsets = [(x0, y0)             for (x0, y0, x1, y1) in tiles]
    return save_name, img_np, h, w, crops, offsets


# =====================================================================
# -------- GPU INFERENCE (one image) — main thread only --------
# =====================================================================

def infer_tiles(model, img_np, crops, offsets, img_w, img_h, cfg):
    """GPU step for one image: run YOLO on the pre-cut tiles in GPU-sized batches (+ optional
    full-image pass), shift every box back to ORIGINAL-image coords, and return them as a
    float32 array [N,5] = x1,y1,x2,y2,conf. Only the main thread calls this (it is the one place
    that touches the model); the CPU slicing was already done by load_and_slice."""
    all_preds = []
    tb = max(1, cfg.method.sahi_tile_batch_size)
    # run tiles in batches so peak VRAM stays bounded (each tile ~ one slice^2 forward)
    for i in range(0, len(crops), tb):
        batch_crops   = crops[i:i + tb]
        batch_offsets = offsets[i:i + tb]
        results = model(batch_crops, size=cfg.method.sahi_slice_size)  # boxes in each crop's coords
        for det, (ox, oy) in zip(results.tolist(), batch_offsets):
            p = det.xyxy[0].cpu().numpy()  # [n,6] x1,y1,x2,y2,conf,cls
            if len(p) == 0:
                continue
            p = p[:, :5].copy()            # keep x1,y1,x2,y2,conf
            p[:, [0, 2]] += ox             # shift tile-local → original coords
            p[:, [1, 3]] += oy
            all_preds.append(p)

    # optional full-image pass: backstops heads bigger than the overlap band (boxes already in original coords)
    if cfg.method.sahi_full_image_pass:
        results = model([img_np], size=cfg.method.target_image_size)
        p = results.xyxy[0].cpu().numpy()
        if len(p) > 0:
            all_preds.append(p[:, :5].copy())

    if all_preds:
        preds = np.concatenate(all_preds, axis=0).astype(np.float32)
    else:
        preds = np.zeros((0, 5), dtype=np.float32)
    # clip to image bounds (tiles never exceed it, but the full-image pass might by a pixel)
    preds[:, [0, 2]] = np.clip(preds[:, [0, 2]], 0, img_w)
    preds[:, [1, 3]] = np.clip(preds[:, [1, 3]], 0, img_h)
    return preds


def merge_preds(preds, img_h, img_w, cfg):
    """Merge the overlap duplicates from all tiles into one box per head using sahi's NMM/NMS.
    Returns merged detections [M,5] = x1,y1,x2,y2,conf."""
    if len(preds) == 0:
        return preds
    obj_preds = [
        ObjectPrediction(bbox=[float(x1), float(y1), float(x2), float(y2)],
                         category_id=0, category_name="wheat",
                         score=float(c), full_shape=[img_h, img_w])
        for (x1, y1, x2, y2, c) in preds
    ]
    metric = cfg.method.sahi_match_metric.upper()   # IOS (robust to fragments) or IOU
    thr    = cfg.method.sahi_match_threshold
    if cfg.method.sahi_merge.upper() == "NMS":
        pp = NMSPostprocess(match_threshold=thr, match_metric=metric, class_agnostic=True)
    else:
        pp = NMMPostprocess(match_threshold=thr, match_metric=metric, class_agnostic=True)
    merged = pp(obj_preds)
    return np.array([[*o.bbox.to_xyxy(), o.score.value] for o in merged], dtype=np.float32)


# =====================================================================
# -------- SAVE (one image) — same on-disk contract as yolo_v1 --------
# =====================================================================

def save_sahi_result(preds, original_img, save_name,
                     bbox_folder, bboxes_with_conf_folder, yolo_vis_folder, cfg):
    """Split merged boxes into good/bad by conf, draw them, and save the SAME files as yolo_v1:
    bboxes/*.pt (good boxes [N,4] for SAM), bboxes_with_conf/*.pt (all [N,5], metrics only), yolo_vis/*.jpg."""
    annotated = original_img.copy()
    good_boxes = []
    good_count = bad_count = 0

    if len(preds) > 0:
        mask = preds[:, 4] >= cfg.method.conf_threshold_good_box
        good = preds[mask]
        bad  = preds[~mask]
        good_count, bad_count = len(good), len(bad)

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = cfg.method.label_font_scale * 0.7
        # good boxes (blue) — (0,0,255) on an RGB array = blue
        if good_count > 0:
            good_boxes = good[:, :4].copy()
            if cfg.method.show_detected_boxes:
                for x1, y1, x2, y2, c in good:
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 0, 255), thickness=cfg.method.box_thickness)
                    if cfg.method.show_labels:
                        # same as yolo_v1: white halo first (thickness+1) then colored text (thickness-1)
                        conf_text = f"{c:.2f}"
                        pos = (int(x1), int(y1) - 8)
                        cv2.putText(annotated, conf_text, pos, font, font_scale, (255, 255, 255),
                                    thickness=cfg.method.box_thickness + 1, lineType=cv2.LINE_AA)
                        cv2.putText(annotated, conf_text, pos, font, font_scale, (0, 0, 255),
                                    thickness=max(1, cfg.method.box_thickness - 1), lineType=cv2.LINE_AA)
        # rejected boxes (red)
        if cfg.method.show_rejected_boxes and bad_count > 0:
            for x1, y1, x2, y2, c in bad:
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                              (255, 30, 30), thickness=cfg.method.box_thickness)
                if cfg.method.show_labels:
                    # red label sits below the box (y1+25) like yolo_v1, white halo + red text
                    conf_text = f"{c:.2f}"
                    pos = (int(x1), int(y1) + 25)
                    cv2.putText(annotated, conf_text, pos, font, font_scale, (255, 255, 255),
                                thickness=cfg.method.box_thickness + 1, lineType=cv2.LINE_AA)
                    cv2.putText(annotated, conf_text, pos, font, font_scale, (255, 30, 30),
                                thickness=max(1, cfg.method.box_thickness - 1), lineType=cv2.LINE_AA)

    # bboxes/*.pt — good boxes only (4 cols) for SAM
    if len(good_boxes) > 0:
        torch.save(torch.tensor(good_boxes), os.path.join(bbox_folder, f"{save_name}.pt"))
    else:
        torch.save(torch.tensor([]), os.path.join(bbox_folder, f"{save_name}.pt"))

    # bboxes_with_conf/*.pt — all preds with conf (5 cols) for AP eval, metrics mode only
    if bboxes_with_conf_folder is not None:
        if len(preds) > 0:
            torch.save(torch.tensor(preds[:, :5], dtype=torch.float32),
                       os.path.join(bboxes_with_conf_folder, f"{save_name}.pt"))
        else:
            torch.save(torch.zeros((0, 5), dtype=torch.float32),
                       os.path.join(bboxes_with_conf_folder, f"{save_name}.pt"))

    Image.fromarray(annotated).save(os.path.join(yolo_vis_folder, f"{save_name}.jpg"), quality=90)
    return good_count, bad_count


def _merge_and_save(preds, img_np, img_h, img_w, save_name,
                    bbox_folder, bboxes_with_conf_folder, yolo_vis_folder, cfg):
    """The pipeline's save stage for one image (runs in a background thread, overlapping the next
    image's GPU work): merge the tile overlap-duplicates, then draw + write bboxes/*.pt + yolo_vis/*.jpg.
    Pure CPU (sahi NMM + cv2 + torch.save of CPU tensors) — never touches the model. Returns (good,bad)."""
    merged = merge_preds(preds, img_h, img_w, cfg)
    return save_sahi_result(merged, img_np, save_name,
                            bbox_folder, bboxes_with_conf_folder, yolo_vis_folder, cfg)


# =====================================================================
#  MAIN SAHI DETECTION — ALL PLOTS  (same signature as run_yolo_phase)
# =====================================================================

def run_yolo_phase_sahi(image_folders, cfg):
    """Run SAHI tiled YOLO over all plots; returns the total number of good boxes detected.
    Drop-in replacement for run_yolo_phase — produces the identical bboxes/ output contract."""
    print("\n" + "=" * 50)
    print(" PHASE 1 (SAHI): LOADING YOLO AND PROCESSING ALL PLOTS")
    print("=" * 50)
    print(f"  slice={cfg.method.sahi_slice_size}px  overlap={cfg.method.sahi_overlap_ratio} "
          f"merge={cfg.method.sahi_merge}/{cfg.method.sahi_match_metric}@{cfg.method.sahi_match_threshold} "
          f"full_image_pass={cfg.method.sahi_full_image_pass}  tile_batch={cfg.method.sahi_tile_batch_size}")

    weights_dir = os.path.join(os.path.dirname(__file__), "..", "weights")
    yolo_dir    = os.path.join(os.path.dirname(__file__), "..", "yolov5")
    wheat_model = os.path.join(weights_dir, cfg.method.wheat_yolo_model)

    if not os.path.exists(wheat_model):
        print(f"ERROR: Wheat model not found at {wheat_model}")
        return 0

    # Load YOLO once (same as yolo_v1). conf=nms_floor so tiles keep the full range before merge.
    model = torch.hub.load(yolo_dir, 'custom', path=wheat_model, source='local')
    model.conf = cfg.method.conf_threshold_nms_floor
    model.iou  = cfg.method.iou_threshold_nms
    model.classes = list(cfg.method.classes_to_detect)

    total_run_boxes = 0

    for folder in image_folders:
        plot_name = os.path.relpath(os.path.dirname(folder), cfg.dataset.input_dir)
        print(f"\n[SAHI Phase] Processing Plot: {plot_name}")

        base_plot_path   = os.path.dirname(folder)
        base_result_path = get_mask_generation_result_path(cfg, plot_name)
        yolo_vis_folder  = os.path.join(base_result_path, "yolo_vis")
        bbox_folder      = os.path.join(base_result_path, "bboxes")
        if cfg.only_labeled_images:
            bboxes_with_conf_folder = os.path.join(base_result_path, "bboxes_with_conf")
            reset_folder(bboxes_with_conf_folder)
        else:
            bboxes_with_conf_folder = None
        reset_folder(yolo_vis_folder)
        reset_folder(bbox_folder)

        # Gather images (same filtering as yolo_v1: labeled-only for metrics, else limit_images)
        image_files = glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg'))
        if cfg.only_labeled_images:
            label_dir = os.path.join(base_plot_path, 'manual_label')
            labeled_stems = {os.path.splitext(f)[0] for f in os.listdir(label_dir)
                             if f.endswith('.txt')} if os.path.isdir(label_dir) else set()
            image_files = [f for f in image_files
                           if os.path.splitext(os.path.basename(f))[0] in labeled_stems]
            print(f"---ONLY_LABELED_IMAGES: filtered to {len(image_files)} labeled images")
        elif cfg.limit_images > 0:
            image_files = image_files[:cfg.limit_images]

        start = time.perf_counter()
        plot_good = plot_bad = 0
        n_images = len(image_files)
        save_futures = []  # collect save futures so we can harvest box counts at the end

        # Pipeline (overlap CPU with GPU): while the GPU runs image N's tiles, two background
        # threads run — one prefetches+slices image N+1 (load_and_slice), one merges+draws+saves
        # image N-1 (_merge_and_save). Same 2-slot structure as yolo_v1's run_yolo_phase, only the
        # unit here is a whole image. GPU tile inference stays on the main thread (one image at a time
        # so all its tiles share a batch). torch.cuda.synchronize() blocking the main thread is fine —
        # the worker threads are independent OS threads and keep running while it waits.
        if n_images > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # kick off the load+slice of image 0 (runs in the background while we set up)
                load_future = executor.submit(load_and_slice, image_files[0], cfg)
                prev_save = None  # holds (preds, img_np, h, w, save_name) waiting to be merged+saved

                for idx in range(n_images):
                    # collect the prepared crops for the current image — usually already done,
                    # since load_and_slice ran during the previous image's GPU call
                    save_name, img_np, h, w, crops, offsets = load_future.result()
                    if idx == 0:
                        print(f"  tiling {w}x{h} → {len(crops)} tiles of {cfg.method.sahi_slice_size}px "
                              f"(overlap {cfg.method.sahi_overlap_ratio})")

                    # prefetch+slice the NEXT image in the background
                    if idx + 1 < n_images:
                        load_future = executor.submit(load_and_slice, image_files[idx + 1], cfg)

                    # submit merge+save of the PREVIOUS image in the background
                    if prev_save is not None:
                        sf = executor.submit(_merge_and_save, *prev_save,
                                             bbox_folder, bboxes_with_conf_folder, yolo_vis_folder, cfg)
                        save_futures.append(sf)
                        prev_save = None

                    # GPU: run YOLO on this image's tiles (main thread blocks; load+save run in parallel)
                    preds = infer_tiles(model, img_np, crops, offsets, w, h, cfg)
                    prev_save = (preds, img_np, h, w, save_name)

                    if cfg.method.show_time_yolo and (idx + 1) % 10 == 0:
                        print(f"  ...{idx + 1}/{n_images} images")

                # merge+save the last image (no GPU step left to overlap; executor waits on __exit__)
                if prev_save is not None:
                    sf = executor.submit(_merge_and_save, *prev_save,
                                         bbox_folder, bboxes_with_conf_folder, yolo_vis_folder, cfg)
                    save_futures.append(sf)

                # harvest good/bad counts from all save futures
                for sf in save_futures:
                    g, b = sf.result()
                    plot_good += g
                    plot_bad  += b

        wall = time.perf_counter() - start
        print(f"-> SAHI detected {plot_good} good wheat heads ({plot_bad} below threshold) "
              f"across {len(image_files)} images in {wall:.1f}s "
              f"({wall / max(1, len(image_files)):.2f}s/img)")
        total_run_boxes += plot_good
        torch.cuda.empty_cache()
        gc.collect()

    # free VRAM for SAM
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return total_run_boxes
