"""
yolo_v2.py — pipelined YOLO inference (upgraded from yolo_v1.py)

HOW IT WORKS:
Each RAM chunk (BATCH_SIZE_RAM_FILES_YOLO images) is split into GPU sub-batches
(BATCH_SIZE_YOLO images each). These sub-batches are processed in a 3-stage pipeline:

    Stage 1 — Pre-Processing (CPU):  resize images to TARGET_IMAGE_SIZE with letterbox padding
    Stage 2 — GPU Inference:         run YOLO on the resized batch, get raw bounding boxes
    Stage 3 — Post-Processing (CPU): reverse letterbox math, draw boxes, save .pt tensors + .jpg

In yolo_v1 these 3 stages ran sequentially for the entire chunk — GPU waited for all resizes,
CPU waited for the full GPU pass. In v2 they are pipelined per sub-batch:

    while GPU runs sub-batch N   →  CPU resizes sub-batch N+1  (pre-processing, background thread)
                                 →  CPU saves sub-batch N-1    (post-processing, background thread)

This keeps the GPU busy continuously and hides most of the CPU resize/save time.

CONCURRENCY STRUCTURE:
- One outer ThreadPoolExecutor (max_workers=2): one slot for the resize future, one for the save future.
- Each of those tasks spawns its own inner pool (MAX_THREADS//2 threads) for per-image parallelism.
- MAX_THREADS//2 per inner pool (not MAX_THREADS) because resize and save run simultaneously —
  using full MAX_THREADS for each would over-subscribe the CPU and starve YOLO's NMS step.
- The main thread runs GPU inference and blocks on torch.cuda.synchronize() — this is fine because
  the background threads are independent OS threads and keep running while the main thread waits.
"""

import os
import glob
import time
import concurrent.futures
import gc
import numpy as np
import cv2
import torch
from PIL import Image
import shutil

# Import from config
from config_v1 import *


# =====================================================================
# -------- HELPER FUNCTIONS (identical to yolo_v1.py) --------
# =====================================================================

def reset_folder(folder_path):
    """Delete and recreate a folder. Faster and safer than deleting every item inside."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Deletes the folder and everything inside
    os.makedirs(folder_path, exist_ok=True) # Recreates the empty folder


def resize_single_image(img_path, target_size):
    """For Parallelized CPU Image Resizing. 
    Scale the image so the longer side fits target_size, then pad to a square with grey borders (letterbox)."""
    img_orig = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img_orig.size
    # 1. Scale ratio (new / old) (takes the one which original_size side x or y was bigger)
    r = min(target_size / orig_w, target_size / orig_h)
    # 2. Compute unpadded dimensions
    new_unpad_w = int(round(orig_w * r))
    new_unpad_h = int(round(orig_h * r))
    # 3. Compute padding (compared to letterbox, we use auto=False to ensure all images in a batch
    # are exactly target_size square), so that we can give them as a batch to the model
    dw = target_size - new_unpad_w
    dh = target_size - new_unpad_h
    dw /= 2  # Divide padding into 2 sides
    dh /= 2
    # Calculate exact top/bottom/left/right padding using YOLO's odd-pixel math
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 4. Resize the image using PIL RESIZE_METHOD
    img_resized = img_orig.resize((new_unpad_w, new_unpad_h), resample=RESIZE_METHOD)
    # 5. Create the gray canvas and paste the image over it
    canvas = Image.new('RGB', (target_size, target_size), (114, 114, 114))
    canvas.paste(img_resized, (left, top))
    # 6. Return the pad_info needed for reversing the boxes later
    pad_info = (r, left, top)
    return np.array(canvas), np.array(img_orig), pad_info


def save_single_result(_, result, original_img, pad_info, original_path, bbox_folder, bboxes_with_conf_folder, yolo_vis_folder):
    """Reverse letterbox math to map boxes back to original image coordinates, 
    draw them, and save .pt tensors + .jpg visualization."""
    save_name = os.path.splitext(os.path.basename(original_path))[0]
    r, pad_left, pad_top = pad_info  # Extract math from the resize step
    # 1. Move to NumPy to escape PyTorch overhead, so its way faster now
    preds = result.xyxy[0].cpu().numpy()  # Get YOLO predictions
    good_count, bad_count = 0, 0
    good_boxes_for_sam = []
    # Make a single clean copy of the original image
    annotated_img = np.array(original_img).copy()
    if len(preds) > 0:
        # 2. VECTORIZED MATH. The math on all 600 boxes simultaneously, this is faster than a python for loop
        preds[:, [0, 2]] -= pad_left  # x_min, x_max
        preds[:, [1, 3]] -= pad_top   # y_min, y_max
        preds[:, :4] /= r  # Divide by the scale to stretch back to original high-res pixels
        # Clip boxes to ensure they don't accidentally go outside the image boundary
        orig_h, orig_w = original_img.shape[:2]
        preds[:, [0, 2]] = np.clip(preds[:, [0, 2]], 0, orig_w)
        preds[:, [1, 3]] = np.clip(preds[:, [1, 3]], 0, orig_h)
        # 3. VECTORIZED FILTERING. This mask instantly separates good and bad boxes in C++
        mask = preds[:, 4] >= CONF_THRESHOLD_GOOD_BOX
        good_preds = preds[mask]
        bad_preds = preds[~mask]
        good_count = len(good_preds)
        bad_count = len(bad_preds)
        # Pre-cast coordinates to integers instantly for OpenCV
        if good_count > 0:
            good_coords = good_preds[:, :4].astype(int)
            good_confs = good_preds[:, 4]
            good_boxes_for_sam = good_preds[:, :4].copy()  # Save raw coordinates for SAM
        if bad_count > 0:
            bad_coords = bad_preds[:, :4].astype(int)
            bad_confs = bad_preds[:, 4]
        # 4. Draw Good Boxes (Blue) - directly on image
        if SHOW_GOOD_BOXES and good_count > 0:
            for j in range(good_count):
                x1, y1, x2, y2 = good_coords[j]
                conf = good_confs[j]
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), thickness=BOX_THICKNESS)
                if SHOW_LABELS:
                    conf_text = f"{conf:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = LABEL_FONT_SCALE * 0.7
                    pos = (x1, y1 - 8)
                    cv2.putText(annotated_img, conf_text, pos, font, font_scale, (255, 255, 255), thickness=BOX_THICKNESS + 1, lineType=cv2.LINE_AA)
                    cv2.putText(annotated_img, conf_text, pos, font, font_scale, (0, 0, 255), thickness=BOX_THICKNESS - 1, lineType=cv2.LINE_AA)
        # 5. Draw Bad Boxes (Red) - directly on image
        if SHOW_REJECTED_RED_BOXES and bad_count > 0:
            for j in range(bad_count):
                x1, y1, x2, y2 = bad_coords[j]
                conf = bad_confs[j]
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 30, 30), thickness=BOX_THICKNESS)
                if SHOW_LABELS:
                    conf_text = f"{conf:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = LABEL_FONT_SCALE * 0.7
                    pos = (x1, y1 + 25)
                    cv2.putText(annotated_img, conf_text, pos, font, font_scale, (255, 255, 255), thickness=BOX_THICKNESS + 1, lineType=cv2.LINE_AA)
                    cv2.putText(annotated_img, conf_text, pos, font, font_scale, (255, 30, 30), thickness=max(1, BOX_THICKNESS - 1), lineType=cv2.LINE_AA)
    # 6. Save Tensors for SAM (4 cols: x1,y1,x2,y2 — good boxes only)
    if len(good_boxes_for_sam) > 0:
        valid_tensor = torch.tensor(good_boxes_for_sam)
        torch.save(valid_tensor, os.path.join(bbox_folder, f"{save_name}.pt"))
    else:
        torch.save(torch.tensor([]), os.path.join(bbox_folder, f"{save_name}.pt"))
    # 6b. Save ALL preds with confidence (5 cols: x1,y1,x2,y2,conf) for AP evaluation — only in metrics mode
    if bboxes_with_conf_folder is not None:
        if len(preds) > 0:
            torch.save(torch.tensor(preds[:, :5].copy(), dtype=torch.float32),
                       os.path.join(bboxes_with_conf_folder, f"{save_name}.pt"))
        else:
            torch.save(torch.zeros((0, 5), dtype=torch.float32),
                       os.path.join(bboxes_with_conf_folder, f"{save_name}.pt"))
    # 7. Save JPG
    out_path = os.path.join(yolo_vis_folder, f"{save_name}.jpg")
    Image.fromarray(annotated_img).save(out_path, quality=90)

    return good_count, bad_count


def save_debug_image_yolo(resized_imgs, folder):
    """Save the first resized image to visually verify the letterbox output."""
    if resized_imgs:
        Image.fromarray(resized_imgs[0]).save(os.path.join(folder, "DEBUG_RESIZED.jpg"))
        print(f"DEBUG: Resized sample saved to {folder}")



# =====================================================================
# -------- PIPELINE HELPERS (new in v2) --------
# =====================================================================

def _resize_sub_batch(files, target_size):
    """Resize one GPU sub-batch of images in parallel, returns list of (resized, original, pad_info).
    Uses MAX_THREADS//2 so resize and save can run simultaneously without starving NMS on the main thread."""
    n_workers = min(max(1, MAX_THREADS // 2), len(files))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(lambda p: resize_single_image(p, target_size), files))


def _save_sub_batch(det_list, orig_imgs, pad_infos, files,
                    bbox_folder, bboxes_with_conf_folder, yolo_vis_folder):
    """Save results for one GPU sub-batch in parallel, returns list of (good_count, bad_count).
    Uses MAX_THREADS//2 so resize and save can run simultaneously without starving NMS on the main thread."""
    n_workers = min(max(1, MAX_THREADS // 2), len(files))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(
            lambda i: save_single_result(
                i, det_list[i], orig_imgs[i], pad_infos[i], files[i],
                bbox_folder, bboxes_with_conf_folder, yolo_vis_folder
            ),
            range(len(files))
        ))


def print_performance_report_yolo(num_images, wall_time):
    """Print timing summary. Wall time is used because resize/GPU/save now overlap."""
    avg = wall_time / max(1, num_images)
    print(f"=" * 45)
    print(f"       PLOT PERFORMANCE REPORT ({num_images} images)")
    print(f"{'Total Time (pipelined):':<28} {wall_time:>8.2f}s") # Total time = prep + gpu + disk
    print(f"{'Avg per image:':<28} {avg:>8.2f}s")
    print("="*45 + "\n")



# =====================================================================
#  MAIN YOLO INFERENCE — PIPELINED (ALL PLOTS)
# =====================================================================
def run_yolo_phase(image_folders):
    print("\n" + "=" * 50)
    print(" PHASE 1: LOADING YOLO AND PROCESSING ALL PLOTS")
    print("=" * 50)

    if not os.path.exists(WHEAT_YOLO_MODEL):
        print(f"ERROR: Wheat model not found at {WHEAT_YOLO_MODEL}")
        return 0

    # Load YOLO Model ONCE & Load custom model using local repo
    model = torch.hub.load(YOLO_DIR, 'custom', path=WHEAT_YOLO_MODEL, source='local')
    model.conf = CONF_THRESHOLD_GOOD_AND_BAD_BOX
    model.iou = IOU_THRESHOLD
    model.classes = CLASSES_TO_DETECT

    total_run_boxes = 0

    for folder in image_folders:
        plot_name = folder.split(os.sep)[-2]  # Get parent folder name (ex: plot_461)
        print(f"\n[YOLO Phase] Processing Plot: {plot_name}")

        # Setup Output Directories
        base_plot_path = os.path.dirname(folder)
        yolo_vis_folder = os.path.join(base_plot_path, "yolo_vis")
        bbox_folder = os.path.join(base_plot_path, "bboxes")
        # only create bboxes_with_conf when running in metrics mode — no point saving it for full runs
        if ONLY_LABELED_IMAGES:
            bboxes_with_conf_folder = os.path.join(base_plot_path, "bboxes_with_conf")
            reset_folder(bboxes_with_conf_folder)
        else:
            bboxes_with_conf_folder = None
        reset_folder(yolo_vis_folder)
        reset_folder(bbox_folder)

        # Get Images
        image_files = glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg'))
        if ONLY_LABELED_IMAGES:
            # only keep images that have a manual label for metrics testing and ignores LIMIT_IMAGES
            label_dir = os.path.join(base_plot_path, 'manual_label')
            labeled_stems = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')} if os.path.isdir(label_dir) else set()
            if LIMIT_IMAGES > 0:
                print(f"---ONLY_LABELED_IMAGES=True: ignoring LIMIT_IMAGES={LIMIT_IMAGES}")
            image_files = [f for f in image_files if os.path.splitext(os.path.basename(f))[0] in labeled_stems]
            print(f"---ONLY_LABELED_IMAGES: filtered to {len(image_files)} labeled images")
            for f in image_files:
                print(f"---ONLY_LABELED_IMAGES: using image: {os.path.basename(f)}")
        elif LIMIT_IMAGES > 0:
            image_files = image_files[:LIMIT_IMAGES]

        total_wall_time = 0.0
        total_plot_boxes, total_plot_bad_boxes = 0, 0

        # Chunking loop to protect RAM
        for chunk_start in range(0, len(image_files), BATCH_SIZE_RAM_FILES_YOLO):
            chunk_files = image_files[chunk_start : chunk_start + BATCH_SIZE_RAM_FILES_YOLO]
            print(f"  -> Processing chunk {chunk_start} to {chunk_start + len(chunk_files)} of {len(image_files)} images...")

            # Split chunk into GPU-sized sub-batches
            sub_batches = [chunk_files[i : i + BATCH_SIZE_YOLO]
                           for i in range(0, len(chunk_files), BATCH_SIZE_YOLO)]
            n_sub = len(sub_batches)

            start_chunk = time.perf_counter()

            # Outer executor has 2 slots: one for the resize future, one for the save future.
            # Each of those tasks spawns its own inner pool (MAX_THREADS//2) for per-image parallelism.
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

                # --- PRE-PROCESSING: kick off resize for sub-batch 0 immediately ---
                # It runs in the background while the loop sets up, so it's often already
                # done by the time we call .result() below.
                resize_future = executor.submit(_resize_sub_batch, sub_batches[0], TARGET_IMAGE_SIZE)

                prev_data = None   # holds (det_list, orig_imgs, pad_infos, files) from the previous GPU step
                save_futures = []  # collect save futures so we can harvest box counts at the end

                for b in range(n_sub):
                    # --- PRE-PROCESSING: collect resize results for the current sub-batch ---
                    # .result() blocks until the parallel resize is done — usually already finished
                    # since it ran during the previous GPU call.
                    sub_data = resize_future.result()
                    resized_imgs = [x[0] for x in sub_data]  # letterboxed images → go to GPU
                    orig_imgs    = [x[1] for x in sub_data]  # originals → needed for box reversal later
                    pad_infos    = [x[2] for x in sub_data]  # (scale, pad_left, pad_top) per image

                    if b == 0 and chunk_start == 0 and SHOW_DEBUG_YOLO_RESIZE:
                        save_debug_image_yolo(resized_imgs, yolo_vis_folder)

                    # --- PRE-PROCESSING: submit resize for the NEXT sub-batch ---
                    # This starts immediately and runs in parallel while the GPU works below.
                    if b + 1 < n_sub:
                        resize_future = executor.submit(_resize_sub_batch, sub_batches[b + 1], TARGET_IMAGE_SIZE)

                    # --- POST-PROCESSING: submit save for the PREVIOUS sub-batch ---
                    # Also starts immediately and runs in parallel while the GPU works below.
                    # save_single_result reverses letterbox math, draws boxes, writes .pt + .jpg.
                    if prev_data is not None:
                        prev_det, prev_orig, prev_pad, prev_files = prev_data
                        sf = executor.submit(_save_sub_batch, prev_det, prev_orig, prev_pad, prev_files,
                                             bbox_folder, bboxes_with_conf_folder, yolo_vis_folder)
                        save_futures.append(sf)
                        prev_data = None  # drop reference so RAM can be freed once the save thread is done

                    # --- GPU INFERENCE: run YOLO on the current sub-batch ---
                    # Main thread blocks here. The resize (next) and save (prev) run on CPU in parallel.
                    torch.cuda.synchronize()
                    batch_results = model(resized_imgs, size=TARGET_IMAGE_SIZE)
                    torch.cuda.synchronize()
                    det_list = batch_results.tolist()

                    # store results so we can submit the save on the next loop iteration
                    prev_data = (det_list, orig_imgs, pad_infos, list(sub_batches[b]))

                # --- POST-PROCESSING: save the last sub-batch ---
                # No next GPU call to overlap with, but we still submit so it runs in the background
                # while the executor waits for all futures to finish on __exit__.
                if prev_data is not None:
                    prev_det, prev_orig, prev_pad, prev_files = prev_data
                    sf = executor.submit(_save_sub_batch, prev_det, prev_orig, prev_pad, prev_files,
                                         bbox_folder, bboxes_with_conf_folder, yolo_vis_folder)
                    save_futures.append(sf)

                # Collect box counts from all completed save futures
                for sf in save_futures:
                    for good_c, bad_c in sf.result():
                        total_plot_boxes += good_c
                        total_plot_bad_boxes += bad_c

            total_wall_time += time.perf_counter() - start_chunk
            gc.collect()

        # Final Performance Report for the entire plot
        print(f"-> YOLO detected a total of {total_plot_boxes} good wheat heads across {len(image_files)} images.")
        print(f"-> YOLO detected a total of {total_plot_bad_boxes} wheat heads below threshold as bad boxes.")
        if SHOW_TIME_YOLO:
            print_performance_report_yolo(len(image_files), total_wall_time)

        total_run_boxes += total_plot_boxes
        torch.cuda.empty_cache()
        gc.collect()

    # Yolo is done: Delete the model to free up VRAM for SAM.
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return total_run_boxes
