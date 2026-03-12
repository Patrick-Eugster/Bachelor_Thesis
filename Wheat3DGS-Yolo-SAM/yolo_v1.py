# yolo_module.py
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
#-------- HELPER FUNCTIONS FOR YOLO --------
# =====================================================================

# Deletes all contents of a folder and recreates it,
# since thats cheaper than to go through the folder and delete every single item.
def reset_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path) # Deletes the folder and everything inside
    os.makedirs(folder_path, exist_ok=True)  # Recreates the empty folder


# For Parallelized CPU Image Resizing, we scale the longer side down to target_size px and add padding
def resize_single_image(img_path, target_size):
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
    dw /= 2 # Divide padding into 2 sides
    dh /= 2
    # Calculate exact top/bottom/left/right padding using YOLO's odd-pixel math
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 4. Resize the image using PIL RESIZE_METHOD
    img_resized = img_orig.resize((new_unpad_w, new_unpad_h), resample=RESIZE_METHOD)
    # 5. Create the gray canvas and paste the image over it
    # Note: canvas size will exactly equal (target_size, target_size)
    # we use grey since that will give less sharp edges compared to white or black
    canvas = Image.new('RGB', (target_size, target_size), (114, 114, 114))
    canvas.paste(img_resized, (left, top))
    # 6. Return the pad_info needed for reversing the boxes later
    pad_info = (r, left, top)
    return np.array(canvas), np.array(img_orig), pad_info


# Reverses the letterbox math, saves .pt, and saves the high-res JPG.
def save_single_result(i, result, original_img, pad_info, original_path, bbox_folder, yolo_vis_folder):
    save_name = os.path.splitext(os.path.basename(original_path))[0]
    r, pad_left, pad_top = pad_info # Extract math from the resize step
    # 1. Move to NumPy to escape PyTorch overhead, so its way faster now
    preds = result.xyxy[0].cpu().numpy() # Get YOLO predictions
    good_count, bad_count = 0, 0
    good_boxes_for_sam = []
    # Make a single clean copy of the original image
    annotated_img = np.array(original_img).copy()
    if len(preds) > 0:
        # 2. VECTORIZED MATH. The math on all 600 boxes simultaneously, this is faster than a python for loop
        preds[:, [0, 2]] -= pad_left # x_min, x_max
        preds[:, [1, 3]] -= pad_top # y_min, y_max
        preds[:, :4] /= r # Divide by the scale to stretch back to original high-res pixels
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
            good_boxes_for_sam = good_preds[:, :4].copy() # Save raw coordinates for SAM
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
                    cv2.putText(annotated_img, conf_text, pos, font, font_scale, (255, 255, 255), thickness=BOX_THICKNESS+1, lineType=cv2.LINE_AA)
                    cv2.putText(annotated_img, conf_text, pos, font, font_scale, (0, 0, 255), thickness=BOX_THICKNESS-1, lineType=cv2.LINE_AA)
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
    # 6. Save Tensors for SAM
    if len(good_boxes_for_sam) > 0:
        valid_tensor = torch.tensor(good_boxes_for_sam)
        torch.save(valid_tensor, os.path.join(bbox_folder, f"{save_name}.pt"))
    else:
        torch.save(torch.tensor([]), os.path.join(bbox_folder, f"{save_name}.pt"))
    # 7. Save JPG   
    out_path = os.path.join(yolo_vis_folder, f"{save_name}.jpg")
    Image.fromarray(annotated_img).save(out_path, quality=90)
    
    return good_count, bad_count


def print_performance_report_yolo(num_images, prep_t, gpu_t, disk_t):
    comb_inf = prep_t + gpu_t
    total_t = prep_t + gpu_t + disk_t
    print(f"\n" + "="*45)
    print(f"       PLOT PERFORMANCE REPORT ({num_images} images)")
    print("="*45)
    print(f"{'CPU Parallel Resize Time:':<26} {prep_t:>8.2f}s")
    print(f"{'Pure GPU YOLO Math Time:':<26} {gpu_t:>8.2f}s")
    print(f"{'Total GPU Inference Time:':<26} {comb_inf:>8.2f}s")
    print(f"{'Post-Process & Disk Write Time:':<26} {disk_t:>8.2f}s")
    print("-" * 45)
    print(f"{'TOTAL PLOT TIME:':<26} {total_t:>8.2f}s")
    print("="*45 + "\n")


# Debug, for checking the resized image
def save_debug_image_yolo(resized_imgs, folder):        
    if resized_imgs:
        Image.fromarray(resized_imgs[0]).save(os.path.join(folder, "DEBUG_RESIZED.jpg"))
        print(f"DEBUG: Resized sample saved to {folder}")



# =====================================================================
#  MAIN YOLO INFERENCE (ALL PLOTS)
# =====================================================================
def run_yolo_phase(image_folders):
    print("\n" + "="*50)
    print(" PHASE 1: LOADING YOLO AND PROCESSING ALL PLOTS")
    print("="*50)
    
    if not os.path.exists(WHEAT_YOLO_MODEL):
        print(f"ERROR: Wheat model not found at {WHEAT_YOLO_MODEL}")
        return 0
    # Load YOLO Model ONCE & Load custom model using local repo
    model = torch.hub.load(YOLO_DIR, 'custom', path=WHEAT_YOLO_MODEL, source='local')
    model.conf = CONF_THRESHOLD_GOOD_AND_BAD_BOX # to fix the "aggressive" boxes or adjust as seen fit
    model.iou = IOU_THRESHOLD    
    model.classes = CLASSES_TO_DETECT  

    total_run_boxes = 0

    for folder in image_folders:
        plot_name = folder.split(os.sep)[-2] # Get parent folder name (ex: plot_461)
        print(f"\n[YOLO Phase] Processing Plot: {plot_name}")
        
        # Setup Output Directories
        base_plot_path = os.path.dirname(folder)
        yolo_vis_folder = os.path.join(base_plot_path, "yolo_vis")
        bbox_folder = os.path.join(base_plot_path, "bboxes")
        reset_folder(yolo_vis_folder)
        reset_folder(bbox_folder)
        
        # Get Images
        image_files = glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg'))
        if LIMIT_IMAGES > 0: # Limit Images per plot if wanted
            image_files = image_files[:LIMIT_IMAGES]

        total_prep_time, total_gpu_time, total_disk_time = 0.0, 0.0, 0.0
        total_plot_boxes, total_plot_bad_boxes = 0, 0
        
        # Chunking loop to protect RAM
        for chunk_start in range(0, len(image_files), BATCH_SIZE_RAM_FILES_YOLO):
            chunk_files = image_files[chunk_start : chunk_start + BATCH_SIZE_RAM_FILES_YOLO]
            print(f"  -> Processing chunk {chunk_start} to {chunk_start + len(chunk_files)} of {len(image_files)} images...")
            
            # 1. Parallel Pre-Processing (Resizing & Caching)
            start_prep = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                results_data = list(executor.map(lambda p: resize_single_image(p, TARGET_IMAGE_SIZE), chunk_files))
            
            resized_images = [x[0] for x in results_data]
            original_images = [x[1] for x in results_data]
            pad_infos = [x[2] for x in results_data] # Extract the padding math
            
            prep_time = time.time() - start_prep
            total_prep_time += prep_time
            
            # 2. GPU Inference
            torch.cuda.synchronize()
            start_gpu = time.time()
            det_list = []
            for b_idx in range(0, len(resized_images), BATCH_SIZE_YOLO):
                batch_imgs = resized_images[b_idx : b_idx + BATCH_SIZE_YOLO]
                batch_results = model(batch_imgs, size=TARGET_IMAGE_SIZE)
                det_list.extend(batch_results.tolist())
            torch.cuda.synchronize()
            
            gpu_time = time.time() - start_gpu
            total_gpu_time += gpu_time
            
            # 3. Parallel Post-Processing & Disk Save
            start_disk = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                # Use all MAX_THREADS CPU cores to scale back and save images at once
                # Capture the returned (good_count, bad_count) tuples in a list
                box_counts = list(executor.map(lambda i: save_single_result(
                    i, det_list[i], original_images[i], pad_infos[i], chunk_files[i], bbox_folder, yolo_vis_folder
                ), range(len(det_list))))
                
            disk_time = time.time() - start_disk
            total_disk_time += disk_time
            
            # Unpack the list of tuples and add them to our total trackers
            for good_c, bad_c in box_counts:
                total_plot_boxes += good_c        
                total_plot_bad_boxes += bad_c 
                   
            # Memory Clean-Up for This/Current CHUNK
            if chunk_start == 0 and SHOW_DEBUG_YOLO_RESIZE:
                save_debug_image_yolo(resized_images, yolo_vis_folder)
                
            del results_data, resized_images, original_images, det_list
            gc.collect()

        # 4. Final Performance Report for the entire plot
        print(f"-> YOLO detected a total of {total_plot_boxes} good wheat heads across {len(image_files)} images.")
        print(f"-> YOLO detected a total of {total_plot_bad_boxes} wheat heads below threshold as bad boxes.")
        if SHOW_TIME_YOLO:
            print_performance_report_yolo(len(image_files), total_prep_time, total_gpu_time, total_disk_time)
        
        total_run_boxes += total_plot_boxes
        torch.cuda.empty_cache()
        gc.collect()

    # End of Yolo Main --> Yolo is done: Delete the model to free up VRAM for SAM.
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return total_run_boxes