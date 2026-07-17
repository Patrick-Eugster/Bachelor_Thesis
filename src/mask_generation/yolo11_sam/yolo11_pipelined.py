"""
yolo11_pipelined.py — YOLOv11 detector phase (drop-in alternative to yolo_sam_v1's YOLOv5 phase).

Uses the 3DKD-wheat YOLOv11-large spike model (best_yolo11l_40ep.pt, from the olzumst HF space)
via the already-installed `ultralytics` package. It writes the SAME outputs as the YOLOv5 detector
(bboxes/<stem>.pt of good xyxy boxes, yolo_vis/<stem>.jpg overlays, optional bboxes_with_conf/) so
the shared SAM phase + all evaluation code read it unchanged. Pick it with `method=yolo11_sam`.

WHY it's much shorter than yolo_v1_pipelined.py: ultralytics does letterboxing, NMS and
coordinate-reversal internally and returns boxes already in ORIGINAL image pixels, so we don't
reimplement the letterbox pipeline. We still honor the same config knobs: ROI grey-out + box
filter, conf_threshold_nms_floor (keep floor), conf_threshold_good_box (good/bad split),
iou_threshold_nms, classes_to_detect, only_labeled_images, limit_images, and the yolo_vis drawing.

Nothing here imports from yolo_sam_v1 / yolov5 — this method is fully isolated from the YOLOv5 path.
"""

import os
import glob
import time
import gc
import shutil

import numpy as np
import cv2
from PIL import Image
import torch
from ultralytics import YOLO

from wheat_utils.path_utils import get_mask_generation_result_path
from mask_generation.roi_mask import apply_roi, roi_keep_mask, roi_crop_box
from mask_generation.gt_labels import gt_labeled_stems


def reset_folder(folder_path):
    """Delete and recreate a folder — cheaper/safer than removing every item inside."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)


def draw_overlay(img_bgr, good_xyxy, good_conf, bad_xyxy, bad_conf, cfg):
    """Draw good boxes (blue) + rejected boxes (red) with optional conf labels onto a BGR copy."""
    out = img_bgr.copy()
    th = cfg.method.box_thickness
    fs = cfg.method.label_font_scale * 0.7
    if cfg.method.show_detected_boxes:
        for (x1, y1, x2, y2), c in zip(good_xyxy.astype(int), good_conf):
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), th)  # blue in BGR
            if cfg.method.show_labels:
                cv2.putText(out, f"{c:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, fs,
                            (255, 255, 255), th + 1, cv2.LINE_AA)
                cv2.putText(out, f"{c:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, fs,
                            (255, 0, 0), max(1, th - 1), cv2.LINE_AA)
    if cfg.method.show_rejected_boxes:
        for (x1, y1, x2, y2), c in zip(bad_xyxy.astype(int), bad_conf):
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), th)  # red in BGR
            if cfg.method.show_labels:
                cv2.putText(out, f"{c:.2f}", (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, fs,
                            (255, 255, 255), th + 1, cv2.LINE_AA)
                cv2.putText(out, f"{c:.2f}", (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, fs,
                            (0, 0, 255), max(1, th - 1), cv2.LINE_AA)
    return out


def process_one_image(model, img_path, cfg, bbox_folder, bboxes_with_conf_folder, yolo_vis_folder, device):
    """Run YOLOv11 on one image, ROI-filter, split good/bad by conf, save .pt + overlay.
    Returns (good_count, bad_count)."""
    save_name = os.path.splitext(os.path.basename(img_path))[0]
    # Load via PIL then convert RGB->BGR: byte-identical to cv2.imread but SILENT on the
    # Samsung-firmware phone JPEGs (cv2 prints "Invalid SOS parameters for sequential JPEG").
    # ultralytics wants BGR for a numpy input, so we convert. Same trick as the SAM phase.
    try:
        img_bgr = cv2.cvtColor(np.array(Image.open(img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        img_bgr = None
    if img_bgr is None:
        print(f"   UNREADABLE  {save_name}")
        torch.save(torch.tensor([]), os.path.join(bbox_folder, f"{save_name}.pt"))
        return 0, 0
    # ROI grey-out (no-op unless roi.enabled). Fill is channel-symmetric so BGR is fine.
    img_bgr = apply_roi(img_bgr, img_path, cfg)
    orig_h, orig_w = img_bgr.shape[:2]

    # ROI-crop (no-op unless roi.crop_before_inference): crop to the plot region so it fills imgsz
    # (bigger heads + less VRAM). Boxes come back in crop coords → offset to full-image px below.
    crop = roi_crop_box(img_path, cfg, orig_w, orig_h)
    if crop is not None:
        cx0, cy0, cx1, cy1 = crop
        infer_img = img_bgr[cy0:cy1, cx0:cx1]
    else:
        cx0, cy0 = 0, 0
        infer_img = img_bgr

    # ultralytics does letterbox+NMS internally and returns boxes in ORIGINAL pixels.
    # conf = the NMS floor so low-conf boxes survive for metrics; we split good/bad ourselves below.
    r = model.predict(infer_img, imgsz=cfg.method.imgsz, conf=cfg.method.conf_threshold_nms_floor,
                      iou=cfg.method.iou_threshold_nms, max_det=cfg.method.max_det,
                      classes=list(cfg.method.classes_to_detect), device=device, verbose=False)[0]

    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = r.boxes.conf.cpu().numpy().astype(np.float32)
        if crop is not None:  # crop → full-image coords so SAM/eval/overlay all agree
            xyxy[:, [0, 2]] += cx0
            xyxy[:, [1, 3]] += cy0
    else:
        xyxy = np.zeros((0, 4), dtype=np.float32)
        conf = np.zeros((0,), dtype=np.float32)

    # clip to image + ROI box filter (drop boxes outside the plot polygon; no-op unless roi.enabled)
    if len(xyxy):
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, orig_w)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, orig_h)
        keep = roi_keep_mask(xyxy, img_path, cfg, orig_w, orig_h)
        xyxy, conf = xyxy[keep], conf[keep]

    # good/bad split (good = passed to SAM; bad = shown red in yolo_vis)
    good = conf >= cfg.method.conf_threshold_good_box
    good_xyxy, good_conf = xyxy[good], conf[good]
    bad_xyxy, bad_conf = xyxy[~good], conf[~good]

    # save good boxes (4 cols) for SAM
    if len(good_xyxy):
        torch.save(torch.tensor(good_xyxy.copy()), os.path.join(bbox_folder, f"{save_name}.pt"))
    else:
        torch.save(torch.tensor([]), os.path.join(bbox_folder, f"{save_name}.pt"))
    # save ALL preds with conf (5 cols) for AP eval — only in metrics/conf-histogram mode
    if bboxes_with_conf_folder is not None:
        if len(xyxy):
            allp = np.concatenate([xyxy, conf[:, None]], axis=1).astype(np.float32)
            torch.save(torch.tensor(allp), os.path.join(bboxes_with_conf_folder, f"{save_name}.pt"))
        else:
            torch.save(torch.zeros((0, 5), dtype=torch.float32),
                       os.path.join(bboxes_with_conf_folder, f"{save_name}.pt"))
    # overlay
    overlay = draw_overlay(img_bgr, good_xyxy, good_conf, bad_xyxy, bad_conf, cfg)
    cv2.imwrite(os.path.join(yolo_vis_folder, f"{save_name}.jpg"), overlay,
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    return int(good.sum()), int((~good).sum())


def run_yolo_phase_yolo11(image_folders, cfg):
    """YOLOv11 detection over all plots. Same output contract as run_yolo_phase (YOLOv5)."""
    print("\n" + "=" * 50)
    print(" PHASE 1: LOADING YOLOv11 AND PROCESSING ALL PLOTS")
    print("=" * 50)

    weights_dir = os.path.join(os.path.dirname(__file__), "..", "weights")
    model_path = os.path.join(weights_dir, cfg.method.yolo11_model)
    if not os.path.exists(model_path):
        print(f"ERROR: YOLOv11 model not found at {model_path}")
        return 0

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)  # ultralytics loads the yolo11 .pt (no pip install; uses installed ultralytics)
    print(f"Loaded YOLOv11 model: {cfg.method.yolo11_model}  (task={model.task}, device={device})")

    total_run_boxes = 0

    for folder in image_folders:
        plot_name = os.path.relpath(os.path.dirname(folder), cfg.dataset.input_dir)
        print(f"\n[YOLOv11 Phase] Processing Plot: {plot_name}")

        base_plot_path = os.path.dirname(folder)
        base_result_path = get_mask_generation_result_path(cfg, plot_name)
        yolo_vis_folder = os.path.join(base_result_path, "yolo_vis")
        bbox_folder = os.path.join(base_result_path, "bboxes")
        if cfg.only_labeled_images or cfg.get("save_bboxes_conf", False):
            bboxes_with_conf_folder = os.path.join(base_result_path, "bboxes_with_conf")
            reset_folder(bboxes_with_conf_folder)
        else:
            bboxes_with_conf_folder = None
        reset_folder(yolo_vis_folder)
        reset_folder(bbox_folder)

        # gather images (same filtering rules as the YOLOv5 phase)
        image_files = glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg'))
        if cfg.only_labeled_images:
            label_dir = os.path.join(base_plot_path, 'manual_label')
            labeled_stems = gt_labeled_stems(label_dir)   # .txt boxes OR mask-GT (_sets/, _gt_mask.png)
            image_files = [f for f in image_files
                           if os.path.splitext(os.path.basename(f))[0] in labeled_stems]
            print(f"---ONLY_LABELED_IMAGES: filtered to {len(image_files)} labeled images")
        elif cfg.limit_images > 0:
            image_files = image_files[:cfg.limit_images]
        image_files = sorted(image_files)

        total_plot_boxes, total_plot_bad_boxes = 0, 0
        t0 = time.perf_counter()
        for i, img_path in enumerate(image_files):
            g, b = process_one_image(model, img_path, cfg, bbox_folder,
                                     bboxes_with_conf_folder, yolo_vis_folder, device)
            total_plot_boxes += g
            total_plot_bad_boxes += b
            if (i + 1) % 20 == 0:
                print(f"  -> {i + 1}/{len(image_files)} images...")
                torch.cuda.empty_cache()
        wall = time.perf_counter() - t0

        print(f"-> YOLOv11 detected {total_plot_boxes} good wheat heads across {len(image_files)} images.")
        print(f"-> YOLOv11 detected {total_plot_bad_boxes} boxes below threshold (rejected).")
        if cfg.method.show_time_yolo and len(image_files):
            print(f"   ({wall:.1f}s total, {wall / len(image_files):.2f}s/image)")

        total_run_boxes += total_plot_boxes
        torch.cuda.empty_cache()
        gc.collect()

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return total_run_boxes
