# config.py
import os
import datetime
import torch
from PIL import Image

# =====================================================================
# --- CONFIGURATION ---
# =====================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))          # src/instance_segmentation/yolo_sam_v1/
WEIGHTS_DIR = os.path.join(BASE_DIR, "..", "weights")             # src/instance_segmentation/weights/
YOLO_DIR    = os.path.join(BASE_DIR, "..", "yolov5")              # src/instance_segmentation/yolov5/

# Model Paths
WHEAT_YOLO_MODEL = os.path.join(WEIGHTS_DIR, "wheat_head_detection_model.pt")
SAM_CHECKPOINT = os.path.join(WEIGHTS_DIR, "sam_vit_h_4b8939.pth")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- DETECTION THRESHOLDS ---
CONF_THRESHOLD_NMS_FLOOR = 0.2  # YOLO model.conf — NMS floor, keep low to capture full confidence range for AP curve
CONF_THRESHOLD_DETECTION = 0.35 # post-NMS filter — boxes below this are discarded and not passed downstream
IOU_THRESHOLD_NMS = 0.45        # NMS suppression threshold — boxes overlapping more than this are removed, default 0.45
CLASSES_TO_DETECT = [0]         # YOLO class filter — 0 = wheat head (only class in this model)

# --- IMAGE PROCESSING ---
# Resize algorithm options: Image.LANCZOS, Image.BICUBIC, Image.BILINEAR, Image.NEAREST
RESIZE_METHOD = Image.LANCZOS
TARGET_IMAGE_SIZE = 1280        # YOLO input resolution in pixels, must be multiple of 32 (YOLO default=640)
BATCH_SIZE_YOLO = 25            # images per GPU batch during YOLO inference — lower if GPU VRAM runs out
RAM_CHUNK_SIZE_YOLO = 100       # images loaded into RAM at once — limits memory usage on large plots
BATCH_SIZE_SAM_BOX = 1          # boxes per SAM call — set_image() processes 1 image at a time anyway (size 1 was best)
MAX_THREADS = 10                 # CPU threads for parallel image load/save I/O

# --- VISUALIZATION ---
SHOW_LABELS = True               # show confidence labels on boxes in yolo_vis/ output images
SHOW_REJECTED_BOXES = True       # show rejected boxes (below CONF_THRESHOLD_DETECTION) in red
SHOW_DETECTED_BOXES = True       # show accepted detections in blue
BOX_THICKNESS = 2                # bounding box line thickness in pixels
LABEL_FONT_SCALE = 1             # font size for confidence labels on boxes

# --- DEBUG / TIMING ---
DEBUG_YOLO_RESIZE = False        # save a debug image showing the letterbox resize for the first batch
SHOW_TIME_YOLO = True            # print per-plot YOLO timing summary
SHOW_TIME_SAM = True             # print per-plot SAM timing summary
SHOW_TIME_TOTAL = True           # print total pipeline timing at the end

# --- RUN CONTROLS ---
ONLY_YOLO = False                # stop after YOLO phase, skip SAM entirely
LIMIT_PLOTS = 0                  # process only N plots (0 = all)
LIMIT_IMAGES = 0                 # process only N images per plot (0 = all)
ONLY_LABELED_IMAGES = False      # only run on images that have a manual label in manual_label/ — for metrics (ignores LIMIT_IMAGES)

WANDB_ENABLED = False            # log SAM progress + GPU/RAM stats to wandb dashboard (wandb.ai)

# --- DATASET TOGGLE ---
USE_PHONE_DATA = False

# --- EXPERIMENT NAMING ---
# Controls where results (bboxes, masks, yolo_vis, sam_vis) are saved:
#   results/instance_segmentation/{camera}/{plot}/yolo_sam_v1/{experiment}/
#
# EXPERIMENT_NAME options:
#   "initial"      — default name for a first/scratch run (no date appended, safe to overwrite)
#   "my_run_name"  — any custom string to identify this experiment
#   ""             — auto-generate a pure timestamp: "2025-04-28_1430"
#
# APPEND_DATE options (ignored when EXPERIMENT_NAME is "" or "initial"):
#   False          — use name as-is:         "my_run_name"
#   True           — append date to name:    "my_run_name_2025-04-28"
EXPERIMENT_NAME = "initial"  # "initial" is the default name for a first run
APPEND_DATE     = False      # True = append date to custom name (ignored for "initial" and pure timestamp)

ROOT_DIR = os.path.join(BASE_DIR, "..", "..", "..")  # workspace root

if USE_PHONE_DATA:
    INPUT_DIR  = os.path.join(ROOT_DIR, "input_plots", "phone")
    RESULT_DIR = os.path.join(ROOT_DIR, "results", "instance_segmentation", "phone")
    print("-> Using Dataset: PHONE DATA")
else:
    INPUT_DIR  = os.path.join(ROOT_DIR, "input_plots", "fip")
    RESULT_DIR = os.path.join(ROOT_DIR, "results", "instance_segmentation", "fip")
    print("-> Using Dataset: FIP DATA")


def get_experiment_name():
    """Return final experiment name based on EXPERIMENT_NAME and APPEND_DATE.

    ""        → pure timestamp "2025-04-28_1430"           (APPEND_DATE ignored)
    "initial" → "initial"                                  (APPEND_DATE ignored)
    "my_run"  + APPEND_DATE=False → "my_run"
    "my_run"  + APPEND_DATE=True  → "my_run_2025-04-28"
    """
    if not EXPERIMENT_NAME:
        # pure timestamp — APPEND_DATE ignored to avoid double date
        return datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    if EXPERIMENT_NAME == "initial":
        # fixed scratch name — never append date
        return "initial"
    if APPEND_DATE:
        return f"{EXPERIMENT_NAME}_{datetime.datetime.now().strftime('%Y-%m-%d')}"
    return EXPERIMENT_NAME


def get_result_path(image_folder):
    """Derive output result path for a given image folder.

    input_plots/fip/plot_461/images/ → results/instance_segmentation/fip/plot_461/yolo_sam_v1/{experiment}/
    """
    plot_name = os.path.basename(os.path.dirname(image_folder))
    return os.path.join(RESULT_DIR, plot_name, "yolo_sam_v1", get_experiment_name())
    
