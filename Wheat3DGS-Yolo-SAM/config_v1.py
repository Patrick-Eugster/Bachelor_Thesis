# config.py
import os
import torch
from PIL import Image

# =====================================================================
# --- CONFIGURATION ---
# =====================================================================
# Script must be run in the same folder as this file!
BASE_DIR = os.getcwd()
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
YOLO_DIR = os.path.join(BASE_DIR, "yolov5")

# Model Paths
WHEAT_YOLO_MODEL = os.path.join(WEIGHTS_DIR, "wheat_head_detection_model.pt")
SAM_CHECKPOINT = os.path.join(WEIGHTS_DIR, "sam_vit_h_4b8939.pth")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- SETTINGS / CONSTANTS ---
CONF_THRESHOLD_GOOD_AND_BAD_BOX = 0.05 # So that we can even see which wheat heads didnt get chosen by a small margin
CONF_THRESHOLD_GOOD_BOX = 0.35 # Minimum confidence to show a box, google colab had 0.05
IOU_THRESHOLD = 0.45 # Maximum allowed overlap between boxes, default 0.45
CLASSES_TO_DETECT = [0] # Only show class 0 (usually 'wheat'), technically here exists only wheat


# Image Resizing Algorithm (Options: Image.LANCZOS, Image.BICUBIC, Image.BILINEAR, Image.NEAREST)
RESIZE_METHOD = Image.LANCZOS
TARGET_IMAGE_SIZE = 1280 # rescaling size for the yolo model. default=640, must be a number x32
BATCH_SIZE_YOLO = 25 # protect GPU VRAM
BATCH_SIZE_RAM_FILES_YOLO = 100 # Protects System RAM: How many images to load at once
BATCH_SIZE_SAM_BOX = 1 # fix number of boxes to process at once (otherwise RAM/VRAM wont be enough)
# weirdly size 1 was best or it barely made a difference in time for me?
MAX_THREADS = 10

SHOW_LABELS = False 
SHOW_REJECTED_RED_BOXES = False
SHOW_GOOD_BOXES = True # default true (blue boxes)
BOX_THICKNESS = 2
LABEL_FONT_SCALE = 1

# --- TEST CONTROLS ---
SHOW_DEBUG_YOLO_RESIZE = False
SHOW_TIME_YOLO = True
SHOW_TIME_SAM = True   
SHOW_TIME_TOTAL = True

ONLY_YOLO = True
LIMIT_PLOTS = 0   # How many plots to process for YOLO and SAM (0 = all)
LIMIT_IMAGES = 0  # How many images per plot for YOLO and SAM (0 = all)
ONLY_LABELED_IMAGES = True  # For Metrics, only process images that have a manual label (ignores LIMIT_IMAGES)

# --- DATASET TOGGLE ---
USE_PHONE_DATA = False  

if USE_PHONE_DATA:
    DATA_DIR = os.path.join(BASE_DIR, "data_phone")
    print("-> Using Dataset: PHONE DATA")
else:
    DATA_DIR = os.path.join(BASE_DIR, "data")
    print("-> Using Dataset: FIP DATA")
    
