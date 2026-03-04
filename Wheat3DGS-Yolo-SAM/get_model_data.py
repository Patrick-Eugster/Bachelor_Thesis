import torch
import sys

# Tell Python where to find the YOLOv5 class definitions
sys.path.append('./yolov5')

model_path = "weights/wheat_head_detection_model.pt"
model = torch.load(model_path, map_location='cpu', weights_only=False)

if type(model) is dict and 'model' in model:
    if hasattr(model['model'], 'args'):
        print("\n--- Model Training Arguments ---")
        print(model['model'].args)
    else:
        print("\nModel loaded, but training arguments not found.")