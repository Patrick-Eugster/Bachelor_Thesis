import os
import torch
import numpy as np
from PIL import Image

########### Begin of Image Helper Functions, will migrate to utils dir later ###########
def normalize_to_0_1(img_tensor):
    # Normalize PIL loaded image tensor from 0-255 to 0-1
    if torch.max(img_tensor) > 1.0:
        return (img_tensor / 255.0).clamp(0.0, 1.0)
    else:
        return img_tensor

def PILtoTorch(pil_image, resolution, normalize=True):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)).float()
    if normalize:
        resized_image = normalize_to_0_1(resized_image)
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1) # (H,W,3) -> (3,H,W)
    elif len(resized_image.shape) == 2:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    else:
        raise ValueError("PIL.Image shape not recognized")

def binarize_mask(mask_tensor):
    assert torch.min(mask_tensor) >= 0.0 and torch.max(mask_tensor) <= 1.0, "Mask tensor should be in the range [0, 1]"
    if mask_tensor.shape[0] == 1:
        mask_tensor = torch.where(mask_tensor > 0, 1.0, 0.0)
    elif mask_tensor.shape[0] == 3:
        mask_tensor = (mask_tensor > 0.0).any(dim=0).unsqueeze(dim=0).float()
        assert mask_tensor.shape[0] == 1
    else:
        raise ValueError("Mask tensor should have 1 or 3 channels")
    # assert mask_tensor has two unique value 0 and 1
    assert torch.all((mask_tensor == 0) | (mask_tensor == 1)), "Mask tensor should have two unique values 0 and 1"
    return mask_tensor

def gray_tensor_to_PIL(tensor : torch.Tensor):
    return Image.fromarray((torch.clamp(tensor.detach().cpu(), 0, 1).numpy().squeeze() * 255.0).astype(np.uint8))

def rgb_tensor_to_PIL(tensor : torch.Tensor):
    return Image.fromarray((np.transpose(torch.clamp(tensor.detach().cpu(), 0, 1).numpy(), (1, 2, 0)) * 255.0).astype(np.uint8))

def get_bbox_from_mask(mask):
    object_pixels = np.argwhere(mask == 1)
    if object_pixels.size == 0:
        return None
    # Get the min and max x and y coordinates
    y_min, x_min = object_pixels.min(axis=0)
    y_max, x_max = object_pixels.max(axis=0)
    # Return the bounding box in xyxy format
    return (x_min, y_min, x_max, y_max)

def is_overlapping(box1, box2):
    if box1 is None or box2 is None:
        return False
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    
    # Check if one box is to the left or right of the other, or if one is above or below the other
    if x_max1 < x_min2 or x_max2 < x_min1:
        return False  # One box is to the left of the other
    if y_max1 < y_min2 or y_max2 < y_min1:
        return False  # One box is above the other
    return True

def calculate_bbox_iou(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Calculate the intersection coordinates
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    # Calculate the area of the intersection
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    intersection_area = inter_width * inter_height

    # Calculate the areas of each box
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def calculate_seg_iou(mask1, mask2):
    # Calculate intersection (logical AND)
    intersection = np.logical_and(mask1, mask2)

    # Calculate union (logical OR)
    union = np.logical_or(mask1, mask2)

    # Compute IoU
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0
    return iou

def get_bbox_from_mask_gpu(mask_bool):
    """GPU version of get_bbox_from_mask — input is a CUDA bool tensor (H, W)."""
    pixels = mask_bool.nonzero(as_tuple=False)  # (N, 2): each row is [row_idx, col_idx]
    if pixels.numel() == 0:
        return None
    y_min, x_min = pixels.min(dim=0).values
    y_max, x_max = pixels.max(dim=0).values
    return (x_min.item(), y_min.item(), x_max.item(), y_max.item())

def calculate_seg_iou_gpu(mask1, mask2):
    """GPU version of calculate_seg_iou — both inputs are CUDA bool tensors (H, W)."""
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return (intersection / union).item() if union > 0 else 0.0

def build_mask_crop(mask_bool):
    """Take a full-frame bool mask and keep ONLY its tight bounding-box crop:
    returns (y0, y1, x0, x1, crop, area), or None for an empty mask.
    A mask is 0 outside its bbox, so this crop carries all its information while
    using ~1000x less memory — lets us cache every mask instead of re-reading the
    PNG in find_match. See docs/segmentation_3d/SEGMENTATION_3D_RUNTIME.md."""
    pixels = mask_bool.nonzero(as_tuple=False)  # (N, 2): [row, col] of every True pixel
    if pixels.numel() == 0:
        return None
    y0, x0 = pixels.min(dim=0).values
    y1, x1 = pixels.max(dim=0).values
    y0, x0, y1, x1 = int(y0), int(x0), int(y1) + 1, int(x1) + 1  # +1 -> exclusive upper bound
    # numpy round-trip with an explicit .copy() GUARANTEES the crop owns its own buffer and can
    # NEVER be a view pinning the full-frame mask (a torch view would report a tiny numel() while
    # secretly retaining the whole 12 MB frame -> the RAM blow-up we saw). torch.from_numpy then
    # wraps that independent buffer, so the full-frame mask is freed as soon as the worker returns.
    crop_np = mask_bool[y0:y1, x0:x1].cpu().numpy().copy()
    crop = torch.from_numpy(crop_np)
    return (y0, y1, x0, x1, crop, int(crop_np.sum()))

def calculate_seg_iou_gpu_crop(pred_seg, pred_area, entry):
    """IoU between the full-frame rendered blob (pred_seg) and a mask stored only as
    its tight-bbox crop (entry from build_mask_crop). pred_area = pred_seg.sum() (passed
    in so it's computed once per view, not per candidate).
    Numerically identical to calculate_seg_iou_gpu(full_mask, pred_seg): the mask is 0
    outside its crop so the intersection is unchanged, and by inclusion-exclusion the
    union equals |mask| + |pred| - |intersection| exactly. Uses the same torch ops as
    calculate_seg_iou_gpu so the returned float matches bit-for-bit."""
    y0, y1, x0, x1, crop, area = entry
    crop = crop.to(pred_seg.device, non_blocking=True)  # cache lives in CPU RAM; move the tiny crop to GPU per use
    intersection = (pred_seg[y0:y1, x0:x1] & crop).sum()  # only where the mask lives
    union = area + pred_area - intersection  # area may be a python int -> broadcasts with the GPU tensors
    return (intersection / union).item() if union > 0 else 0.0

########### End of Image Helper Functions ###########

########### Begin of Visualization Helper Functions ###########

def overlay_img_w_mask(image_pil, mask_pil, color="red"):
    if color == "red":
        overlay = Image.new("RGBA", image_pil.size, (255, 0, 0, 0))    
        overlay = Image.composite(Image.new("RGBA", image_pil.size, (255, 0, 0, 128)), overlay, mask_pil)
    elif color == "blue":
        overlay = Image.new("RGBA", image_pil.size, (0, 0, 255, 0))    
        overlay = Image.composite(Image.new("RGBA", image_pil.size, (0, 0, 255, 128)), overlay, mask_pil)
    image_pil = image_pil.convert("RGBA")
    image_with_overlay = Image.alpha_composite(image_pil, overlay)
    image_with_overlay_rgb = image_with_overlay.convert("RGB")
    return image_with_overlay_rgb

def vis_image_w_overlay(img_tensor, save_dir, save_name, pred_seg, overlap_seg=None, resize_factor=1):
    """
    Args:
        pred_seg: segmentation rendered from 3DGS
        overlap_seg: seg obtained from SAM with largest IOU between pred_seg
    """
    image_pil = rgb_tensor_to_PIL(img_tensor)
    mask_pil = Image.fromarray(pred_seg.astype(np.uint8) * 255)
    image_with_overlay = overlay_img_w_mask(image_pil, mask_pil, color="red")
    if overlap_seg is not None:
        mask_pil = Image.fromarray(overlap_seg.astype(np.uint8) * 255)
        image_with_overlay = overlay_img_w_mask(image_with_overlay, mask_pil, color="blue")
    if resize_factor != 1:
        width, height = image_with_overlay.size                
        new_size = (width // resize_factor, height // resize_factor)
        image_with_overlay = image_with_overlay.resize(new_size)
    image_with_overlay.save(os.path.join(save_dir, f"{save_name}.jpg"))

########### End of Visualization Helper Functions ###########