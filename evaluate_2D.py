import cv2
import glob
import numpy as np

directory = "plot_463"
type = "fruit"
threshold = 128

gt_image = cv2.imread(glob.glob(f"{directory}/*gt_mask.png")[0], cv2.IMREAD_GRAYSCALE)  # Ground Truth
binary1 = (gt_image >= threshold).astype(np.uint8)
if type == "wheat":
    wheat_image = cv2.imread(glob.glob(f"{directory}/*wheatgs.png")[0], cv2.IMREAD_GRAYSCALE)
    assert gt_image.shape == wheat_image.shape
    binary2 = (wheat_image >= threshold).astype(np.uint8)
elif type == "fruit":
    fruit_image = cv2.imread(glob.glob(f"{directory}/*fruitnerf.png")[0], cv2.IMREAD_GRAYSCALE)
    assert fruit_image.shape == fruit_image.shape
    binary2 = (fruit_image >= threshold).astype(np.uint8)

# Compute metrics
intersection = np.logical_and(binary1, binary2).sum()
union = np.logical_or(binary1, binary2).sum()
true_positives = intersection
false_positives = (binary2.sum() - intersection)  # Pixels in pred but not in GT
false_negatives = (binary1.sum() - intersection)  # Pixels in GT but not in pred

# Calculate IoU, Precision, Recall, and F1-score
iou = intersection / union if union > 0 else 0
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Print results
print(f"IoU: {iou:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")

# Create a 3-channel RGB image for visualization
h, w = gt_image.shape
visualization = np.zeros((h, w, 3), dtype=np.uint8)

# Define colors (BGR format for OpenCV)
gray = [128, 128, 128]    # Ground truth only (gray)
light_red = [128, 213, 255]  # Target only (light red)
red = [0, 0, 255]         # Overlap (red)

# Assign colors based on conditions
visualization[np.logical_and(binary1 == 1, binary2 == 1)] = red       # Overlap (red)
visualization[np.logical_and(binary1 == 1, binary2 == 0)] = gray      # Ground truth only (gray)
visualization[np.logical_and(binary1 == 0, binary2 == 1)] = light_red # Target only (light red)

base_image = cv2.imread(glob.glob(f"{directory}/*img.png")[0])
base_image = base_image.astype(np.float32) / 255.0  # Normalize to range [0,1]
visualization = visualization.astype(np.float32) / 255.0  # Normalize

# Alpha blending (adjust opacity)
alpha = 0.6  # Opacity level (0 = invisible, 1 = fully visible)
overlayed_image = cv2.addWeighted(visualization, alpha, base_image, 1 - alpha, 0)

# Convert back to uint8 for saving
overlayed_image = (overlayed_image * 255).astype(np.uint8)

# Save and display the visualization
cv2.imwrite(f"{directory}/GT-{type}.png", overlayed_image)
# cv2.imshow("Comparison", visualization)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
