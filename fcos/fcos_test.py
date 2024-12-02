import torchvision
# from torchvision.models.detection. import FastRCNNPredictor
from torch.utils.data import DataLoader
# from torchvision.transforms import ToTensor
import torch
import sys
import os

from custom_dataset import CustomDataset

print('Starting')

CLASSES = ('Background', 'Bottle', 'Can', 'Chain', 'Drink-carton', 'Hook', 'Propeller', 'Shampoo-', 'Standing-', 'Tire', 'Valve', 'Wall')
SAVE_PATH = './fcos.pth'

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the script directory
custom_dataset_path = os.path.join(script_dir, "custom_dataset.py")
images_dir = os.path.join(script_dir, "marine_debris_data/images")
annotations_dir = os.path.join(script_dir, "marine_debris_data/annotations")

# Add the script directory to sys.path
sys.path.append(script_dir)

print('Loading dataset')

# Initialize dataset with relative paths
train_dataset = CustomDataset(images_dir=images_dir, annotations_dir=annotations_dir, train_set=False)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device: {device}')

print('Init model')

# Initialize the model
num_classes = len(CLASSES)  # Number of classes (including background)
model = torchvision.models.detection.fcos_resnet50_fpn(num_classes=num_classes)

model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
model.eval()

# Replace the pre-trained predictor with a custom one
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move model to the appropriate device (GPU or CPU)
model.to(device)

print(f"Dataset initialized with {len(train_dataset)} samples.")
print("Creating DataLoader...")
test_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
print("DataLoader created successfully.")

all_predictions = []
all_targets = []

with torch.no_grad():  # No need to compute gradients for inference
    for batch_idx, (images, targets) in enumerate(test_loader):
        print(f"Evaluating batch {batch_idx}/{len(test_loader)}")

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Run model inference
        prediction = model(images)  # Prediction is a list of dicts with 'boxes' and 'labels'

        # Collect results for evaluation
        for p, t in zip(prediction, targets):
            all_predictions.append(p)
            all_targets.append(t)

# You can compute metrics like mAP (mean Average Precision) or IoU here.
# Below is an example of how you can calculate IoU and average it.

def calculate_iou(predictions, targets):
    iou_scores = []
    for p, t in zip(predictions, targets):
        pred_boxes = p['boxes'].cpu().numpy()
        true_boxes = t['boxes'].cpu().numpy()

        # Assuming only one true box and one predicted box per image
        # Calculate IoU for each predicted box with the true box
        for pred_box in pred_boxes:
            for true_box in true_boxes:
                iou = compute_iou(pred_box, true_box)  # Define compute_iou to calculate IoU between two boxes
                iou_scores.append(iou)

    return iou_scores

def compute_iou(box1, box2):
    # Calculate Intersection over Union (IoU) between two bounding boxes
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    # Calculate intersection
    inter_x1 = max(x1, x1_gt)
    inter_y1 = max(y1, y1_gt)
    inter_x2 = min(x2, x2_gt)
    inter_y2 = min(y2, y2_gt)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate union
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union_area = area1 + area2 - inter_area

    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# Calculate IoU for all predictions
iou_scores = calculate_iou(all_predictions, all_targets)
avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
print(f"Average IoU: {avg_iou:.4f}")

# Optionally, you can use a metric like mean Average Precision (mAP) if you're using a library like COCO API
# Code for mAP calculation can be added here if required

print("Testing complete.")
