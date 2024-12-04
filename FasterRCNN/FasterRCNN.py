import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch
import sys
import os
from torchvision.ops import nms



## Get the current script directory (where FasterRCNN.py is located)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (one level above the current script directory)
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

# Define paths relative to the parent directory
custom_dataset_path = os.path.join(parent_dir, "custom_dataset.py")  # Path to custom_dataset.py
images_dir = os.path.join(parent_dir, "marine_debris_data/images")  # Path to images
annotations_dir = os.path.join(parent_dir, "marine_debris_data/annotations")  # Path to annotations

# Add the parent directory to sys.path so Python can locate custom_dataset
sys.path.append(parent_dir)

# Import the CustomDataset class
from custom_dataset import CustomDataset

# Initialize dataset with relative paths
train_dataset = CustomDataset(images_dir=images_dir, annotations_dir=annotations_dir, train_set=True)
test_dataset = CustomDataset(images_dir=images_dir, annotations_dir=annotations_dir, train_set=False)

print("train dataset length: ", len(train_dataset))
print("test dataset length: ", len(test_dataset))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

# Replace the pre-trained predictor with a custom one
num_classes = 12  # Number of classes (including background)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move model to the appropriate device (GPU or CPU)
model.to(device)

# print(f"Dataset initialized with {len(train_dataset)} samples.")
# print("Creating DataLoader...")
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
# print("DataLoader created successfully.")

# print("Setting up optimizer...")
# optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
# print("Optimizer initialized.")

# # Train the model
# num_epochs = 10
# model.train()
# for epoch in range(num_epochs):
#     print(f"Starting training for epoch {epoch + 1}...")
#     for batch_idx, (images, targets) in enumerate(train_loader):
#         print(f"Processing batch {batch_idx + 1}...")

#         # Move images and targets to the same device as the model (GPU or CPU)
#         images = [img.to(device) for img in images]
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         print(f"Batch contains {len(images)} images.")

#         # Compute the loss
#         loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())
#         print(f"Total loss for batch {batch_idx + 1}: {losses.item()}")

#         # Backpropagate and optimize
#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()

#     print(f"Epoch {epoch + 1} completed, Loss: {losses.item()}")

# print("Training completed.")

# # Save the trained model
# model_save_path = os.path.join(script_dir, "faster_rcnn_model.pth")
# torch.save(model.state_dict(), model_save_path)
# print(f"Model saved to {model_save_path}")

test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

print(f"Test dataset initialized with {len(test_dataset)} samples.")

# Load the saved model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)  # Initialize a new model
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# Ensure the model is loaded on CPU if CUDA is not available
model.load_state_dict(torch.load(
    "C:/Users/ces20/OneDrive/Desktop/Computer-Vision-Final-Project/FasterRCNN/faster_rcnn_model.pth", 
    map_location=torch.device('cpu')
))
model.to(device)
print("Model loaded for testing.")

# Set the model to evaluation mode
model.eval()

from torchvision.ops import box_iou, nms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
import torch

def calculate_iou(predictions, ground_truths):
        x1, y1, w1, h1 = predictions
        x2, y2, w2, h2 = ground_truths

        x1_max = x1 + w1
        y1_max = y1 + h1
        x2_max = x2 + w2
        y2_max = y2 + h2

        x_inter_min = max(x1, x2)
        y_inter_min = max(y1, y2)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)

        inter_width = max(0, x_inter_max - x_inter_min)
        inter_height = max(0, y_inter_max - y_inter_min)

        inter_area = inter_width * inter_height

        box1_area = w1 * h1
        box2_area = w2 * h2

        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0

        return iou

def applyNMS(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    
    sorted_indices = np.argsort(scores)[::-1]

    keep = []

    while len(sorted_indices) > 0:
        current = sorted_indices[0]
        keep.append(current)

        remaining_indicies =  sorted_indices[1:]
        filtered_indices = []

        for idx in remaining_indicies:
            iou = calculate_iou(boxes[current], boxes[idx]) 
            if iou < iou_threshold:
                filtered_indices.append(idx)
        
        sorted_indices = np.array(filtered_indices)

    return keep

    

# Initialize tqdm progress bar
print("Starting testing with visualization...")
with torch.no_grad():
    total_iou = 0
    total_predictions = 0
    total_ground_truths = 0
    correct_predictions = 0

    # For mAP calculation
    all_predictions = []
    all_ground_truths = []

    # Use tqdm for progress tracking
    for batch_idx, (images, targets) in enumerate(tqdm(test_loader, desc="Testing Progress")):
        # Move images and targets to the appropriate device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Get predictions from the model
        outputs = model(images)

        # Loop through each image in the batch
        for i, output in enumerate(outputs):
            ground_truth_boxes = targets[i]['boxes'].cpu().numpy()
            ground_truth_labels = targets[i]['labels'].cpu().numpy()

            predicted_boxes = output['boxes'].cpu().numpy()
            predicted_labels = output['labels'].cpu().numpy()
            predicted_scores = output['scores'].cpu().numpy()

            keep_indices = applyNMS(predicted_boxes, predicted_scores, iou_threshold=0.5)
            predicted_boxes = predicted_boxes[keep_indices]
            predicted_scores = predicted_scores[keep_indices]
            predicted_labels = predicted_labels[keep_indices]


            #Apply NMS (Non-Maximum Suppression) to remove redundant predictions
            # keep = nms(torch.tensor(predicted_boxes), torch.tensor(predicted_scores), iou_threshold=0.5)
            # predicted_boxes = predicted_boxes[keep]
            # predicted_labels = predicted_labels[keep]
            # predicted_scores = predicted_scores[keep]


            # Ensure predicted_boxes and ground_truth_boxes are 2D tensors with shape (N, 4)
            # predicted_boxes = torch.tensor(predicted_boxes) if len(predicted_boxes.shape) == 1 else predicted_boxes
            # ground_truth_boxes = torch.tensor(ground_truth_boxes) if len(ground_truth_boxes.shape) == 1 else ground_truth_boxes

            # Collect predictions and ground truths for mAP
            all_predictions.append({
                'boxes': predicted_boxes,
                'labels': predicted_labels,
                'scores': predicted_scores
            })
            all_ground_truths.append({
                'boxes': ground_truth_boxes,
                'labels': ground_truth_labels
            })

            # Visualize every 20 test images
            if batch_idx % 10 == 0 and i == 0:
                img = images[i].cpu()
                img = F.to_pil_image(img)  # Convert to PIL image for plotting

                fig, ax = plt.subplots(1, figsize=(12, 8))
                ax.imshow(img)

                # Plot ground truth boxes in green
                for box in ground_truth_boxes:
                    x_min, y_min, x_max, y_max = box
                    rect = patches.Rectangle(
                        (x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=2, edgecolor='green', facecolor='none'
                    )
                    ax.add_patch(rect)

                # Plot predicted boxes in red
                for idx, box in enumerate(predicted_boxes):
                    x_min, y_min, x_max, y_max = box
                    rect = patches.Rectangle(
                        (x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)

                    # Display score and label
                    label = f"{predicted_labels[idx]}: {predicted_scores[idx]:.2f}"
                    ax.text(
                        x_min, y_min - 10, label,
                        color='red', fontsize=10, backgroundcolor='white'
                    )

                plt.title(f"Image {batch_idx + 1} - Predictions (Red) & Ground Truth (Green)")
                plt.axis('off')
                plt.show()



            # for predicted in predicted_boxes:
            #     for truth in ground_truth_boxes:
            #         print(predicted)
            #         print(truth)
            #         pcalculate_iou(predicted, truth))

            # IoU and classification accuracy computation
            for pred_idx, pred_box in enumerate(predicted_boxes):
                best_iou = 0
                best_truth_idx = -1

                for truth_idx, truth_box in enumerate(ground_truth_boxes):
                    iou = calculate_iou(pred_box, truth_box)
                    if iou > best_iou:  # Find the ground truth box with the highest IoU
                        best_iou = iou
                        best_truth_idx = truth_idx

                if best_iou >= 0.5:  # Valid detection based on IoU threshold
                    total_predictions += 1
                    if predicted_labels[pred_idx] == ground_truth_labels[best_truth_idx]:
                        correct_predictions += 1

            # Count total ground truths
            total_ground_truths += len(ground_truth_boxes)

            # IoU and classification accuracy computation
    #         iou_matrix = box_iou(
    #             torch.tensor(predicted_boxes),
    #             torch.tensor(ground_truth_boxes)
    #         )

    #         for j, pred_box in enumerate(predicted_boxes):
    #             max_iou, max_idx = iou_matrix[j].max(dim=0)
    #             if max_iou > 0.5:  # Consider as valid detection
    #                 total_iou += max_iou.item()
    #                 total_predictions += 1
    #                 if predicted_labels[j] == ground_truth_labels[max_idx]:
    #                     correct_predictions += 1
    #             total_ground_truths += len(ground_truth_boxes)

    # # Calculate average IoU
    # average_iou = total_iou / total_predictions if total_predictions > 0 else 0
    classification_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0


    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    import numpy as np

    def compute_map(predictions, ground_truths, num_classes, iou_threshold=0.5):
        aps = []
        all_precision = []
        all_recall = []
        all_tp = []
        all_fp = []
        all_scores = []

        # Initialize the plot for individual class Precision-Recall curves
        plt.figure(figsize=(10, 6))
        
        for cls in range(num_classes):
            tp, fp, scores = [], [], []
            total_ground_truths = 0

            for preds, gts in zip(predictions, ground_truths):
                preds_cls = [
                    {'box': box, 'score': score}
                    for box, label, score in zip(preds['boxes'], preds['labels'], preds['scores'])
                    if label == cls
                ]
                gts_cls = [
                    {'box': box}
                    for box, label in zip(gts['boxes'], gts['labels'])
                    if label == cls
                ]
                
                total_ground_truths += len(gts_cls)

                preds_cls.sort(key=lambda x: x['score'], reverse=True)

                matched = set()
                for pred in preds_cls:
                    scores.append(pred['score'])
                    ious = [calculate_iou(pred['box'], gt['box']) for gt in gts_cls]
                    best_iou_idx = np.argmax(ious) if ious else -1
                    if best_iou_idx != -1 and ious[best_iou_idx] >= iou_threshold and best_iou_idx not in matched: 
                        tp.append(1)
                        fp.append(0)
                        matched.add(best_iou_idx)
                    else:
                        tp.append(0)
                        fp.append(1)

            if total_ground_truths == 0:
                aps.append(0)
                continue

            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            recall = tp_cumsum / total_ground_truths

            # Store precision and recall for plotting
            all_precision.append(precision)
            all_recall.append(recall)

            # Store all TP, FP for overall curve
            all_tp.extend(tp)
            all_fp.extend(fp)
            all_scores.extend(scores)

            # Calculate average precision for this class
            ap = np.trapz(precision, recall)
            aps.append(ap)

            # Plot Precision-Recall curve for this class
            plt.plot(recall, precision, label=f'Class {cls} (AP = {ap:.2f})')

        # Finalizing the plot for individual class Precision-Recall curves
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for Each Class')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

        # Now create a separate plot for the overall Precision-Recall curve
        plt.figure(figsize=(10, 6))

        tp_cumsum = np.cumsum(all_tp)
        fp_cumsum = np.cumsum(all_fp)
        precision_all = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall_all = tp_cumsum / len(all_scores)  # Assuming len(all_scores) is the total ground truths across all classes

        # Plot overall Precision-Recall curve
        plt.plot(recall_all, precision_all, label='Overall (mAP = {:.2f})'.format(np.mean(aps)), color='black', linestyle='--')

        # Finalizing the plot for the overall Precision-Recall curve
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Overall Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

        # Calculate mAP
        mAP = np.mean(aps)
        return mAP
    # def compute_map(predictions, ground_truths, iou_threshold=0.5):
    #     aps = []
    #     for cls in range(num_classes):  # Iterate over each class
    #         tp, fp, scores = [], [], []

    #         for preds, gts in zip(predictions, ground_truths):
    #             # Filter predictions and ground truths for the current class
    #             preds_cls = [{'bbox': box, 'score': score} for box, label, score in zip(preds['boxes'], preds['labels'], preds['scores']) if label == cls]
    #             gts_cls = [{'bbox': box} for box, label in zip(gts['boxes'], gts['labels']) if label == cls]

    #             # Sort predictions by confidence
    #             preds_cls.sort(key=lambda x: x['score'], reverse=True)

    #             # Match predictions to ground truths
    #             matched = set()
    #             for pred in preds_cls:
    #                 scores.append(pred['score'])
    #                 ious = [box_iou(torch.tensor([pred['bbox']]), torch.tensor([gt['bbox']]))[0, 0] for gt in gts_cls]
    #                 best_iou_idx = max(range(len(ious)), key=lambda x: ious[x], default=-1)
    #                 if best_iou_idx != -1 and ious[best_iou_idx] >= iou_threshold and best_iou_idx not in matched:
    #                     tp.append(1)
    #                     fp.append(0)
    #                     matched.add(best_iou_idx)
    #                 else:
    #                     tp.append(0)
    #                     fp.append(1)

    #         # Compute precision, recall, and AP for the class
    #         tp_cumsum = torch.cumsum(torch.tensor(tp), dim=0)
    #         fp_cumsum = torch.cumsum(torch.tensor(fp), dim=0)
    #         precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    #         recall = tp_cumsum / len(gts_cls)

    #         # Compute AP as AUC of precision-recall curve
    #         ap = torch.trapz(precision, recall).item()
    #         aps.append(ap)

    #     # Mean over all classes
    #     return sum(aps) / len(aps)

    map_score = compute_map(all_predictions, all_ground_truths, num_classes=12)

    #print(f"Average IoU: {average_iou:.2f}")
    
    print(f"Classification Accuracy: {classification_accuracy:.2f}")
    print(f"Mean Average Precision (mAP): {map_score:.2f}")

print("Testing and visualization completed.")
