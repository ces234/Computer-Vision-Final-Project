import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
import numpy as np
import time
import os
import cv2
import matplotlib.pyplot as plt

# Hyperparameters and constants
BATCH_SIZE = 5
CLASSES = ('Background', 'Bottle', 'Can', 'Chain', 'Drink-carton', 'Hook', 'Propeller', 'Shampoo-', 'Standing-', 'Tire', 'Valve', 'Wall')
NUM_CLASSES = len(CLASSES)  # Including background
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
LR = 1e-4  # Learning rate for optimizer
CHECKPOINT_DIR = './checkpoints'
ANNOTATED_EXAMPLES_DIR = './annotated_examples'
SAVE_INTERVAL = 2  # Save checkpoint and annotated examples every 2 epochs
VISUALIZE_INTERVAL = 2  # Visualize results every 2 epochs

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ANNOTATED_EXAMPLES_DIR, exist_ok=True)

# Function to collate the data
def collate_fn(batch):
    return tuple(zip(*batch))

# Dataset transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for each channel
])

# Load dataset
dataset = CustomDataset('marine_debris_data/images', 'marine_debris_data/annotations', transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)

# Load pre-trained model (FCOS with ResNet50 backbone and FPN)
model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)

# Replace the classifier to match the number of classes for FCOS (not Faster R-CNN)
# FCOS has its own head that doesn't use roi_heads
model.head.classification_head = torchvision.models.detection.fcos.FCOSClassificationHead(
    in_channels=model.head.classification_head.conv[0].in_channels,
    num_classes=NUM_CLASSES,
    prior_probability=0.01
)

# Move the model to the selected device
model.to(DEVICE)

# Setup optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=LR)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Function to save checkpoint
def save_checkpoint(model, optimizer, epoch, scheduler, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")

# Function to visualize and save annotated examples
def visualize_and_save_annotations(model, data_loader, epoch, save_dir):
    model.eval()
    images, targets = next(iter(data_loader))
    images = [image.to(DEVICE) for image in images]
    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

    with torch.no_grad():
        predictions = model(images)

    # Visualize first 5 images and predictions
    for idx in range(min(5, len(images))):
        image = images[idx].cpu().numpy().transpose(1, 2, 0)
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

        # Ground truth boxes (for visualization)
        gt_boxes = targets[idx]['boxes'].cpu().numpy()
        gt_labels = targets[idx]['labels'].cpu().numpy()

        # Predicted boxes (for visualization)
        pred_boxes = predictions[idx]['boxes'].cpu().numpy()
        pred_scores = predictions[idx]['scores'].cpu().numpy()

        # Draw ground truth boxes
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f"{CLASSES[label]}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw predicted boxes (with a threshold score)
        for box, score in zip(pred_boxes, pred_scores):
            if score > 0.5:  # Only draw boxes with score > 0.5
                x1, y1, x2, y2 = box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(image, f"{score:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save the annotated image
        cv2.imwrite(os.path.join(save_dir, f"epoch_{epoch}_example_{idx}.png"), image)

    model.train()

# Training loop with checkpointing and annotation generation
def train(model, train_loader, optimizer, lr_scheduler, epochs):
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        num_batches = len(train_loader)

        for i, data in enumerate(train_loader):
            images, targets = data
            images = [image.to(DEVICE) for image in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            loss_dict = model(images, targets)

            # Compute total loss
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            losses.backward()

            # Update parameters
            optimizer.step()

            total_loss += losses.item()

        lr_scheduler.step()

        # Logging per epoch
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Time: {time.time()-start_time:.2f}s")

        # Save checkpoint every SAVE_INTERVAL epochs
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(model, optimizer, epoch + 1, lr_scheduler, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))

        # Generate annotated examples every VISUALIZE_INTERVAL epochs
        if (epoch + 1) % VISUALIZE_INTERVAL == 0:
            visualize_and_save_annotations(model, train_loader, epoch + 1, ANNOTATED_EXAMPLES_DIR)

# Run the training loop
if __name__ == '__main__':
    train(model, train_loader, optimizer, lr_scheduler, EPOCHS)
