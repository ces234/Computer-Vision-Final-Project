import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch
import sys
import os

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the script directory
custom_dataset_path = os.path.join(script_dir, "custom_dataset.py")
images_dir = os.path.join(script_dir, "marine_debris_data/images")
annotations_dir = os.path.join(script_dir, "marine_debris_data/annotations")

# Add the script directory to sys.path
sys.path.append(script_dir)
from custom_dataset import CustomDataset

# Initialize dataset with relative paths
train_dataset = CustomDataset(images_dir=images_dir, annotations_dir=annotations_dir)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Replace the pre-trained predictor with a custom one
num_classes = 12  # Number of classes (including background)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move model to the appropriate device (GPU or CPU)
model.to(device)

print(f"Dataset initialized with {len(train_dataset)} samples.")
print("Creating DataLoader...")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
print("DataLoader created successfully.")

print("Setting up optimizer...")
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
print("Optimizer initialized.")

num_epochs = 10
model.train()
for epoch in range(num_epochs):
    print(f"Starting training for epoch {epoch + 1}...")
    for batch_idx, (images, targets) in enumerate(train_loader):
        print(f"Processing batch {batch_idx + 1}...")

        # Move images and targets to the same device as the model (GPU or CPU)
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        print(f"Batch contains {len(images)} images.")

        # Compute the loss
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        print(f"Total loss for batch {batch_idx + 1}: {losses.item()}")

        # Backpropagate and optimize
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} completed, Loss: {losses.item()}")

print("Training completed.")
