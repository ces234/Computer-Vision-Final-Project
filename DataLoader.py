import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import os
from convertData import convert_to_voc_format

class CustomDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.annotations_files = sorted(os.listdir(annotations_dir))
        self.transform = transform


    def __len__(self):
        return len(self.image_files)
    
    def __getitem__ (self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")

        annotation_path = os.path.join(self.annotations_dir, self.annotation_files[idx])
        objects = convert_to_voc_format(annotation_path)

        boxes = []
        labels = []
        for obj in objects:
            boxes.append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
            labels.append(class_map[obj['name']])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}    
        if self.transform:
            img = self.transform(img)

        return img, target

class_map = {
    "Bottle": 1,
    "Can": 2,
    "Chain": 3,
    "Drink-carton": 4,
    "Hook": 5,
    "Propeller": 6,
    "Shampoo-bottle": 7,
    "Standing-bottle": 8,
    "Tire": 9,
    "Valve": 10,
    "Wall": 11,
}