from custom_dataset import CustomDataset
import torch.transforms as transforms
import torch

BATCH_SIZE = 5

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CustomDataset('marine_debris_data/images', 'marine_debris_data/annotations', transform)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

for i, data in enumerate(train_loader, 0):
    print(data)
