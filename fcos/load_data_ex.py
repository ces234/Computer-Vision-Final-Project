from custom_dataset import CustomDataset
import torchvision.transforms as transforms
import torch

BATCH_SIZE = 5

CLASSES = ('Background', 'Bottle', 'Can', 'Chain', 'Drink-carton', 'Hook', 'Propeller', 'Shampoo-', 'Standing-', 'Tire', 'Valve', 'Wall')

def collate(x):
    return tuple(zip(*x))

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    dataset = CustomDataset('marine_debris_data/images', 'marine_debris_data/annotations', transform)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate)

    for i, data in enumerate(train_loader):
        print(data)
