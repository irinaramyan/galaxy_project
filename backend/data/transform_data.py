from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import torch

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # teaches model flipping images doesnt matter
    transforms.RandomRotation(180), # ^
    transforms.ColorJitter(brightness=0.3, contrast=0.3), # changes lighting of images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # specifically for ImageNet
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class GalaxyDataset(Dataset):
    def __init__(self, folder, labels, transform=None):
        self.folder = folder 
        self.labels = labels
        self.transform = transform
        self.image_paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img = Image.open(f"images_training_rev1/{row["GalaxyID"]}.jpg")
        img = self.transform(img)
        label = row["label"]

        return img, torch.tensor(label)