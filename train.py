from torchvision import transforms
from torchvision import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Define the transformations for the training dataset
train_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ToTensor(), 
    transforms.Normalize(
        # Mean and std of Imagenet is a common practice
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Define the transformations for the validation dataset (no augmentation, only resizing and normalization)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(
        # Mean and std of Imagenet is a common practice
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Load the training dataset
full_trainval_data = datasets.OxfordIIITPet(
    root='./data', split='trainval', download=True, transform=train_transform
)

# Load the validation dataset (we will split the training data into train and val)
full_val_source = datasets.OxfordIIITPet(
    root='./data', split='trainval', transform=val_transform
)

# Create indices for computationally efficient shuffling and splitting
indices = np.arange(len(full_trainval_data))
targets = full_trainval_data._labels

train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=targets, random_state=42)

train_dataset = Subset(full_trainval_data, train_idx)
val_dataset = Subset(full_val_source, val_idx)

# Create DataLoaders for training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class BreedCNN(nn.Module):
    def __init__(self, num_classes=37):
        super(BreedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Efficient pooling layer to reduce size of data
        self.fc = nn.Linear(32 * 56 * 56, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 32 * 56 * 56)

        x = self.fc(x)

        return x
