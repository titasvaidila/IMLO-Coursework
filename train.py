from torchvision import datasets
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision('medium')

# Define the transformations for the training dataset
train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=30),
    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the transformations for the validation dataset (no augmentation, only resizing and normalization)
val_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

def main():

    class BreedCNN(nn.Module):
        def __init__(self, num_classes=37):
            super(BreedCNN, self).__init__()

            self.block1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.block2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.block3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.block4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.block5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.global_pool = nn.AdaptiveAvgPool2d(1)

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.global_pool(x)
            x = self.classifier(x)
            return x

        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BreedCNN(num_classes=37).to(device)

    sample = torch.randn(2, 3, 224, 224).to(device)
    print(model(sample).shape)

    criterion = nn.CrossEntropyLoss()

    epochs = 30
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.005,
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )

    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)    
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Validation Accuracy: {accuracy:.2f}%")
        print("-" * 30)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_breed_cnn.pth")
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main()