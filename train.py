from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch
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
    
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = BreedCNN(num_classes=37).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 30
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
        optimizer.step()

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