import random
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF


# HYPERPARAMETERS
epochs = 30
lr = 0.001
weight_decay = 0.001
num_classes = 37
img_size = (192, 192)
batch_size = 48
data_path = './data'
model_path = 'model'

image_net_mean = [0.485, 0.456, 0.406]
image_net_std = [0.229, 0.224, 0.225]

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

torch.set_default_device(device)
print(f"Using device: {device}")

# SET SEED
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(seed)

# Compute a tight square bounding-box around the foreground pixels in the trimap and add a margin
def trimap_bbox(trimap_array, margin_frac):
    H, W = trimap_array.shape
    foreground = trimap_array != 2

    # Rare case
    if not foreground.any():
        side = min(H, W)
        return (H - side) // 2, (W - side) // 2, side, side

    # Find rows and columns containing foreground pixels
    rows = np.where(foreground.any(axis=1))[0]
    cols = np.where(foreground.any(axis=0))[0]
    top_row, bottom_row = int(rows[0]), int(rows[-1])
    left_col, right_col = int(cols[0]), int(cols[-1])

    # Tight bounding-box dimensions
    bbox_h = bottom_row - top_row + 1
    bbox_w = right_col - left_col + 1

    # Make the box square by taking the larger dimension and add a margin around it
    side = max(bbox_h, bbox_w)
    side = int(side * (1.0 + 2.0 * margin_frac))
    side = min(side, min(H, W))

    # Centre the square on the bbox midpoint
    cy = (top_row + bottom_row) / 2.0
    cx = (left_col + right_col) / 2.0
    top = int(round(cy - side / 2.0))
    left = int(round(cx - side / 2.0))

    # Ensure the box is within the image boundaries
    top = max(0, min(top, H - side))
    left = max(0, min(left, W - side))

    return top, left, side, side


# DATA AUGMENTATION
train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(size=img_size, scale=(0.75, 1.0), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=image_net_mean, std=image_net_std),
])

eval_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(img_size, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=image_net_mean, std=image_net_std),
])


# DATASET
class PetWithTrimap(Dataset):
    def __init__(self, root, split, train):
        self.base = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types=['category', 'segmentation'],
            download=True,
        )
        self.train = train

        # Augmentations that need to be applied separately to the image
        self.color_jitter = v2.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05,
        )
        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=image_net_mean, std=image_net_std),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, (label, trimap) = self.base[idx]
        trimap_array = np.array(trimap, dtype=np.uint8)

        # Create a tight square crop around the foreground pixels in the trimap,
        # with a random margin for training and a fixed margin for evaluation
        if self.train:
            margin = random.uniform(0.05, 0.20)
        else:
            margin = 0.15
        top, left, h, w = trimap_bbox(trimap_array, margin)
        image = TF.crop(image, top, left, h, w)
        trimap = TF.crop(trimap, top, left, h, w)

        # Apply augmentations
        if self.train:
            # Random resized crop (within the bbox crop)
            i, j, sh, sw = v2.RandomResizedCrop.get_params(
                image, scale=(0.75, 1.0), ratio=(0.85, 1.18),
            )
            image = TF.resized_crop(
                image, i, j, sh, sw, list(img_size),
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=True,
            )
            trimap = TF.resized_crop(
                trimap, i, j, sh, sw, list(img_size),
                interpolation=v2.InterpolationMode.NEAREST,
            )

            # Random horizontal flip
            if random.random() < 0.5:
                image = TF.hflip(image)
                trimap = TF.hflip(trimap)

            # Random rotation
            if random.random() < 0.5:
                angle = random.uniform(-15.0, 15.0)
                image = TF.rotate(
                    image, angle,
                    interpolation=v2.InterpolationMode.BILINEAR,
                )
                trimap = TF.rotate(
                    trimap, angle,
                    interpolation=v2.InterpolationMode.NEAREST,
                    fill=2,  # background class for any newly-exposed pixels
                )

            # Image color jitter
            image = self.color_jitter(image)
        else:
            # For evaluation, just do a centre crop and resize to the target size
            image = TF.resize(
                image, list(img_size),
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=True,
            )
            trimap = TF.resize(
                trimap, list(img_size),
                interpolation=v2.InterpolationMode.NEAREST,
            )

        image_tensor = self.to_tensor(image)

        # Change the trimap to a binary mask,
        # foreground and unknown pixels (0 and 1) become 1, and background pixels (2) become 0
        trimap_array = np.array(trimap, dtype=np.uint8)
        mask_array = (trimap_array != 2).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        # Concatenate the image tensor and mask tensor along the channel dimension to create a 4-channel input
        x = torch.cat([image_tensor, mask_tensor], dim=0)

        return x, int(label)

train_dataset = PetWithTrimap(root=data_path, split='trainval', train = True)
eval_dataset = PetWithTrimap(root=data_path, split='trainval', train = False)

# MODEL
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = F.relu(out)

        return out


class MyCNN(nn.Module):
    def __init__(self, num_classes=37, in_channels=4, dropout=0.4):
        super().__init__()

        # 7x7 kernel sees a large patch of the input image, useful for low-level features
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1),
        )

        self.stage2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1),
        )

        self.stage3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1),
        )

        self.stage4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=1),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x

    # Initialize weights using Kaiming He initialization for convolutional and
    # linear layers, and set batch norm weights to 1 and biases to 0.
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)


# TRAINING
model = MyCNN(num_classes=num_classes, in_channels=4, dropout=0.4).to(device)

criterion = nn.CrossEntropyLoss()
optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=(device.type == 'cuda'),
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=(device.type == 'cuda'),
)


def train_loop(loader, model, criterion, optimiser):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Clear gradients from the previous step
        optimiser.zero_grad()

        # Get model's predictions for the current batch
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Compute gradients (backpropagation)
        loss.backward()

        # Update weights
        optimiser.step()

        # Track running totals for reporting
        running_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100 * correct / total

    print(f"Training Set:   Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy


def eval_loop(loader, model, criterion):
    # Disable dropout and other training-specific layers for evaluation
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient calculations for evaluation, since we won't be updating weights
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100 * correct / total

    print(f"Evaluation Set: Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy


# MAIN TRAINING LOOP

## MODEL LOOP ##
print("[INFO] Training starting...")

train_losses = []
train_accuracies = []
eval_losses = []
eval_accuracies = []
best_acc = 0.0

for e in range(epochs):
    print(f"Epoch [{e+1}/{epochs}]")

    # One epoch of training on the augmented training data
    train_loss, train_acc = train_loop(train_loader, model, criterion, optimiser)

    # Evaluate on the unaugmented evaluation data
    eval_loss, eval_acc = eval_loop(eval_loader, model, criterion)

    # Record the losses and accuracies
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    eval_losses.append(eval_loss)
    eval_accuracies.append(eval_acc)

    # Save if this is the best model so far
    if eval_acc > best_acc:
        best_acc = eval_acc
        torch.save(model.state_dict(), f"{model_path}.pth")
        print(f"New best model saved (accuracy: {best_acc:.2f}%)")

    print("-" * 30)

print(f"Training complete. Best evaluation accuracy: {best_acc:.2f}%")

