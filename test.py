import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

# HYPERPARAMETERS
batch_size = 48
img_size = (192, 192)
num_classes = 37
path = 'model'

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

torch.set_default_device(device)
print(f"Using device: {device}")

# SET SEED
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(seed)
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

# DATA AUGMENTATION FOR EVALUATION
test_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(img_size, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

# TESTING
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")