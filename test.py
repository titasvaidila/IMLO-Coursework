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

# DATA AUGMENTATION FOR EVALUATION
test_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(img_size, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class PetWithTrimap(Dataset):
    def __init__(self, root, split, transform):
        self.base = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types=['category', 'segmentation'],
            download=True,
        )

        self.transform = transform

    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        image, (label, trimap) = self.base[idx]

        image_tensor = self.transform(image)

        trimap = TF.resize(
            trimap,
            list(img_size),
            interpolation=v2.InterpolationMode.NEAREST,
        )

        trimap_array = np.array(trimap, dtype=np.uint8) 
        mask_array = (trimap_array != 2).astype(np.float32) # take all pixels that are not "background" (2) as foreground (1), and background as 0

        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

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

class MyCNN(nn.Module):
    def __init__(self, num_classes=37, in_channels=4, dropout=0.4):
        super().__init__()

        def forward(self, x):
            x = self.stem(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.head(x)
            return x

        # 7x7 kernel sees a large patch of the input image, useful for low-level features
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),   # padding = (kernel_size - 1) / 2
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