from torchvision import datasets    # for loading Oxford-IIIT Pet datasets
from torchvision.transforms import v2   # for data augmentation and normalization
from torchvision.transforms.v2 import functional as TF
from torch.utils.data import Dataset, DataLoader # for creating custom dataset and dataloader
from PIL import Image
import numpy as np
import torch

epoch = 30
lr = 0.001
weight_decay = 0.001
num_classes = 37
img_size = (192, 192)  # resize all images to 192x192
batch_size = 48
data_path = './data'
model_path = 'model'

image_net_mean = [0.485, 0.456, 0.406]
image_net_std = [0.229, 0.224, 0.225]

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

torch.set_default_device(device)
print(f"Using device: {device}")

# DATA AUGMENTATION
train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(size=img_size, scale=(0.75, 1.0), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(img_size, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# DATASET
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

train_dataset = PetWithTrimap(root='./data', split='trainval', transform=train_transform)
test_dataset = PetWithTrimap(root='./data', split='trainval', transform=eval_transform)

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

class MyCNN(nn.Module):
    def __init__(self, num_classes=37, in_channels=4, dropout=0.4):
        super().__init__()

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