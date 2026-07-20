from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import *

# =====================================================
# Normalization (ImageNet)
# =====================================================

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# =====================================================
# Data Augmentation
# =====================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

valid_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# =====================================================
# Dataset Paths
# =====================================================

print("=" * 60)
print("TRAIN :", TRAIN_DIR)
print("VALID :", VALID_DIR)
print("TEST  :", TEST_DIR)
print("=" * 60)

# =====================================================
# Datasets
# =====================================================

train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    root=VALID_DIR,
    transform=valid_transform
)

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=valid_transform
)

CLASS_NAMES = train_dataset.classes

# =====================================================
# DataLoaders
# =====================================================
# IMPORTANT:
# On macOS + Python 3.13 use num_workers=0
# to avoid multiprocessing errors.

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False
)

# =====================================================
# Dataset Information
# =====================================================

print("\nDataset Loaded Successfully")
print("-" * 60)
print("Classes           :", CLASS_NAMES)
print("Training Images   :", len(train_dataset))
print("Validation Images :", len(valid_dataset))
print("Testing Images    :", len(test_dataset))
print("-" * 60)