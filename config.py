import os
import torch

# =====================================================
# Device
# =====================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# =====================================================
# Base Directory
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================
# Dataset
# =====================================================

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "real_vs_fake",
    "real-vs-fake"
)

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VALID_DIR = os.path.join(DATASET_PATH, "valid")
TEST_DIR = os.path.join(DATASET_PATH, "test")

# =====================================================
# Image
# =====================================================

IMAGE_SIZE = 224

# =====================================================
# Training
# =====================================================

BATCH_SIZE = 32
EPOCHS = 15

LEARNING_RATE = 2e-5

NUM_CLASSES = 2

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 5

# =====================================================
# Model
# =====================================================

MODEL_NAME = "google/vit-base-patch16-224"

# =====================================================
# Save
# =====================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pth"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")

SEED = 42