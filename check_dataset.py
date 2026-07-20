from dataset import *

print()

print("=" * 60)

print("Classes")

print(CLASS_NAMES)

print()

print("Train Images :", len(train_dataset))

print("Validation Images :", len(valid_dataset))

print("Test Images :", len(test_dataset))

print()

print("Train Batches :", len(train_loader))

print("Validation Batches :", len(valid_loader))

print("Test Batches :", len(test_loader))

print("=" * 60)