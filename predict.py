import torch
from PIL import Image
from torchvision import transforms

from config import *
from model import build_model

# ----------------------------------------------------
# ImageNet Normalization
# ----------------------------------------------------

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# ----------------------------------------------------
# Load Model (only once)
# ----------------------------------------------------

model = build_model()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

CLASS_NAMES = ["Fake", "Real"]


# ----------------------------------------------------
# Prediction Function
# ----------------------------------------------------

def predict(image):

    image = image.convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(tensor).logits

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities, dim=1)

    prediction = prediction.item()

    return {
        "label": CLASS_NAMES[prediction],
        "confidence": confidence.item() * 100,
        "fake_probability": probabilities[0][0].item() * 100,
        "real_probability": probabilities[0][1].item() * 100,
    }