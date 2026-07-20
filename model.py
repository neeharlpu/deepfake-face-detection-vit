import torch
from transformers import ViTForImageClassification

from config import *


def build_model():

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    # ---------------------------------------------------
    # Compatible with different transformers versions
    # ---------------------------------------------------

    if hasattr(model.vit, "encoder"):
        transformer_layers = model.vit.encoder.layer
    elif hasattr(model.vit, "layers"):
        transformer_layers = model.vit.layers
    else:
        raise AttributeError("Cannot find transformer encoder layers.")

    # Unfreeze last two transformer blocks
    for layer in transformer_layers[-2:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Final LayerNorm
    if hasattr(model.vit, "layernorm"):
        for param in model.vit.layernorm.parameters():
            param.requires_grad = True

    elif hasattr(model.vit, "layer_norm"):
        for param in model.vit.layer_norm.parameters():
            param.requires_grad = True

    return model


def count_parameters(model):

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("=" * 60)
    print(f"Total Parameters     : {total:,}")
    print(f"Trainable Parameters : {trainable:,}")
    print("=" * 60)