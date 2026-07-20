import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from tqdm import tqdm

from config import *
from dataset import test_loader
from model import build_model


def evaluate():

    print("=" * 60)
    print("Loading Best Model...")
    print("=" * 60)

    model = build_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Supports both state_dict and checkpoint dictionary
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []
    all_probs = []

    total_loss = 0

    start = time.time()

    with torch.no_grad():

        for images, labels in tqdm(test_loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images).logits

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())

    end = time.time()

    avg_loss = total_loss / len(test_loader)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\n")
    print("=" * 60)
    print("OFFICIAL TEST RESULTS")
    print("=" * 60)

    print(f"Test Loss      : {avg_loss:.4f}")
    print(f"Test Accuracy  : {accuracy*100:.2f}%")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=["Fake","Real"]
    )

    print("\nClassification Report\n")
    print(report)

    with open("classification_report.txt","w") as f:
        f.write(report)

    ####################################################
    # Confusion Matrix
    ####################################################

    cm = confusion_matrix(all_labels, all_preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fake","Real"]
    )

    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()

    ####################################################
    # ROC Curve
    ####################################################

    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png", dpi=300)
    plt.close()

    ####################################################
    # Save Predictions
    ####################################################

    df = pd.DataFrame({
        "True Label": all_labels,
        "Predicted Label": all_preds,
        "Probability": all_probs
    })

    df.to_csv("predictions.csv", index=False)

    ####################################################
    # Save Results
    ####################################################

    with open("evaluation_results.txt","w") as f:

        f.write("OFFICIAL TEST RESULTS\n")
        f.write("="*40 + "\n")
        f.write(f"Test Loss      : {avg_loss:.4f}\n")
        f.write(f"Test Accuracy  : {accuracy*100:.2f}%\n")
        f.write(f"Precision      : {precision:.4f}\n")
        f.write(f"Recall         : {recall:.4f}\n")
        f.write(f"F1 Score       : {f1:.4f}\n")
        f.write(f"AUC            : {roc_auc:.4f}\n")
        f.write(f"Evaluation Time: {end-start:.2f} seconds\n")

    print("\nSaved Files:")
    print("classification_report.txt")
    print("evaluation_results.txt")
    print("predictions.csv")
    print("confusion_matrix.png")
    print("roc_curve.png")

    print(f"\nEvaluation Time : {end-start:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()