import os
import torch
import torch.nn as nn
from tqdm import tqdm

from config import *
from model import build_model
from dataset import train_loader, valid_loader


CHECKPOINT_PATH = "/content/drive/MyDrive/cohort/models/checkpoint_latest.pth"


def main():

    print("=" * 60)
    print("Using Device :", DEVICE)
    print("=" * 60)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    model = build_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss().to(DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    ####################################################
    # Resume from checkpoint if available
    ####################################################

    if os.path.exists(CHECKPOINT_PATH):

        print("\nLoading latest checkpoint...\n")

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]
        history = checkpoint["history"]

        print(f"Resuming from Epoch {start_epoch + 1}\n")

    ####################################################
    # Training Loop
    ####################################################

    for epoch in range(start_epoch, EPOCHS):

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        #############################
        # TRAIN
        #############################

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc="Training")

        for images, labels in train_bar:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images).logits

            loss = criterion(outputs, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100*correct/total:.2f}%"
            )

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        #############################
        # VALIDATION
        #############################

        model.eval()

        running_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():

            valid_bar = tqdm(valid_loader, desc="Validation")

            for images, labels in valid_bar:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images).logits

                loss = criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                valid_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{100*correct/total:.2f}%"
                )

        val_loss = running_loss / len(valid_loader)
        val_acc = 100 * correct / total

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        ####################################################
        # Save latest checkpoint
        ####################################################

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history
        }

        torch.save(checkpoint, CHECKPOINT_PATH)

        print("💾 Latest checkpoint saved.")

        ####################################################
        # Epoch Summary
        ####################################################

        print("\n==============================")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.2f}%")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.2f}%")
        print("==============================")

        ####################################################
        # Save Best Model
        ####################################################

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0

            torch.save(model.state_dict(), MODEL_PATH)

            print("✅ Best model saved.")

        else:

            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}/{EARLY_STOPPING_PATIENCE})"
            )

            if patience_counter >= EARLY_STOPPING_PATIENCE:

                print("\n🛑 Early stopping triggered.")
                break

    print("\n======================================")
    print("Training Finished")
    print(f"Best Validation Loss : {best_val_loss:.4f}")
    print("======================================")


if __name__ == "__main__":
    main()