import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import sys

# Add the project root to the Python import path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import FundusDataset
from src.models import DualEfficientNet
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score, f1_score, roc_auc_score, precision_score
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# Define data transforms and preprocessing pipeline.
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Build the training and validation datasets.
train_dataset = FundusDataset(
    csv_file="train_set_light_cutmix_v1.csv",
    root_dir="NEW_dataset_cutmix_light_1",
    transform=transform,
)
test_dataset = FundusDataset(
    csv_file="val_set_light_cutmix_v1.csv",
    root_dir="NEW_dataset_cutmix_light_1",
    transform=transform,
)

# Build the training and validation loaders.
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Instantiate model, loss, and optimizer.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualEfficientNet(num_classes=8).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Helper that appends a single line to a text file.
def save_output_to_file(filename, content):
    with open(filename, 'a') as f:
        f.write(content + '\n')

# Training loop with evaluation and checkpointing.
def train_and_evaluate_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=10):
    accumulation_steps = 4   # Gradient accumulation steps.
    scaler = GradScaler()
    best_recall = 0          # Track the best validation recall.

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        running_loss = 0.0    # Accumulator for the raw loss.
        total_samples = 0     # Counter for processed samples.

        for i, (left_imgs, right_imgs, labels) in enumerate(train_dataloader):
            # Move mini-batch to the selected device.
            left_imgs = left_imgs.to(device)
            right_imgs = right_imgs.to(device)
            labels = labels.to(device)

            # Forward pass with mixed precision.
            with autocast():
                outputs = model(left_imgs, right_imgs)
                loss = criterion(outputs, labels) / accumulation_steps

            # Accumulate the raw loss (undo the division by accumulation_steps).
            running_loss += loss.item() * accumulation_steps * left_imgs.size(0)
            total_samples += left_imgs.size(0)

            # Backward pass with gradient scaling.
            scaler.scale(loss).backward()

            # Optimizer step once every accumulation_steps.
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # Average loss across all samples seen in the epoch.
        epoch_loss = running_loss / total_samples
        output = f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}"
        print(output)
        save_output_to_file('output.txt', output)

        # --- Validation phase ---
        model.eval()  # Switch to evaluation mode (disable dropout, etc.).
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for left_imgs, right_imgs, labels in test_dataloader:
                left_imgs = left_imgs.to(device)
                right_imgs = right_imgs.to(device)
                labels = labels.to(device)
                outputs = model(left_imgs, right_imgs)
                all_labels.append(labels.cpu().numpy())
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

        all_labels = np.concatenate(all_labels, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_preds = (all_outputs > 0.5).astype(int)

        # Compute evaluation metrics.
        num_classes = all_labels.shape[1]

        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average='macro')

        # Cohen's kappa (per-class averaged).
        kappa_scores = []
        for i in range(num_classes):
            unique_labels = np.unique(all_labels[:, i])
            if len(unique_labels) >= 2:
                kappa = cohen_kappa_score(all_labels[:, i], all_preds[:, i])
            else:
                kappa = 0.0
            kappa_scores.append(kappa)
        kappa_avg = np.nanmean(kappa_scores)

        # F1 score (macro-averaged).
        f1_macro = f1_score(all_labels, all_preds, average='macro')

        # ROC-AUC (per-class averaged).
        auc_scores = []
        for i in range(num_classes):
            unique_labels = np.unique(all_labels[:, i])
            if len(unique_labels) >= 2:
                auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
            else:
                auc = 0.5
            auc_scores.append(auc)
        auc_avg = np.nanmean(auc_scores)

        # Macro-averaged precision.
        precision = precision_score(all_labels, all_preds, average='macro')

        # Print and persist the validation metrics.
        output = (
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Acc: {accuracy:.4f}, "
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, "
            f"Kappa: {kappa_avg:.4f}, "
            f"F1: {f1_macro:.4f}, "
            f"AUC: {auc_avg:.4f}"
        )
        print(output)
        save_output_to_file('output.txt', output)

        # --- Checkpointing ---
        # Save the best model (tracked by validation recall).
        if recall > best_recall and epoch > 10:
            best_recall = recall
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'recall': recall,
            }, 'best_model.pth')
            output = f"New best model saved at epoch {epoch+1} with recall {recall:.4f}"
            print(output)
            save_output_to_file('output.txt', output)

        # Periodic checkpoint every 25 epochs.
        if (epoch + 1) % 25 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f'checkpoint_epoch{epoch+1}.pth')
            output = f"Checkpoint saved at epoch {epoch+1}"
            print(output)
            save_output_to_file('output.txt', output)

        # Final snapshot at the end of training.
        if epoch == num_epochs - 1:
            torch.save(model.state_dict(), 'final_model.pth')
            output = "Final model saved"
            print(output)
            save_output_to_file('output.txt', output)

        # Switch back to training mode.
        model.train()

        torch.cuda.empty_cache()  # Release unused CUDA memory after each epoch.

# Start training and evaluation.
train_and_evaluate_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=50)
