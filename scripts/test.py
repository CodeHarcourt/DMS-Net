import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import sys
import os

# Add the project root to the Python import path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import DualEfficientNet

# Load a trained DMS-Net checkpoint (keep the same architecture used during training).
model = DualEfficientNet(num_classes=8)
model.load_state_dict(torch.load('final_model.pth'))
model.eval()

# --- Image preprocessing pipeline. ---
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension.

# --- Heatmap visualization helper. ---
def visualize_heatmap(image_tensor, heatmap, save_path):
    # Reverse ImageNet-style normalization for display.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image = image_tensor * std + mean

    # Move from tensors to numpy arrays for plotting.
    image   = image.squeeze().permute(1, 2, 0).detach().numpy()
    heatmap = heatmap.squeeze().detach().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot the raw (denormalized) image.
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Plot the image with an overlayed feature heatmap.
    ax2.imshow(image)
    ax2.imshow(heatmap, cmap='viridis', alpha=0.5)
    ax2.set_title('Feature Heatmap')
    ax2.axis('off')

    plt.savefig(save_path)
    plt.close()

# --- Generate a CASFM feature heatmap for a sample image. ---
def generate_heatmap(image_path):
    img_tensor = preprocess_image(image_path)

    # Forward pass: extract intermediate CASFM fusion features.
    with torch.no_grad():
        left_feat = model.feature_extractor(img_tensor)
        processed_feat = model.left_processor(left_feat)
        # Reuse the same features for both branches to visualize the fusion head.
        heatmap = model.dmff_fusion(processed_feat, processed_feat)

    # Collapse channels into a single saliency map and normalize it.
    heatmap = torch.mean(heatmap, dim=1, keepdim=True)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Resize the heatmap to match the preprocessed input resolution.
    heatmap = F.interpolate(heatmap, size=(224, 224), mode='bilinear')

    visualize_heatmap(img_tensor, heatmap, 'heatmap_result.png')

# Example usage (update the path to a real fundus image).
generate_heatmap('/root/autodl-tmp/NEW_dataset_cutmix_light_1/5417_left.jpg')
