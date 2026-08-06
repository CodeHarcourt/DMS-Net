import torch
from torchvision import transforms
from PIL import Image
import argparse
import numpy as np
import sys
import os

# Add the project root to the Python import path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import DualEfficientNet

# --- 1. CLI argument parsing ---
def get_args():
    parser = argparse.ArgumentParser(description='Fundus image disease prediction')
    parser.add_argument('--left',  type=str, required=True, help='Path to the left-eye fundus image')
    parser.add_argument('--right', type=str, required=True, help='Path to the right-eye fundus image')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                        help='Path to the trained checkpoint (default: checkpoints/best_model.pth)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold (default: 0.5)')
    return parser.parse_args()

# --- 2. Data preprocessing pipeline (should match training). ---
def get_transform():
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

# --- 3. Condition names (ordered to match the training label layout). ---
CLASS_NAMES = [
    'Normal',            # N
    'Diabetes',          # D
    'Glaucoma',          # G
    'Cataract',          # C
    'AMD',               # A
    'Hypertension',      # H
    'Myopia',            # M
    'Other Abnormality', # O
]

# --- 4. Inference function for a single image pair. ---
def predict(model, left_path, right_path, transform, device, threshold):
    # Load images.
    left_img  = Image.open(left_path).convert('RGB')
    right_img = Image.open(right_path).convert('RGB')

    # Apply preprocessing and add a batch dimension.
    if transform is not None:
        left_img  = transform(left_img).unsqueeze(0)
        right_img = transform(right_img).unsqueeze(0)

    # Move tensors to the selected device.
    left_img  = left_img.to(device)
    right_img = right_img.to(device)

    # Forward pass (inference).
    with torch.no_grad():
        outputs = model(left_img, right_img)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    # Build readable prediction output.
    predictions = (probs > threshold).astype(int)
    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probs)):
        if pred == 1:
            results.append(f"{CLASS_NAMES[i]} ({prob * 100:.1f}%)")
    return results

# --- 5. Entrypoint. ---
def main():
    args = get_args()

    # Pick the execution device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model with pretrained weights.
    model = DualEfficientNet(num_classes=8).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Run inference on the requested image pair.
    transform = get_transform()
    results = predict(model, args.left, args.right, transform, device, args.threshold)

    # Print results to the console.
    print("\nPrediction Results:")
    if len(results) == 0:
        print("No significant abnormalities detected.")
    else:
        print("Detected conditions:")
        for r in results:
            print(f"- {r}")

if __name__ == "__main__":
    main()
