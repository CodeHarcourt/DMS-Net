import torch
from torchvision import transforms
from PIL import Image
import argparse
import numpy as np
import os
import sys
from datetime import datetime

# Add the project root to the Python import path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import DualEfficientNet

# Condition names (ordered to match the training label layout).
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

# --- CLI argument parsing ---
def get_args():
    parser = argparse.ArgumentParser(description='Batch fundus image prediction')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Folder containing left/right fundus image pairs')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                        help='Path to the trained checkpoint (default: checkpoints/best_model.pth)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold (default: 0.5)')
    return parser.parse_args()

# --- Data preprocessing pipeline (should match training). ---
def get_transform():
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

# --- Inference function for a single image pair. ---
def predict(model, left_path, right_path, transform, device, threshold):
    left_img  = Image.open(left_path).convert('RGB')
    right_img = Image.open(right_path).convert('RGB')

    if transform is not None:
        left_img  = transform(left_img).unsqueeze(0)
        right_img = transform(right_img).unsqueeze(0)

    left_img  = left_img.to(device)
    right_img = right_img.to(device)

    with torch.no_grad():
        outputs = model(left_img, right_img)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    predictions = (probs > threshold).astype(int)
    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probs)):
        if pred == 1:
            results.append(f"{CLASS_NAMES[i]} ({prob * 100:.1f}%)")
    return results

# --- Main batch-prediction loop. ---
def batch_predict(model, input_dir, transform, device, threshold):
    result_dir = os.path.join(input_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    # Auto-discover left/right image pairs by the "_left.jpg" / "_right.jpg" convention.
    image_pairs = []
    for filename in os.listdir(input_dir):
        if filename.endswith("_left.jpg"):
            left_path  = os.path.join(input_dir, filename)
            right_path = left_path.replace("_left.jpg", "_right.jpg")
            if os.path.exists(right_path):
                image_pairs.append((left_path, right_path))

    print(f"Found {len(image_pairs)} fundus image pairs to process...")

    for idx, (left_path, right_path) in enumerate(image_pairs, 1):
        print(f"\nProcessing {idx}/{len(image_pairs)}")
        print(f"Analyzing: {os.path.basename(left_path)}")

        try:
            results = predict(model, left_path, right_path, transform, device, threshold)
            base_name = os.path.basename(left_path).replace("_left.jpg", "")
            output_path = os.path.join(result_dir, f"{base_name}_result.txt")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("Fundus Image Analysis Report\n")
                f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Left-eye image:  {os.path.abspath(left_path)}\n")
                f.write(f"Right-eye image: {os.path.abspath(right_path)}\n")
                f.write("\n".join(results) if results else "No significant abnormalities detected.")

            print(f"[OK] Result saved to: {os.path.relpath(output_path)}")

        except Exception as e:
            print(f"[FAIL] Failed to process {filename}: {str(e)}")

    print(f"\nBatch processing complete! Results saved to: {os.path.relpath(result_dir)}")

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model.
    model = DualEfficientNet(num_classes=8).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Run the batch loop.
    transform = get_transform()
    batch_predict(model, args.input_dir, transform, device, args.threshold)
    print("\ndone")

if __name__ == "__main__":
    main()
