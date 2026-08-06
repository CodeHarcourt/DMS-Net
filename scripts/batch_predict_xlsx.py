import torch
from torchvision import transforms
from PIL import Image
import argparse
import numpy as np
import os
import sys
from datetime import datetime
import pandas as pd
import re

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

# Short codes used as CSV/Excel column headers.
CLASS_CODES = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

# --- CLI argument parsing ---
def get_args():
    parser = argparse.ArgumentParser(description='Excel-backed batch fundus prediction')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Folder containing left/right fundus image pairs')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                        help='Path to the trained checkpoint (default: checkpoints/best_model.pth)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--excel', type=str, default=None,
                        help='Path to the Excel spreadsheet used to drive the predictions')
    parser.add_argument('--output_excel', type=str, default=None,
                        help='Output path for the populated Excel results')
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
# Returns:
#   text_results       : human-readable condition strings
#   binary_predictions : per-class 0/1 thresholded labels
#   probabilities      : per-class raw probability scores
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
    text_results = []
    for i, (pred, prob) in enumerate(zip(predictions, probs)):
        if pred == 1:
            text_results.append(f"{CLASS_NAMES[i]} ({prob * 100:.1f}%)")

    return text_results, predictions, probs

# --- Main batch-prediction loop (with optional Excel round-trip). ---
def batch_predict(model, input_dir, transform, device, threshold,
                  excel_path=None, output_excel=None):
    result_dir = os.path.join(input_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    # Load the driver spreadsheet if provided.
    df = None
    if excel_path and os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        print(f"Loaded Excel file: {excel_path}")
        print(f"Spreadsheet contains {len(df)} rows")
        print(f"Spreadsheet columns: {df.columns.tolist()}")

    image_pairs = []
    patient_ids = []

    # Excel-driven mode: pull patient IDs and filenames from the spreadsheet.
    if df is not None:
        for _, row in df.iterrows():
            patient_id   = row['ID']
            left_fundus  = row['Left-Fundus']
            right_fundus = row['Right-Fundus']

            if isinstance(left_fundus, str) and isinstance(right_fundus, str):
                left_path  = os.path.join(input_dir, left_fundus)
                right_path = os.path.join(input_dir, right_fundus)

                if os.path.exists(left_path) and os.path.exists(right_path):
                    image_pairs.append((left_path, right_path))
                    patient_ids.append(patient_id)
    else:
        # Directory-scan mode: auto-discover pairs by "*_left.*" / "*_right.*".
        for filename in os.listdir(input_dir):
            lowered = filename.lower()
            if "_left" in lowered and filename.lower().endswith((".jpg", ".jpeg", ".png")):
                left_path = os.path.join(input_dir, filename)
                right_filename = filename.replace("_left", "_right")
                right_path = os.path.join(input_dir, right_filename)

                if os.path.exists(right_path):
                    image_pairs.append((left_path, right_path))
                    match = re.search(r'(\d+)', filename)
                    patient_id = match.group(1) if match else f"patient_{len(patient_ids)}"
                    patient_ids.append(patient_id)

    print(f"Found {len(image_pairs)} fundus image pairs to process...")

    results_data = []

    for idx, ((left_path, right_path), patient_id) in enumerate(
            zip(image_pairs, patient_ids), 1):
        print(f"\nProcessing {idx}/{len(image_pairs)}")
        print(f"Analyzing patient ID: {patient_id}")

        try:
            text_results, binary_predictions, probabilities = predict(
                model, left_path, right_path, transform, device, threshold,
            )

            # Build the structured result row.
            result_row = {
                'ID':           patient_id,
                'Left-Fundus':  os.path.basename(left_path),
                'Right-Fundus': os.path.basename(right_path),
            }
            for i, code in enumerate(CLASS_CODES):
                result_row[code] = int(binary_predictions[i])

            results_data.append(result_row)

            # Write the human-readable text report.
            output_path = os.path.join(result_dir, f"{patient_id}_result.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("Fundus Image Analysis Report\n")
                f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Patient ID: {patient_id}\n")
                f.write(f"Left-eye image:  {os.path.basename(left_path)}\n")
                f.write(f"Right-eye image: {os.path.basename(right_path)}\n")
                f.write("\n".join(text_results) if text_results
                        else "No significant abnormalities detected.")

            print(f"[OK] Result saved to: {os.path.relpath(output_path)}")

        except Exception as e:
            print(f"[FAIL] Failed to process patient ID {patient_id}: {str(e)}")

    # --- Write results back to Excel. ---
    if df is not None:
        disease_cols = {name: code for name, code in zip(CLASS_NAMES, CLASS_CODES)}
        results_df = pd.DataFrame(results_data)

        # Merge predictions back into the original spreadsheet.
        for _, result_row in results_df.iterrows():
            patient_id = result_row['ID']
            mask = df['ID'] == patient_id

            if mask.any():
                for class_name in CLASS_NAMES:
                    col_name = disease_cols[class_name]
                    if col_name in df.columns:
                        value = int(result_row[col_name])
                        df.loc[mask, col_name] = value
                        print(f"Updated {col_name} for patient {patient_id} -> {value}")

        output_file = output_excel or excel_path.replace('.xlsx', '_results.xlsx')
        df.to_excel(output_file, index=False)
        print(f"\nPredictions written to Excel: {output_file}")
    else:
        # Create a standalone results spreadsheet.
        results_df = pd.DataFrame(results_data)
        output_file = output_excel or os.path.join(input_dir, "prediction_results.xlsx")
        results_df.to_excel(output_file, index=False)
        print(f"\nPredictions written to new Excel file: {output_file}")

    print(f"\nBatch processing complete! Text results saved to: {os.path.relpath(result_dir)}")

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
    batch_predict(model, args.input_dir, transform, device, args.threshold,
                  args.excel, args.output_excel)
    print("\ndone")

if __name__ == "__main__":
    main()
