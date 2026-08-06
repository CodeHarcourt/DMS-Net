# DMS-Net: Dual-Modality Eye Disease Screening Network

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Deep learning model for multi-label eye disease classification from paired fundus images**

</div>

## Overview

DMS-Net is a deep learning model for multi-label eye disease classification from fundus images. By simultaneously processing left- and right-eye images and leveraging a dual-modality feature fusion mechanism, it achieves accurate recognition of eight common ocular conditions.

### Key Features

- **Dual-branch fusion architecture**: Processes left and right eye images together, extracting and fusing bilateral features
- **Calibrated Analogous Semantic Fusion (CASFM)**: Novel cross-eye fusion module for enhanced feature representation
- **OmniPool Spatial Integration (OSIM)**: Combines spatial attention with multi-scale pooling
- **Multi-label classification**: Simultaneously detects eight common eye conditions
- **Mixed precision training**: Accelerated training with automatic mixed precision (AMP)

### Supported Disease Categories

| Class Label | Condition Name |
|:-----------:|:---------------|
| N           | Normal         |
| D           | Diabetes       |
| G           | Glaucoma       |
| C           | Cataract       |
| A           | Age-related Macular Degeneration (AMD) |
| H           | Hypertension   |
| M           | Myopia         |
| O           | Other Abnormality |

## Project Structure

```
DMS-Net/
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── .gitattributes              # Git attribute configuration
├── LICENSE                     # MIT License
├── CONTRIBUTING.md             # Contribution guidelines
├── CITATION.cff                # Citation metadata
│
├── configs/                    # Configuration files
│   └── config.yaml             # Hyperparameter configuration
│
├── data/                       # Data directory
│   ├── raw/                    # Raw dataset (images)
│   ├── processed/              # Preprocessed data
│   ├── train_set_light_cutmix_v1.csv
│   └── val_set_light_cutmix_v1.csv
│
├── src/                        # Source code
│   ├── models/                 # Model definitions
│   │   ├── dmsnet.py           # Main DMS-Net model
│   │   ├── modules.py          # Core modules (OSIM, fusion utils)
│   │   ├── dmff.py             # Alignment modules (CASFM / CIAM / CCAM)
│   │   └── __init__.py
│   │
│   ├── data/                   # Data pipeline
│   │   ├── dataset.py          # Dataset class
│   │   └── __init__.py
│   │
│   ├── utils/                  # Utility functions
│   └── configs/                # Runtime configuration helpers
│
├── scripts/                    # Runnable scripts
│   ├── train.py                # Training entrypoint
│   ├── predict.py              # Single pair prediction
│   ├── batch_predict.py        # Directory-based batch prediction
│   ├── batch_predict_xlsx.py   # Excel-backed batch prediction
│   └── test.py                 # Feature visualization / heatmaps
│
├── checkpoints/                # Saved model weights
└── outputs/                    # Generated artifacts
    ├── predictions/
    └── visualizations/
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA >= 11.0 (recommended for GPU acceleration)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CodeHarcourt/DMS-Net.git
cd DMS-Net
```

### 2. Create a virtual environment (recommended)

```bash
conda create -n dmsnet python=3.8
conda activate dmsnet
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

Organize your dataset as follows:

```
data/
├── train_set_light_cutmix_v1.csv  # Training annotations
├── val_set_light_cutmix_v1.csv    # Validation annotations
└── raw/                           # Image files
```

Expected CSV format:

```csv
filename,left_path,right_path,N,D,G,C,A,H,M,O
001,001_left.jpg,001_right.jpg,1,0,0,0,0,0,0,0
...
```

### Training

```bash
cd scripts
python train.py
```

Main training options:
- `--batch_size`: Batch size (default: 16)
- `--num_epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--data_path`: Dataset root path

### Inference

#### Single image pair

```bash
cd scripts
python predict.py \
  --left  path/to/left.jpg \
  --right path/to/right.jpg \
  --model checkpoints/best_model.pth
```

#### Batch prediction (folder-based)

```bash
cd scripts
python batch_predict.py \
  --input_dir  data/test_images \
  --output     outputs/predictions/results.csv
```

#### Batch prediction with Excel

```bash
cd scripts
python batch_predict_xlsx.py \
  --input_dir     data/test_images \
  --excel         data/test_list.xlsx \
  --output_excel  outputs/predictions/results.xlsx
```

### Feature visualization

```bash
cd scripts
python test.py
```

## Model Architecture

### Core Components

1. **Shared Backbone**: Pretrained ResNet-152 feature extractor (weight-shared for both eyes)
2. **OSIM** (OmniPool Spatial Integrator Module): Multi-scale spatial feature integration
3. **CASFM** (Calibrated Analogous Semantic Fusion Module): Cross-eye semantic fusion with shared pooling
4. **CIAM** (Cross-Modal Integrative Alignment Module): Integrative cross-branch alignment with CAFM
5. **CCAM** (Cross-Modal Contrastive Alignment Module): Contrastive cross-branch alignment
6. **Feature Fusion Head**: Paired-feature fusion controller
7. **ML-Decoder**: Multi-label query-based classifier

### Benchmark

Reported performance on the validation split:

| Metric    | Score |
|-----------|:-----:|
| Accuracy  |   -   |
| Precision |   -   |
| Recall    |   -   |
| F1-Score  |   -   |
| AUC       |   -   |

*Update with numbers after training.*

## Training Recipes

- Mixed precision (AMP) for throughput
- Gradient accumulation (4 steps) to simulate larger batches
- Light augmentations: Resize + ImageNet normalization
- Learning rate: 1e-4
- Optimizer: Adam
- Criterion: BCEWithLogitsLoss

## FAQ

**Q: How can I use my own dataset?**
A: Prepare CSV annotations in the format shown above and organize the images under `data/raw/`.

**Q: I'm running out of GPU memory.**
A: Reduce `--batch_size` or increase the gradient accumulation steps.

**Q: How can I tune the detection threshold?**
A: Adjust the `--threshold` argument in the prediction scripts.

## Contributing

Contributions, issues, and pull requests are welcome!
See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with the [PyTorch](https://pytorch.org/) deep learning framework
- Pretrained weights sourced from ImageNet

## Contact

- Project page: [https://github.com/CodeHarcourt/DMS-Net](https://github.com/CodeHarcourt/DMS-Net)
- Issues: [GitHub Issues](https://github.com/CodeHarcourt/DMS-Net/issues)

---

**Disclaimer**: This project is intended for academic research and educational purposes only. It should not be used directly for clinical diagnosis.
