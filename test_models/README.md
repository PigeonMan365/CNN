# Inference Pipeline

Standalone inference system for malware detection models using TorchScript models.

## Overview

This project performs inference and evaluation using TorchScript models placed in `infer/models/`. It supports preprocessing, inference, and comprehensive reporting with metrics.

## Project Structure

```
infer/
├── config.yaml              # Configuration file
├── main_infer.py            # CLI entrypoint
├── preprocess_infer.py      # Preprocessing module
├── infer.py                 # Model discovery and inference
├── report.py                # Metrics and reporting
├── generate_dummy_data.py   # Generate test data
├── models/                  # Place TorchScript models here
│   └── cnn_<mode>_<seed>_<iter>.ts.pt
├── input/                   # Input files (organized by label for metrics)
│   ├── benign/             # Benign files (required for metrics)
│   └── malware/             # Malware files (required for metrics)
├── output/                  # Preprocessed PNG images
│   ├── resize/             # Resize mode images
│   └── truncate/           # Truncate mode images
└── logs/                    # Reports and metrics
    ├── infer_report.csv     # Detailed predictions
    └── infer_summary.json   # Aggregated metrics
```

## Quick Start

### 1. Place Models

Copy your TorchScript models to `infer/models/` with the naming pattern:
```
cnn_<mode>_<seed>_<iter>.ts.pt
```

Example:
- `cnn_resize_0_1.ts.pt`
- `cnn_truncate_42_2.ts.pt`

### 2. Prepare Input Files

**For metrics computation (recommended):**
Organize files into labeled directories:
```
infer/input/
├── benign/     # Files known to be benign
└── malware/    # Files known to be malware
```

**For inference only (no metrics):**
Place files directly in `infer/input/` (any structure)

### 3. Run Pipeline

```bash
# Step 1: Preprocess files
python infer/main_infer.py preprocess

# Step 2: Run inference
python infer/main_infer.py infer

# Step 3: Generate report and metrics
python infer/main_infer.py report
```

## Commands

### Preprocess
Converts binary files to PNG images using resize and truncate modes.

```bash
python infer/main_infer.py preprocess
```

- Scans `infer/input/` for files
- Converts to PNGs in `infer/output/{resize,truncate}/`
- Uses SHA256 hash as filenames

### Infer
Runs inference with all discovered models.

```bash
python infer/main_infer.py infer
```

- Discovers models in `infer/models/`
- Loads each model and runs inference
- Generates predictions for all preprocessed images

### Report
Generates CSV report and computes metrics (if labels available).

```bash
python infer/main_infer.py report
```

- Saves predictions to `infer/logs/infer_report.csv`
- Computes metrics if `infer/input/{benign,malware}/` exists
- Saves summary to `infer/logs/infer_summary.json`

## Generating Test Data

Use the dummy data generator to create test files:

```bash
# Generate 10 files per label (default)
python infer/generate_dummy_data.py

# Generate more files
python infer/generate_dummy_data.py --num-files 20

# Generate and preprocess automatically
python infer/generate_dummy_data.py --preprocess

# Generate binary files
python infer/generate_dummy_data.py --binary
```

This creates files in `infer/input/benign/` and `infer/input/malware/` for testing.

## Metrics Computation

**Metrics are only computed when files are organized in `benign/` and `malware/` folders.**

When labels are available, the system computes:
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Confusion Matrix**: TP, TN, FP, FN counts

Metrics are computed per model and aggregated across all models.

## Configuration

Edit `infer/config.yaml` to customize:

```yaml
paths:
  input_root: "infer/input"
  output_root: "infer/output"
  models_root: "infer/models"
  logs_root: "infer/logs"

inference:
  resize_target_size: [64, 64]      # Target size for resize mode
  truncate_target_size: [256, 256]  # Target size for truncate mode
  threshold: 0.5                     # Classification threshold
  device: auto                       # auto | cpu | cuda
```

**Important**: Target sizes must match the training configuration for accurate results.

## Model Discovery

Models are automatically discovered by scanning `infer/models/` for files matching:
```
cnn_<mode>_<seed>_<iter>.ts.pt
```

Where:
- `mode`: `resize` or `truncate`
- `seed`: Training seed (integer)
- `iter`: Iteration/version number (integer)

Example: `cnn_resize_42_1.ts.pt` → mode=resize, seed=42, iter=1

## Output Files

### infer_report.csv
Detailed predictions for each file and model:
```csv
file,model_name,mode,probability,prediction,label
abc123...,cnn_resize_0_1,resize,0.823456,malware,1
abc123...,cnn_truncate_0_1,truncate,0.712345,malware,1
```

### infer_summary.json
Aggregated metrics:
```json
{
  "has_labels": true,
  "n_labeled_files": 20,
  "metrics": {
    "by_model": {
      "cnn_resize_0_1": {
        "roc_auc": 0.95,
        "pr_auc": 0.92,
        "f1": 0.88,
        ...
      }
    },
    "aggregate": {
      "mean_roc_auc": 0.94,
      "mean_pr_auc": 0.91,
      "mean_f1": 0.87
    }
  }
}
```

## Requirements

- Python 3.8+
- PyTorch
- Pillow (PIL)
- PyYAML
- scikit-learn (for metrics)
- tqdm (optional, for progress bars)

Install dependencies:
```bash
pip install torch pillow pyyaml scikit-learn tqdm
```

## Notes

- Files are identified by SHA256 hash for consistency
- Preprocessing is idempotent (skips existing PNGs)
- Models must match the preprocessing target sizes used during training
- Labels are discovered automatically from folder structure
- Without labels, predictions are still generated but metrics cannot be computed

## Troubleshooting

**No models found:**
- Check that models are in `infer/models/`
- Verify naming pattern: `cnn_<mode>_<seed>_<iter>.ts.pt`

**No metrics computed:**
- Ensure files are in `infer/input/benign/` and `infer/input/malware/`
- Check that labels were discovered (see report output)

**Preprocessing errors:**
- Verify input files are readable
- Check that target sizes match training config

