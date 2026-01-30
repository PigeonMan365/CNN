# Project Usage Guide

Complete breakdown of all commands and their functionality in the Malware Detection CNN system.

---

## Table of Contents

1. [Main Commands](#main-commands)
2. [Command Details](#command-details)
3. [Workflow Examples](#workflow-examples)
4. [Configuration](#configuration)
5. [Directory Structure](#directory-structure)

---

## Main Commands

### Base Command Format
```bash
python main.py <command> [options]
```

All commands accept `--config <path>` to specify a custom config file (default: `config.yaml`)

---

## Command Details

### 1. `python main.py verify`

**Purpose**: Verify environment setup and get system recommendations

**What it does**:
- Checks Python, PyTorch, CUDA, GPU availability
- Inspects CPU cores and RAM
- Validates all required dependencies from `requirements.txt`
- Analyzes dataset from `logs/conversion_log.csv`
- Provides recommendations for:
  - Batch size based on available memory
  - Number of workers for data loading
  - Cache size based on free disk space
  - Whether to use AMP (Automatic Mixed Precision)
  - Pin memory settings

**Auto-creates**:
- `dataset/output/{benign,malware}/{resize,truncate}/` directories
- `logs/` directory
- `cache/` directory
- Rebuilds `logs/conversion_log.csv` by scanning `dataset/output/`

**Usage**:
```bash
python main.py verify
python main.py verify --config custom_config.yaml
```

**When to use**: 
- First-time setup
- After adding new dependencies
- To get optimal configuration recommendations
- To verify dataset structure

---

### 2. `python main.py convert`

**Purpose**: Convert binary files into grayscale PNG images using both preprocessing methods

**What it does**:
- Scans `dataset/input/{benign,malware}/` for files
- Converts each file using both methods:
  - **Resize mode**: Maps all bytes to grayscale, dynamic height, resizes to target_size (default 64×64)
    - NOTE: Information loss occurs during resize - only 4,096 bytes can be represented in 64×64 output
  - **Truncate mode**: Entropy-aware selection of high-information chunks, maps to target_size (default 256×256)
    - Can represent up to 65,536 bytes in 256×256 output
- Outputs to `dataset/output/{benign,malware}/{resize,truncate}/`
- Each PNG is named using the file's SHA256 hash (deterministic)
- Skips files that already have converted images (idempotent)
- Rebuilds `logs/conversion_log.csv` after conversion

**Output structure**:
```
dataset/output/
├── benign/
│   ├── resize/*.png
│   └── truncate/*.png
└── malware/
    ├── resize/*.png
    └── truncate/*.png
```

**Usage**:
```bash
python main.py convert
python main.py convert --config custom_config.yaml
```

**Direct script usage**:
```bash
python preprocessing/convert.py --config config.yaml
python preprocessing/convert.py --rebuild-only  # Only rebuild CSV
python preprocessing/convert.py --skip-convert   # Skip conversion, just rebuild CSV
```

**When to use**:
- After adding new files to `dataset/input/`
- When you need to regenerate all images
- To rebuild the conversion log after manual changes

---

### 3. `python main.py train`

**Purpose**: Train the MalNet-FocusAug CNN model for malware detection

**What it does**:
- Loads images from `dataset/output/` filtered by mode
- Uses grouped stratified K-fold cross-validation (by SHA256 to prevent data leakage)
- Trains CNN with:
  - Mild positive oversampling (5-10% positives per batch)
  - Configurable optimizer (Adam/AdamW)
  - Learning rate scheduling (OneCycleLR)
  - Gradient clipping
  - Optional AMP for faster training
- Computes metrics:
  - PR-AUC (primary metric)
  - ROC-AUC
  - Max F1 score
  - F1 at operating point (by FPR budget)
- Saves best checkpoint per fold
- Exports best fold as TorchScript model to `export_models/`
- Supports interrupt/resume functionality

**Options**:
- `--mode {resize|truncate|both}`: Which preprocessing method to train
  - `resize`: Train only resize mode
  - `truncate`: Train only truncate mode
  - `both`: Train resize then truncate with same seed
- `--seed <int>`: Random seed (default: from seed ledger or config)
- `--resume`: Resume interrupted training (or most recent interrupt)

**Auto-resume behavior**:
- If no `--resume` flag and an interrupt checkpoint exists, automatically resumes
- Interrupt checkpoints saved at: `runs/<mode>_seed<seed>/fold<fold>_interrupt.pt`

**Output**:
- Checkpoints: `runs/<mode>_seed<seed>/fold<fold>_best.pt`
- Exports: `export_models/cnn_<mode>_<seed>_<iter>.ts.pt`
- Metadata: `export_models/cnn_<mode>_<seed>_<iter>.meta.json`

**Usage**:
```bash
# Train with defaults from config.yaml
python main.py train

# Train resize mode with specific seed
python main.py train --mode resize --seed 42

# Train both modes sequentially (same seed)
python main.py train --mode both

# Resume interrupted training
python main.py train --resume

# Resume specific mode/seed
python main.py train --mode resize --seed 5 --resume
```

**When to use**:
- To train new models
- To compare resize vs truncate methods
- To reproduce experiments with specific seeds
- To resume interrupted training sessions

---

### 4. `python main.py reset`

**Purpose**: Clean project artifacts while preserving dataset

**What it deletes**:
- `runs/` - All training runs and checkpoints
- `export_models/` - All exported TorchScript models
- `cache/` - File and tensor caches
- `logs/` - All log files (except preserves dataset)
- `tmp/` - Temporary files
- Seed ledger (`runs/seed_state.json`)

**What it preserves**:
- `dataset/input/` - Original binary files
- `dataset/output/` - Converted PNG images

**What it rebuilds**:
- `logs/conversion_log.csv` - Rebuilt by scanning `dataset/output/`
- Directory skeleton for dataset output

**Usage**:
```bash
python main.py reset
```

**When to use**:
- Before starting fresh experiments
- To free up disk space
- After completing experiments
- When you want to clean artifacts but keep your dataset

---

### 5. `python main.py orchestrate plan`

**Purpose**: Create a resumable multi-run experiment plan

**What it does**:
- Creates a plan with multiple training runs
- Each round trains resize then truncate with the same seed
- Seeds increment sequentially per round
- Plan is saved to `logs/orchestrate_plans.json`
- Can be resumed if interrupted

**Options**:
- `--runs <int>`: Number of rounds to schedule (default: 1)

**Plan format**:
```
Round 0: resize@seed0 → truncate@seed0
Round 1: resize@seed1 → truncate@seed1
Round 2: resize@seed2 → truncate@seed2
```

**Usage**:
```bash
# Create plan for 5 rounds
python main.py orchestrate plan --runs 5

# Create single round plan
python main.py orchestrate plan --runs 1
```

**When to use**:
- When running multiple experiments systematically
- For hyperparameter sweeps
- To ensure consistent seed progression across experiments

---

### 6. `python main.py orchestrate resume`

**Purpose**: Resume execution of an orchestrated experiment plan

**What it does**:
- Loads the most recent plan from `logs/orchestrate_plans.json`
- Executes remaining jobs in the plan
- Skips jobs already marked as "DONE"
- Updates plan status as jobs complete
- Updates seed ledger when plan completes

**Usage**:
```bash
python main.py orchestrate resume
```

**When to use**:
- After creating a plan with `orchestrate plan`
- To continue interrupted orchestrated experiments
- To execute planned experiments in the background

---

### 7. `python main.py test convert`

**Purpose**: Test the conversion system with randomly generated files

**What it does**:
1. Generates random text files (letters, numbers, whitespace)
2. Places them in `dataset/input/{benign,malware}/` directories
3. Runs the conversion process
4. Verifies all files were converted successfully
5. Optionally cleans up test files

**Options**:
- `--num-files <int>`: Number of test files per label (default: 5)
- `--no-cleanup`: Keep test files after test completes
- `--config <path>`: Path to config file

**Usage**:
```bash
# Basic test with 5 files per label
python main.py test convert

# Test with 10 files per label
python main.py test convert --num-files 10

# Test and keep generated files
python main.py test convert --no-cleanup

# Test with custom config
python main.py test convert --config custom_config.yaml
```

**When to use**:
- To verify conversion system is working
- After modifying preprocessing code
- To test on a fresh setup
- For debugging conversion issues

---

## Workflow Examples

### Initial Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify environment
python main.py verify

# 3. (Optional) Test conversion
python main.py test convert

# 4. Place your files in dataset/input/benign/ and dataset/input/malware/

# 5. Convert files to images
python main.py convert

# 6. Train a model
python main.py train --mode resize --seed 0
```

### Comparing Methods
```bash
# Train both methods with same seed
python main.py train --mode both --seed 42

# Or train separately
python main.py train --mode resize --seed 42
python main.py train --mode truncate --seed 42
```

### Multi-Experiment Workflow
```bash
# 1. Create experiment plan
python main.py orchestrate plan --runs 10

# 2. Execute plan (can interrupt and resume)
python main.py orchestrate resume
```

### Resuming Interrupted Training
```bash
# Auto-resume most recent interrupt
python main.py train

# Resume specific run
python main.py train --mode resize --seed 5 --resume
```

---

## Configuration

### Config File: `config.yaml`

**Key sections**:

**Paths**:
- `paths.input_roots`: List of input directories (default: `["dataset/input"]`)
- `paths.images_root`: Output directory for converted images (default: `"dataset/output"`)
- `paths.conversion_log`: Path to conversion log CSV (default: `"logs/conversion_log.csv"`)
- `paths.cache_root`: Cache directory (default: `"cache"`)

**Training**:
- `training.default_mode`: Default mode for training (default: `"resize"`)
- `training.kfold`: Number of folds for cross-validation (default: `3`)
- `training.batch_size`: Batch size (default: `8`)
- `training.epochs`: Number of training epochs (default: `3`)
- `training.optimizer`: Optimizer type `"adam"` or `"adamw"` (default: `"adamw"`)
- `training.scheduler`: Learning rate scheduler `"none"` or `"onecycle"` (default: `"onecycle"`)
- `training.device`: Device `"auto"`, `"cpu"`, or `"cuda"` (default: `"auto"`)

**Metrics**:
- `training.metrics.primary_global`: Primary metric (default: `"pr_auc"`)
- `training.metrics.operating_point.type`: Operating point type (default: `"fpr_budget"`)
- `training.metrics.operating_point.value`: FPR budget (default: `0.001`)

---

## Directory Structure

```
project_root/
├── dataset/
│   ├── input/
│   │   ├── benign/          # Input benign files
│   │   └── malware/         # Input malware files
│   └── output/
│       ├── benign/
│       │   ├── resize/      # Resize mode images
│       │   └── truncate/    # Truncate mode images
│       └── malware/
│           ├── resize/      # Resize mode images
│           └── truncate/    # Truncate mode images
├── logs/
│   ├── conversion_log.csv   # Index of all converted images
│   └── orchestrate_plans.json  # Experiment plans
├── runs/
│   ├── seed_state.json      # Current seed tracker
│   └── <mode>_seed<seed>/
│       ├── fold<fold>_best.pt       # Best checkpoint per fold
│       └── fold<fold>_interrupt.pt  # Interrupt checkpoint
├── export_models/
│   ├── cnn_<mode>_<seed>_<iter>.ts.pt    # TorchScript exports
│   └── cnn_<mode>_<seed>_<iter>.meta.json # Export metadata
├── cache/                    # File and tensor caches
├── tmp/                      # Temporary files
├── preprocessing/            # Conversion scripts
├── training/                 # Model and dataset code
└── utils/                    # Utility functions
```

---

## Command Summary Table

| Command | Purpose | Key Options |
|---------|---------|-------------|
| `verify` | Check environment & get recommendations | `--config` |
| `convert` | Convert binaries to images | `--config` |
| `train` | Train CNN model | `--mode`, `--seed`, `--resume` |
| `reset` | Clean artifacts (preserve dataset) | None |
| `orchestrate plan` | Create multi-run plan | `--runs` |
| `orchestrate resume` | Resume orchestrated plan | None |
| `test convert` | Test conversion with random files | `--num-files`, `--no-cleanup` |

---

## Tips

1. **First Time**: Always run `python main.py verify` to check your setup
2. **Conversion**: Only needed when `dataset/input/` changes
3. **Training**: Use `--mode both` to fairly compare resize vs truncate
4. **Interrupts**: Training automatically saves checkpoints, use `--resume` to continue
5. **Orchestration**: Best for systematic multi-experiment runs
6. **Testing**: Use `test convert` to verify the pipeline before real data

---

End of Usage Guide

