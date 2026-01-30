# Complete System Explanation: Malware Detection CNN

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture & Data Flow](#architecture--data-flow)
3. [File-by-File Breakdown](#file-by-file-breakdown)
4. [Preprocessing Methods](#preprocessing-methods)
5. [Training Pipeline](#training-pipeline)
6. [Model Architecture](#model-architecture)
7. [Configuration System](#configuration-system)
8. [How Everything Works Together](#how-everything-works-together)

---

## System Overview

This is a **malware detection system** that converts binary files (executables, malware samples) into grayscale images and trains a CNN (MalNet-FocusAug) to classify them as benign or malware. The system uses two preprocessing methods for comparative evaluation and includes features for reproducible experiments, resumable training, and TorchScript deployment.

### Key Features
- **Two preprocessing methods**: Resize (lossy compression) and Truncate (entropy-aware selection)
- **Group-aware evaluation**: Prevents data leakage by grouping samples by SHA256 hash
- **Resumable training**: Can interrupt and resume training sessions
- **Multi-run orchestration**: Systematic experiment planning and execution
- **TorchScript export**: Models exported for deployment
- **Comprehensive caching**: File-level and tensor-level caching for performance

---

## Architecture & Data Flow

### High-Level Flow

```
Binary Files (dataset/input/)
    ↓
[Preprocessing: convert.py]
    ├─→ Resize Mode → PNG images (64×64 default)
    └─→ Truncate Mode → PNG images (256×256 default)
    ↓
Conversion Log CSV (logs/conversion_log.csv)
    ↓
[Training: train.py]
    ├─→ Grouped Stratified K-Fold Split
    ├─→ Dataset Loading (with caching)
    ├─→ Model Training (MalNet-FocusAug)
    └─→ Best Model Export (TorchScript)
    ↓
Exported Models (export_models/)
```

### Directory Structure

```
CNN/
├── main.py                    # CLI entry point
├── train.py                   # Training orchestration
├── orchestrate.py             # Multi-run experiment planner
├── verify_setup.py            # Environment verification
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
│
├── preprocessing/              # Binary → Image conversion
│   ├── convert.py            # Main conversion logic
│   └── test_convert.py       # Conversion testing
│
├── training/                  # Model & dataset code
│   ├── dataset.py            # PyTorch Dataset class
│   └── model.py              # MalNet-FocusAug CNN
│
├── utils/                     # Utility modules
│   ├── paths.py              # Config loading & path normalization
│   ├── cache_io.py           # File & tensor caching
│   └── sharded_sampler.py    # Resumable data sampling
│
├── dataset/                   # Data storage
│   ├── input/                 # Original binary files
│   │   ├── benign/           # Benign samples
│   │   └── malware/          # Malware samples
│   └── output/               # Converted PNG images
│       ├── benign/
│       │   ├── resize/       # Resize mode images
│       │   └── truncate/      # Truncate mode images
│       └── malware/
│           ├── resize/
│           └── truncate/
│
├── logs/                      # Logs and metadata
│   ├── conversion_log.csv    # Index of all converted images
│   └── orchestrate_plans.json # Experiment plans
│
├── runs/                      # Training runs
│   ├── seed_state.json       # Seed tracking
│   └── <mode>_seed<seed>/    # Per-run checkpoints
│       ├── fold<fold>_best.pt
│       └── fold<fold>_interrupt.pt
│
├── export_models/             # Deployed models
│   ├── cnn_<mode>_<seed>_<iter>.ts.pt
│   └── cnn_<mode>_<seed>_<iter>.meta.json
│
├── cache/                     # Performance caches
│   └── <hash-based structure>/
│
└── test_models/               # Inference/testing utilities
    ├── infer.py
    ├── preprocess_infer.py
    └── report.py
```

---

## File-by-File Breakdown

### Core Entry Points

#### `main.py` - CLI Hub
**Purpose**: Central command-line interface for all operations

**Key Functions**:
- `cmd_verify()`: Runs environment checks and recommendations
- `cmd_convert()`: Triggers binary-to-image conversion
- `cmd_train()`: Initiates training (with auto-resume logic)
- `cmd_reset()`: Cleans artifacts while preserving dataset
- `cmd_orch_plan()` / `cmd_orch_resume()`: Orchestration commands
- `cmd_test()`: Testing utilities

**Seed Management**:
- Tracks current seed in `runs/seed_state.json`
- Auto-increments after successful training runs
- Supports `--mode both` to train resize+truncate with same seed

**Resume Logic**:
- Automatically detects most recent interrupt checkpoint
- Can resume specific runs with `--resume --mode X --seed Y`

#### `train.py` - Training Orchestrator
**Purpose**: Complete training pipeline with cross-validation, metrics, and export

**Key Components**:
1. **Data Loading**:
   - Reads filtered CSV (by mode: resize/truncate)
   - Creates `ByteImageDataset` with optional caching
   - Windows-safe: forces `num_workers=0` on Windows

2. **Cross-Validation**:
   - Grouped stratified K-fold (prevents data leakage by SHA256)
   - Alternative: Holdout split when `kfold=1`
   - Groups samples by SHA256 to ensure same file never in train+val

3. **Training Loop**:
   - Mild positive oversampling (5-10% positives per batch)
   - Adam/AdamW optimizer with OneCycleLR scheduler
   - Gradient clipping, optional AMP
   - Per-epoch validation with comprehensive metrics

4. **Metrics**:
   - PR-AUC (primary), ROC-AUC, Max F1
   - F1 at operating point (by FPR budget, default 0.1%)
   - Saves best checkpoint per fold by PR-AUC

5. **Resume Support**:
   - Saves interrupt checkpoints (`foldX_interrupt.pt`)
   - Tracks state in `runstate.json`
   - Can resume from specific fold/epoch

6. **Export**:
   - Exports best fold as TorchScript
   - Generates metadata JSON with metrics

#### `orchestrate.py` - Multi-Run Planner
**Purpose**: Systematic experiment execution across multiple seeds

**Plan Format**:
```json
{
  "plans": [{
    "id": 1,
    "created_at": "2024-01-01T00:00:00Z",
    "rounds": 3,
    "jobs": [
      {"mode": "resize", "seed": 0, "status": "DONE"},
      {"mode": "truncate", "seed": 0, "status": "DONE"},
      {"mode": "resize", "seed": 1, "status": "PENDING"},
      ...
    ],
    "cursor": 2
  }]
}
```

**Workflow**:
1. `orchestrate plan --runs N`: Creates plan with N rounds
2. Each round trains resize then truncate with same seed
3. `orchestrate resume`: Executes remaining jobs, skipping DONE
4. Updates seed ledger when plan completes

---

### Preprocessing Module

#### `preprocessing/convert.py` - Binary to Image Converter
**Purpose**: Converts binary files to grayscale PNG images using two methods

**Main Functions**:

1. **`bytes_to_img_resize()`** - Resize Method:
   - Maps all bytes to grayscale pixels (0-255)
   - Fixed width (256px), dynamic height: `ceil(len(bytes) / 256)`
   - Pads only final row with zeros
   - **Progressive downsampling**: Multi-stage resize to reduce aliasing
   - **Hybrid entropy sampling** (optional): Combines high-entropy rows with uniform sampling
   - Resizes to target_size (default 64×64 = 4,096 pixels)
   - **Information loss**: Only represents `target_width × target_height` bytes maximum

2. **`bytes_to_img_entropy_truncate()`** - Truncate Method:
   - Divides file into chunks (default 512 bytes, adaptive)
   - Computes Shannon entropy per chunk
   - **Structural awareness**: Always includes headers/section tables (PE/ELF detection)
   - **Multi-entropy sampling**: Selects from top/mid/low entropy bands
   - **Entropy-weighted allocation** (optional): Proportional allocation by weight
   - Maps selected bytes to target_size (default 256×256 = 65,536 bytes)
   - Can represent up to `target_width × target_height` bytes

3. **`convert_file()`** - Single File Converter:
   - Computes SHA256 hash of input file
   - Output filename: `{sha256}.png`
   - Idempotent: skips if output exists
   - Reads config for target sizes and method options

4. **`rebuild_conversion_log()`** - CSV Indexer:
   - Scans `dataset/output/` directory structure
   - Writes CSV: `rel_path,label,mode,sha256`
   - Used after conversion or manual file changes

5. **`run_all()`** - Main Orchestration:
   - Loads config, ensures directory skeleton
   - Collects input files from `dataset/input/{benign,malware}/`
   - Converts each file to both modes
   - Rebuilds conversion log

**Configuration Options** (from `config.yaml`):
- `resize_target_size`: Target size for resize mode (default [64, 64])
- `truncate_target_size`: Target size for truncate mode (default [256, 256])
- `resize_interpolation`: 'bilinear', 'bicubic', 'lanczos', 'area'
- `resize_entropy_hybrid`: Enable hybrid entropy+uniform sampling
- `resize_entropy_ratio`: Fraction of rows by entropy (0.0-1.0)
- `truncate_chunk_size`: Base chunk size (default 512)
- `truncate_entropy_stratify`: Multi-entropy sampling
- `truncate_entropy_weighted`: Proportional allocation
- `truncate_use_frequency`: Byte-frequency augmentation

---

### Training Module

#### `training/dataset.py` - PyTorch Dataset
**Purpose**: Loads PNG images for training with optional caching

**Class: `ByteImageDataset`**

**Initialization**:
- `csv_path`: Path to conversion log CSV
- `images_root`: Base directory for images
- `normalize`: "01" to normalize pixels to [0,1] (default)
- `use_disk_cache`: Enable file-level LRU cache
- `cache_root`: Cache directory
- `cache_max_bytes`: Cache size limit (e.g., "40GB")
- `decode_cache_mem_mb`: In-RAM tensor cache size (0 to disable)

**Data Loading**:
1. Reads CSV and builds item list: `(rel_path, label, sha256)`
2. Resolves paths: absolute paths honored, relative paths under `images_root`
3. **File staging**: If `use_disk_cache=True`, copies PNGs to fast cache (SSD)
4. **Tensor caching**: If `decode_cache_mem_mb > 0`, caches decoded tensors in RAM
5. Returns: `(tensor[1,H,W], label, rel_path)`

**Caching Strategy**:
- **FileLRU**: LRU cache on disk, evicts by total bytes
- **TensorLRU**: In-memory cache for decoded tensors, evicts by byte count
- Cache keys include file mtime/size to invalidate on changes

#### `training/model.py` - MalNet-FocusAug CNN
**Purpose**: CNN architecture for malware classification

**Architecture**:

```
Input: (B, 1, H, W) grayscale image

Stem:
  Conv2d(1→32, 3×3) → BN → Mish

Residual Stages (4 stages):
  Stage 1: ResidualBlock(32→32, kernel=3×3) → AvgPool2d(2)
  Stage 2: ResidualBlock(32→64, kernel=5×5) → AvgPool2d(2)
  Stage 3: ResidualBlock(64→128, kernel=3×3) → AvgPool2d(2)
  Stage 4: ResidualBlock(128→256, kernel=7×7) → AvgPool2d(2)

Each ResidualBlock:
  - Two conv layers (BN + Mish)
  - SE (Squeeze-and-Excitation) block for channel recalibration
  - Skip connection (1×1 projection if channels differ)

Feature Map: (B, 256, H/16, W/16)

Dual Heads:
  GAP: AdaptiveAvgPool2d(1) → (B, 256)
  Attention: Conv(256→32) → Mish → Conv(32→1) → Spatial Softmax
            → Weighted sum → (B, 256)

Fusion:
  Concatenate [GAP || Attention] → (B, 512)
  MLP: Linear(512→128) → Mish → Linear(128→1)

Output: (B,) single logit (BCEWithLogitsLoss)
```

**Key Features**:
- **Mish activation**: `x * tanh(softplus(x))` throughout
- **SE blocks**: Channel attention after 2nd conv in each residual block
- **Attention mechanism**: Spatial attention on final feature map
- **Size-agnostic**: Works with any input size via adaptive pooling
- **No dropout**: Relies on data augmentation and regularization

---

### Utility Modules

#### `utils/paths.py` - Configuration Loader
**Purpose**: Loads and validates `config.yaml`

**Functions**:
- `load_config()`: Reads YAML, validates required keys, normalizes paths
- Cross-platform path normalization (Windows/Unix)
- Raises `ConfigError` if required keys missing

**Required Keys**:
- `paths`: input_roots, images_root, logs_root, conversion_log, tmp_root, cache_root
- `train_io`: data_csv, images_root, runs_root, run_index

#### `utils/cache_io.py` - Caching Utilities
**Purpose**: File and tensor caching for performance

**Classes**:

1. **`FileLRU`** - Disk Cache:
   - LRU cache for staging PNG files
   - Evicts by total bytes when limit exceeded
   - Hash-based directory structure: `cache/{hash[:2]}/{hash[2:4]}/{hash}.png`
   - Scans existing cache on init

2. **`TensorLRU`** - Memory Cache:
   - In-RAM cache for decoded tensors
   - Evicts least recently used when byte limit exceeded
   - Cache key: `(filepath, mtime, size, normalize_mode)`

**Functions**:
- `parse_bytes()`: Converts strings like "40GB" to bytes

#### `utils/sharded_sampler.py` - Resumable Sampling
**Purpose**: Deterministic, resumable data ordering for large datasets

**Class: `ResumableShardedOrder`**

**Features**:
- Splits dataset into shards of size S
- Shuffles shard order and intra-shard order deterministically
- Maintains cursor `(shard_idx, offset)` for resume
- Serializable state for checkpointing

**Use Case**: Enables resuming training mid-epoch for very large datasets

---

### Verification & Testing

#### `verify_setup.py` - Environment Verification
**Purpose**: Comprehensive environment and dataset diagnostics

**Checks**:
1. **System Info**:
   - Python version, PyTorch version
   - CUDA availability, GPU name, VRAM
   - CPU cores, RAM (total/available)

2. **Dependencies**:
   - Parses `requirements.txt`
   - Tests imports for all packages
   - Reports missing dependencies

3. **Paths & Disk**:
   - Validates paths from config
   - Disk usage for images_root and cache_root
   - Checks if cache and dataset on same device

4. **Dataset**:
   - Summarizes `conversion_log.csv`
   - Counts by mode and label

5. **Auto-Bootstrap**:
   - Creates directory skeleton if missing
   - Rebuilds `conversion_log.csv` from `dataset/output/`

6. **Recommendations**:
   - Batch size (based on VRAM/RAM)
   - num_workers (based on CPU cores)
   - prefetch_batches, pin_memory, persistent_workers
   - AMP (Automatic Mixed Precision) recommendation
   - Cache sizing (based on free disk space)

---

### Configuration System

#### `config.yaml` - Central Configuration
**Purpose**: Single source of truth for all settings

**Sections**:

1. **`paths`**:
   - `input_roots`: List of input directories
   - `images_root`: Output directory for converted images
   - `conversion_log`: Path to CSV index
   - `cache_root`: Cache directory
   - `cache_max_bytes`: Cache size limit

2. **`train_io`**:
   - `data_csv`: Training data CSV (usually same as conversion_log)
   - `images_root`: Images directory (usually same as paths.images_root)
   - `runs_root`: Training runs directory

3. **`training`**:
   - **Metrics**: `primary_global` (pr_auc/roc_auc), `operating_point` (FPR budget)
   - **Splits**: `kfold` (default 1), `holdout` (percentage when kfold=1)
   - **Batching**: `batch_size`, `oversample_pos_range` (e.g., "0.05-0.10")
   - **Optimizer**: `optimizer` (adam/adamw), `scheduler` (none/onecycle), `max_lr`, `weight_decay`, `grad_clip`
   - **Preprocessing sizes**: `resize_target_size`, `truncate_target_size`
   - **Resize options**: `resize_interpolation`, `resize_entropy_hybrid`, `resize_entropy_ratio`
   - **Truncate options**: `truncate_chunk_size`, `truncate_entropy_stratify`, `truncate_entropy_weighted`, `truncate_use_frequency`
   - **Performance**: `epochs`, `num_workers`, `prefetch_batches`, `pin_memory`, `persistent_workers`, `amp`, `device`
   - **Mode**: `mode` (resize/truncate/both)
   - **Export**: `export_root`

---

## Preprocessing Methods

### Resize Mode

**Algorithm**:
1. Map all bytes to grayscale pixels (each byte value 0-255 → one pixel)
2. Fixed width: 256 pixels
3. Dynamic height: `ceil(file_size / 256)`
4. Pad only final row with zeros
5. **Optional**: Hybrid entropy+uniform row selection
6. **Progressive downsampling**: Multi-stage resize (1/2 or 1/4 per stage)
7. Final size: `target_size` (default 64×64)

**Information Loss**:
- A 64×64 image = 4,096 pixels
- Each pixel represents one byte value
- Files larger than 4,096 bytes are compressed (lossy)
- Larger `resize_target_size` preserves more information but uses more memory

**Improvements**:
- **Progressive downsampling**: Reduces aliasing artifacts
- **Hybrid sampling**: Preserves both high-entropy detail and global structure
- **Configurable interpolation**: Lanczos (best quality), bilinear (fast), area (fastest)

### Truncate Mode

**Algorithm**:
1. Divide file into chunks (default 512 bytes, adaptive)
2. Compute Shannon entropy per chunk
3. **Structural detection**: Identify PE/ELF headers, section tables
4. **Selection strategies**:
   - **Stratified**: Select from top/mid/low entropy bands + structural chunks
   - **Weighted**: Allocate bytes proportionally to entropy+structural bonuses
5. Concatenate selected chunks
6. Map to fixed-size image: `target_size` (default 256×256 = 65,536 bytes)

**Advantages**:
- Preserves most informative regions (high entropy = code/encrypted data)
- Ignores low-entropy padding/headers (except structural ones)
- Can represent up to 65,536 bytes (vs 4,096 for 64×64 resize)

**Adaptive Features**:
- Chunk size adapts to file type (PE/ELF get larger chunks)
- Chunk size adapts to file size (very large files use larger chunks)
- Chunk size adapts to mean entropy (high entropy → smaller chunks for precision)

---

## Training Pipeline

### Step-by-Step Flow

1. **Data Preparation**:
   ```
   Binary files → convert.py → PNG images → conversion_log.csv
   ```

2. **Training Initiation**:
   ```
   main.py train --mode resize --seed 0
   ↓
   train.py loads config and CSV
   ```

3. **Data Loading**:
   ```
   train.py reads conversion_log.csv
   ↓
   Filters by mode (resize/truncate)
   ↓
   Creates ByteImageDataset
   ↓
   Optional: FileLRU staging cache
   Optional: TensorLRU decode cache
   ```

4. **Cross-Validation Split**:
   ```
   Grouped stratified K-fold:
   - Groups samples by SHA256 (prevents leakage)
   - Stratifies by label (maintains class balance)
   - Creates K folds
   ```

5. **Per-Fold Training**:
   ```
   For each fold:
     - Create train/val DataLoaders
     - StratifiedRatioBatchSampler (5-10% positives)
     - Initialize model (MalNetFocusAug)
     - Initialize optimizer (AdamW) + scheduler (OneCycleLR)
     - For each epoch:
         - Train on train set
         - Validate on val set
         - Compute metrics (PR-AUC, ROC-AUC, F1)
         - Save best checkpoint (by PR-AUC)
         - Save interrupt checkpoint (on SIGINT)
   ```

6. **Model Selection & Export**:
   ```
   Select best fold (highest PR-AUC)
   ↓
   Load best checkpoint
   ↓
   Export as TorchScript (.ts.pt)
   ↓
   Generate metadata JSON
   ```

### Metrics Computation

**Primary Metric**: PR-AUC (Precision-Recall Area Under Curve)
- Better for imbalanced datasets than ROC-AUC
- Computed via `sklearn.metrics.average_precision_score`

**Additional Metrics**:
- ROC-AUC: Overall discriminative ability
- Max F1: Best F1 score across all thresholds
- F1 at Operating Point: F1 at FPR budget (default 0.1%)
- Threshold at Operating Point: Decision threshold for deployment

**Operating Point Selection**:
- Type: `fpr_budget` (False Positive Rate budget)
- Value: 0.001 (0.1% FPR)
- Finds threshold where FPR ≤ budget, maximizes TPR

### Resume Mechanism

**Interrupt Handling**:
1. SIGINT (Ctrl+C) triggers interrupt handler
2. Saves checkpoint: `foldX_interrupt.pt` (model, optimizer, scheduler, epoch)
3. Saves state: `runstate.json` (current_fold, next_epoch, timestamp)

**Resume Process**:
1. `train.py` detects `runstate.json` and interrupt checkpoint
2. Loads model/optimizer/scheduler state
3. Skips completed folds
4. Continues from saved epoch in current fold

**Auto-Resume**:
- `main.py train` automatically resumes most recent interrupt if no `--resume` flag
- Can explicitly resume: `main.py train --resume --mode X --seed Y`

---

## Model Architecture

### MalNet-FocusAug v3

**Design Philosophy**:
- Lightweight but effective for binary classification
- Attention mechanism to focus on important regions
- SE blocks for channel recalibration
- Mish activation for smooth gradients

**Component Details**:

1. **Stem**:
   - Single 3×3 conv: 1 channel → 32 channels
   - BatchNorm + Mish activation
   - No downsampling (preserves spatial resolution)

2. **Residual Blocks**:
   - Two conv layers per block (BN + Mish)
   - SE block after 2nd conv (channel attention)
   - Skip connection (identity or 1×1 projection)
   - Kernel sizes: 3×3, 5×5, 3×3, 7×7 (increasing receptive field)

3. **Downsampling**:
   - AvgPool2d(2) after each residual block
   - Reduces spatial size by 2× per stage
   - Total reduction: 16× (2^4)

4. **Dual Heads**:
   - **GAP**: Global Average Pooling → (B, 256)
   - **Attention**: Spatial attention weights → weighted sum → (B, 256)
   - Attention computed via: Conv → Mish → Conv → Spatial Softmax

5. **Fusion & Classification**:
   - Concatenate [GAP || Attention] → (B, 512)
   - MLP: 512 → 128 → 1
   - Output: Single logit (BCEWithLogitsLoss)

**Size Handling**:
- Input can be any size (handled via adaptive pooling)
- TorchScript export uses fixed size: 256×256 (configurable)

---

## How Everything Works Together

### Complete Workflow Example

**1. Initial Setup**:
```bash
# Verify environment
python main.py verify
# → Checks system, creates directories, rebuilds CSV
# → Prints recommendations for config.yaml
```

**2. Data Conversion**:
```bash
# Place binaries in dataset/input/{benign,malware}/
python main.py convert
# → preprocessing/convert.py runs
# → Converts each file to both modes
# → Outputs: dataset/output/{benign,malware}/{resize,truncate}/*.png
# → Rebuilds: logs/conversion_log.csv
```

**3. Training**:
```bash
# Train resize mode with seed 0
python main.py train --mode resize --seed 0
# → train.py loads config and CSV
# → Filters CSV by mode="resize"
# → Creates grouped stratified folds
# → Trains model per fold
# → Exports best fold: export_models/cnn_resize_0_1.ts.pt
```

**4. Multi-Experiment**:
```bash
# Create plan for 5 rounds
python main.py orchestrate plan --runs 5
# → orchestrate.py creates plan in logs/orchestrate_plans.json

# Execute plan
python main.py orchestrate resume
# → Executes: resize@seed0, truncate@seed0, resize@seed1, ...
# → Can interrupt and resume
```

**5. Inference** (test_models/):
```bash
# Preprocess new files
python test_models/preprocess_infer.py
# → Converts input files to images

# Run inference
python test_models/main_infer.py
# → Loads TorchScript model
# → Runs inference on preprocessed images
# → Generates reports
```

### Data Flow Diagram

```
┌─────────────────┐
│ Binary Files    │
│ (input/)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ convert.py      │
│ - Resize mode   │
│ - Truncate mode │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ PNG Images      │─────▶│ conversion_log   │
│ (output/)       │      │ .csv             │
└────────┬────────┘      └────────┬─────────┘
         │                         │
         │                         ▼
         │                ┌──────────────────┐
         │                │ train.py         │
         │                │ - Filter by mode │
         │                │ - Create folds   │
         │                └────────┬─────────┘
         │                         │
         │                         ▼
         │                ┌──────────────────┐
         │                │ ByteImageDataset │
         │                │ - Load PNGs      │
         │                │ - Cache (opt)    │
         │                └────────┬─────────┘
         │                         │
         │                         ▼
         │                ┌──────────────────┐
         │                │ MalNetFocusAug   │
         │                │ - Training loop  │
         │                │ - Metrics        │
         │                └────────┬─────────┘
         │                         │
         │                         ▼
         │                ┌──────────────────┐
         │                │ TorchScript      │
         │                │ Export           │
         │                └──────────────────┘
         │
         ▼
┌─────────────────┐
│ export_models/  │
│ *.ts.pt         │
└─────────────────┘
```

### Configuration Flow

```
config.yaml
    │
    ├─→ paths.*
    │   └─→ Used by: convert.py, train.py, verify_setup.py
    │
    ├─→ train_io.*
    │   └─→ Used by: train.py
    │
    └─→ training.*
        ├─→ Preprocessing options → convert.py
        ├─→ Training hyperparameters → train.py
        └─→ Performance settings → train.py
```

### Caching Strategy

```
PNG File (HDD/Network)
    │
    ├─→ FileLRU.stage() (if use_disk_cache=True)
    │   └─→ Copy to cache/ (SSD) for faster I/O
    │
    └─→ Image.open() → PIL Image
        │
        ├─→ TensorLRU.get() (if decode_cache_mem_mb > 0)
        │   └─→ Check in-memory cache
        │
        └─→ Convert to tensor → Normalize → Return
            │
            └─→ TensorLRU.put() (cache decoded tensor)
```

### Seed Management

```
runs/seed_state.json
    │
    ├─→ main.py reads current_seed
    │   └─→ Increments after successful training
    │
    └─→ orchestrate.py reads last_seed
        └─→ Plans start from last_seed + 1
```

---

## Key Design Decisions

### Why Two Preprocessing Methods?

- **Resize**: Simple, preserves global structure, but lossy
- **Truncate**: Preserves most informative regions, less lossy for large files
- **Comparison**: Allows evaluation of which method works better for malware detection

### Why Grouped Stratified Splits?

- **Grouping by SHA256**: Prevents data leakage (same file never in train+val)
- **Stratification**: Maintains class balance across folds
- **Reproducibility**: Deterministic splits from seed

### Why Mild Positive Oversampling?

- **Imbalance**: Malware datasets typically have many more benign samples
- **5-10% positives**: Ensures model sees enough positive examples per batch
- **Not aggressive**: Avoids overfitting to minority class

### Why TorchScript Export?

- **Deployment**: TorchScript models are portable, no Python dependency
- **Performance**: Optimized inference, can use TorchScript JIT compiler
- **Reproducibility**: Frozen model state with metadata

### Why Comprehensive Caching?

- **I/O Bottleneck**: Loading thousands of PNGs can be slow
- **FileLRU**: Staging on fast SSD reduces I/O latency
- **TensorLRU**: Avoids re-decoding same images repeatedly
- **Adaptive**: Cache sizes based on available resources

---

## Summary

This system is a **complete, production-ready malware detection pipeline** with:

1. **Flexible preprocessing**: Two methods for comparative evaluation
2. **Robust training**: Group-aware splits, comprehensive metrics, resume support
3. **Performance optimization**: Multi-level caching, Windows-safe data loading
4. **Experiment management**: Seed tracking, orchestration, reproducible exports
5. **Deployment ready**: TorchScript export with metadata

The architecture is modular, well-documented, and designed for both research (comparing methods) and production (deployable models).

