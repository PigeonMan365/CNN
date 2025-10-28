# Malware → Image Conversion + MalNet-FocusAug Training System

This project converts binaries into grayscale images (compress + truncate modes) and trains a CNN (MalNet-FocusAug) with group-aware evaluation, resumable orchestration, reproducible exports, and TorchScript deployment.

---

## 1) How to use the system

### 1.1 — Create & activate a Python environment

**Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

---

### 1.2 — Install dependencies
```bash
pip install -r requirements.txt
```

---

### 1.3 — Verify environment & get recommended settings
```bash
# prints GPU/CPU/RAM, CSV summary, kfold suggestion,
# recommended batch size, workers, cache size, etc.
python main.py verify
```

Apply any recommendations to `config.yaml` manually.

---

### 1.4 — Convert binaries (only needed when dataset/input changes)
```bash
# runs preprocessing/dual_convert.py and both modes via converter.py
python main.py convert
```

This populates:  
```
dataset/output/compress/(benign|malware)/*.png
dataset/output/truncate/(benign|malware)/*.png
```

---

### 1.5 — Train a single model
```bash
python main.py train
python main.py train --mode compress --seed 3
```

Exports appear in:
```
export_models/cnn_<mode>_<seed>_<iter>.ts.pt
```

---

### 1.6 — Multi-run orchestration (resumable)
```bash
python main.py orchestrate plan --runs 3
python main.py orchestrate resume
```

---

### 1.7 — Reset project artifacts
```bash
python main.py reset --yes
```

---

## 2) Model Architecture (MalNet-FocusAug — medium summary)

Input images are grayscale `(1×256×256)`. Architecture:

1) **Stem Conv (3×3)** → BN → ReLU  
2) **Residual Block 1** — two 3×3 convs with BN+ReLU, skip-add, AvgPool(2)  
3) **Residual Block 2** — two 3×3 convs, skip-add (1×1 projection if needed), AvgPool(2)  
4) **Residual Block 3** — same pattern, then AvgPool(2)

Heads:
- GAP → (128)
- Attention-weighted pool → (128)

Concatenate → Linear(256→128) → ReLU → Linear(128→1).

---

## 3) Files Overview (1 sentence each)

**main.py** — CLI for verify/convert/train/orchestrate/reset.  
**train.py** — training loop, splits, caching, export.  
**orchestrate.py** — resumable multi-run execution planner.  
**verify_setup.py** — environment/data inspection + tuning suggestions.  
**preprocessing/** — safe binary → PNG conversion.  
**training/** — dataset loader + CNN model code.  
**export_models/** — TorchScript models saved for deployment.  
**dataset/** — input binaries and converted PNGs organized by mode+label.

---

## 4) Dataset Note

Conversion must be run in an isolated VM.  
Training uses only `.png` from `dataset/output/**` via conversion_log.csv.

---

End of README.

