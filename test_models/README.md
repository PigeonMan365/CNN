# Inference Pipeline + Visualization Suite

Standalone inference and visualization system for malware‑detection CNNs using TorchScript models.  
Includes preprocessing, inference, reporting, **full‑resolution feature‑map inspection**, and an **interactive 3D architecture viewer**.

---

## Overview

This project performs inference and evaluation using TorchScript models placed in `infer/models/`.  
It now also includes a Streamlit‑based visualization interface that supports:

- Full‑resolution feature map visualization (no downsampling)
- Per‑layer activation browsing
- Transparent padding for smaller layers
- A 3D architecture viewer with:
  - Layer‑accurate spatial footprints
  - Transparent padded regions
  - 15% enlarged highlight slice for the active layer
  - Correct stacking order and stable rendering
  - Centered, compact plot layout

---

## New Visualization Features

### Full‑Resolution Feature Maps
All activation maps are now displayed at their **true spatial resolution** (e.g., 64×64, 32×32, 16×16).  
Downsampling has been removed to preserve fidelity.

### Transparent Padding for Smaller Layers
Because deeper CNN layers shrink spatially, the viewer pads each slice to match the first layer’s footprint.  
Padded pixels are set to `NaN` and rendered as **fully transparent**, preserving correct spatial alignment.

### 3D Architecture Viewer
A new 3D visualization mode renders the CNN as stacked activation slices:

- First layer defines the global footprint  
- All subsequent layers are padded to match  
- Highlight slice is scaled **15% larger** than the real slice  
- Highlight opacity is **30%**  
- Padded regions of the highlight are also transparent  
- Strict bottom‑to‑top ordering avoids Matplotlib sorting artifacts  
- Plot is centered and scaled to take minimal page space

### Streamlit Rendering Fixes
Streamlit’s media cache was causing missing‑file errors when many images were generated.  
All images are now rendered via **raw PNG bytes**, bypassing the cache entirely.

### Centered, Compact 3D Plot
The 3D figure is now:

- Rendered at a smaller Matplotlib size  
- Not stretched by Streamlit  
- Centered using a 3‑column layout

---

## Project Structure

```
infer/
├── config.yaml
├── main_infer.py
├── preprocess_infer.py
├── infer.py
├── report.py
├── generate_dummy_data.py
├── models/
├── input/
├── output/
└── logs/
```

The visualization UI lives alongside the inference pipeline and loads activation tensors directly from the model during inspection.

---

## Quick Start

### 1. Place Models
Copy TorchScript models into:

```
infer/models/
```

### 2. Prepare Input Files
Organize into:

```
infer/input/benign/
infer/input/malware/
```

### 3. Run Pipeline

```bash
python infer/main_infer.py preprocess
python infer/main_infer.py infer
python infer/main_infer.py report
```

### 4. Launch Visualization UI
Run the Streamlit app:

```bash
streamlit run app.py
```

(Your project may use a different filename; update accordingly.)

---

## Visualization Modes

### 2D Feature Map Viewer
- Displays all channels for each layer
- Full‑resolution images
- No downsampling
- Rendered via raw PNG bytes to avoid Streamlit cache issues

### 3D Architecture Viewer
- Each layer rendered as a surface at its true spatial size
- Smaller layers padded to match the first layer
- Padded regions fully transparent
- Highlight slice enlarged by 15% and semi‑transparent
- Camera zoom controlled by shrink factor and aspect ratio
- Plot centered and compact

---

## Configuration

Edit `infer/config.yaml` to adjust preprocessing, model paths, and inference settings.

---

## Metrics, Reporting, and Model Discovery

All original pipeline behavior remains unchanged:

- Automatic model discovery
- CSV prediction logs
- JSON summary metrics
- ROC‑AUC, PR‑AUC, F1, precision, recall, confusion matrix

---

## Requirements

- Python 3.8+
- PyTorch
- Pillow
- PyYAML
- scikit‑learn
- tqdm
- Streamlit
- Matplotlib
- NumPy

Install:

```bash
pip install torch pillow pyyaml scikit-learn tqdm streamlit matplotlib numpy
```

---

## Notes

- First layer defines the global spatial footprint for visualization
- All deeper layers are padded and rendered transparently
- Highlight slice is always relative to the current layer’s true size
- Streamlit image rendering uses raw bytes to avoid cache eviction errors
- 3D plot is centered and scaled to reduce page footprint
