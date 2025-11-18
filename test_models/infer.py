#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model discovery and inference loop.

Scans models/ for TorchScript models matching cnn_<mode>_<seed>_<iter>.ts.pt,
loads them, and runs inference on preprocessed images.
"""

from __future__ import annotations

import re
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

try:
    from PIL import Image
except ImportError:
    raise ImportError("Missing Pillow. Install: pip install pillow")

try:
    import yaml
except ImportError:
    raise ImportError("Missing PyYAML. Install: pip install pyyaml")


@dataclass
class ModelInfo:
    """Information about a discovered model."""
    path: Path
    mode: str  # 'resize' or 'truncate'
    seed: str
    iter_id: str
    name: str  # Full filename without extension


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return yaml.safe_load(p.read_text()) or {}


def discover_models(models_root: Path) -> List[ModelInfo]:
    """
    Scan models_root for TorchScript models matching cnn_<mode>_<seed>_<iter>.ts.pt.
    
    Returns:
        List of ModelInfo objects sorted by mode, seed, iter_id
    """
    pattern = re.compile(r"^cnn_(resize|truncate)_(\d+)_(\d+)\.ts\.pt$")
    models = []
    
    if not models_root.exists():
        return models
    
    for model_file in models_root.glob("*.ts.pt"):
        match = pattern.match(model_file.name)
        if match:
            mode, seed, iter_id = match.groups()
            models.append(ModelInfo(
                path=model_file,
                mode=mode,
                seed=seed,
                iter_id=iter_id,
                name=model_file.stem  # Without .ts.pt
            ))
    
    # Sort by mode, seed, iter_id
    models.sort(key=lambda m: (m.mode, int(m.seed), int(m.iter_id)))
    return models


def load_model(model_path: Path, device: torch.device) -> torch.jit.ScriptModule:
    """Load a TorchScript model and set to eval mode."""
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model


def get_device(device_str: str = "auto") -> torch.device:
    """Get torch device from string."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_image(image_path: Path) -> torch.Tensor:
    """
    Load PNG image and convert to normalized tensor.
    
    Returns:
        Tensor of shape (1, 1, H, W) with values in [0, 1]
    """
    img = Image.open(image_path).convert("L")  # Grayscale
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor


def infer_single(model: torch.jit.ScriptModule, image_tensor: torch.Tensor, 
                 device: torch.device, threshold: float = 0.5) -> Tuple[float, str]:
    """
    Run inference on a single image.
    
    Args:
        model: TorchScript model
        image_tensor: Input tensor (1, 1, H, W)
        device: Torch device
        threshold: Probability threshold for binary classification
    
    Returns:
        Tuple of (probability, prediction_string)
    """
    model.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.inference_mode():
        logit = model(image_tensor)
        probability = torch.sigmoid(logit).item()
    
    prediction = "malware" if probability >= threshold else "benign"
    return probability, prediction


def infer_all_models(config_path: str = "config.yaml") -> List[Dict]:
    """
    Run inference with all discovered models on all preprocessed images.
    
    Returns:
        List of prediction dictionaries with keys:
        - file: SHA256 or filename
        - model_name: Model identifier
        - mode: resize or truncate
        - probability: Model output probability
        - prediction: "malware" or "benign"
    """
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    inference = cfg.get("inference", {})
    
    models_root = Path(paths.get("models_root", "models")).resolve()
    output_root = Path(paths.get("output_root", "output")).resolve()
    threshold = float(inference.get("threshold", 0.5))
    device = get_device(inference.get("device", "auto"))
    
    # Discover models
    models = discover_models(models_root)
    if not models:
        print(f"[infer] No models found in {models_root}")
        print(f"[infer] Expected pattern: cnn_<mode>_<seed>_<iter>.ts.pt")
        return []
    
    print(f"[infer] Discovered {len(models)} model(s)")
    for m in models:
        print(f"  - {m.name} (mode={m.mode}, seed={m.seed}, iter={m.iter_id})")
    
    # Collect images by mode
    images_by_mode: Dict[str, List[Path]] = {"resize": [], "truncate": []}
    for mode in ["resize", "truncate"]:
        mode_dir = output_root / mode
        if mode_dir.exists():
            images_by_mode[mode] = sorted(mode_dir.glob("*.png"))
    
    if not any(images_by_mode.values()):
        print(f"[infer] No preprocessed images found in {output_root}")
        print(f"[infer] Run 'python main_infer.py preprocess' first")
        return []
    
    total_images = sum(len(imgs) for imgs in images_by_mode.values())
    print(f"[infer] Found {total_images} preprocessed image(s)")
    
    # Run inference
    results = []
    
    for model_info in models:
        mode = model_info.mode
        images = images_by_mode.get(mode, [])
        
        if not images:
            print(f"[infer] Warning: No images for mode '{mode}', skipping {model_info.name}")
            continue
        
        print(f"[infer] Loading model: {model_info.name}")
        try:
            model = load_model(model_info.path, device)
        except Exception as e:
            print(f"[infer] Error loading {model_info.name}: {e}")
            continue
        
        print(f"[infer] Running inference with {model_info.name} on {len(images)} image(s)...")
        
        for img_path in images:
            try:
                image_tensor = load_image(img_path)
                probability, prediction = infer_single(model, image_tensor, device, threshold)
                
                # Extract SHA256 from filename (assuming format: <sha256>.png)
                file_id = img_path.stem
                
                results.append({
                    "file": file_id,
                    "model_name": model_info.name,
                    "mode": mode,
                    "probability": probability,
                    "prediction": prediction
                })
            except Exception as e:
                print(f"[infer] Error processing {img_path}: {e}")
                continue
    
    print(f"[infer] Completed inference: {len(results)} prediction(s)")
    return results


if __name__ == "__main__":
    import sys
    results = infer_all_models()
    print(f"\n[infer] Total predictions: {len(results)}")
    sys.exit(0)

