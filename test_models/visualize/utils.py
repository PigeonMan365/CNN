import io
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

# Import your real preprocessing functions from preprocess_infer.py
from preprocess_infer import (
    bytes_to_img_resize,
    bytes_to_img_truncate,
    load_config as load_yaml_config
)


# ------------------------------------------------------------
# Helper functions for file handling
# ------------------------------------------------------------

def _is_png(uploaded_file):
    """Check if uploaded file is a PNG by magic bytes."""
    header = uploaded_file.read(8)
    uploaded_file.seek(0)
    return header.startswith(b"\x89PNG\r\n\x1a\n")


def _load_png(uploaded_file):
    """Load uploaded PNG into a grayscale PIL image."""
    data = uploaded_file.read()
    uploaded_file.seek(0)
    return Image.open(io.BytesIO(data)).convert("L")


def _load_binary(uploaded_file):
    """Load uploaded file as raw bytes."""
    data = uploaded_file.read()
    uploaded_file.seek(0)
    return data


# ------------------------------------------------------------
# TorchScript model loader
# ------------------------------------------------------------

def load_torchscript_model(model_path):
    """
    Load a TorchScript model from disk and prepare it for inference.
    Always loads on CPU for visualization.
    """
    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()
    return model


# ------------------------------------------------------------
# Visualization preprocessing
# ------------------------------------------------------------

def preprocess_for_visualization(uploaded_file, cfg):
    """
    Unified preprocessing for visualization.
    Produces grayscale tensors with shapes defined in config:
        resize_target_size: [W, H]
        truncate_target_size: [W, H]

    Accepts ANY uploaded file:
      - Raw binaries (EXE, DLL, ELF, etc.)
      - PNG images
      - Anything else (treated as binary)
    """

    viz_cfg = cfg.get("visualization", {})
    mode = viz_cfg.get("preprocess_mode", "resize").lower()

    resize_w, resize_h = cfg["inference"]["resize_target_size"]
    trunc_w, trunc_h = cfg["inference"]["truncate_target_size"]

    # --------------------------------------------------------
    # Case 1: PNG uploaded → treat as image
    # --------------------------------------------------------
    if _is_png(uploaded_file):
        img = _load_png(uploaded_file)

        if mode == "raw":
            target_w, target_h = img.size

        elif mode == "resize":
            target_w, target_h = resize_w, resize_h
            img = img.resize((target_w, target_h))

        elif mode == "truncate":
            target_w, target_h = trunc_w, trunc_h
            img = img.resize((target_w, target_h))

        else:
            target_w, target_h = resize_w, resize_h
            img = img.resize((target_w, target_h))

    else:
        # --------------------------------------------------------
        # Case 2: Binary uploaded → run malware preprocessing
        # --------------------------------------------------------
        data = _load_binary(uploaded_file)

        if mode == "resize":
            img = bytes_to_img_resize(
                data,
                width=256,
                target_size=(resize_w, resize_h),
                resample=cfg["inference"].get("resize_interpolation", "lanczos"),
                entropy_hybrid=cfg["inference"].get("resize_entropy_hybrid", False),
                entropy_ratio=cfg["inference"].get("resize_entropy_ratio", 0.6)
            )
            target_w, target_h = resize_w, resize_h

        elif mode == "truncate":
            img = bytes_to_img_truncate(
                data,
                target_size=(trunc_w, trunc_h),
                chunk_size=cfg["inference"].get("truncate_chunk_size", 512),
                entropy_stratify=cfg["inference"].get("truncate_entropy_stratify", True),
                entropy_weighted=cfg["inference"].get("truncate_entropy_weighted", False),
                use_frequency=cfg["inference"].get("truncate_use_frequency", False)
            )
            target_w, target_h = trunc_w, trunc_h

        elif mode == "raw":
            img = bytes_to_img_resize(data)
            target_w, target_h = img.size

        else:
            img = bytes_to_img_resize(data, target_size=(resize_w, resize_h))
            target_w, target_h = resize_w, resize_h

        img = img.convert("L")  # ensure grayscale

    # --------------------------------------------------------
    # Convert PIL image → normalized grayscale tensor
    # --------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((target_h, target_w)),
        transforms.ToTensor(),  # produces (1, H, W) for grayscale
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    tensor = transform(img).unsqueeze(0)  # (1, 1, H, W)
    return tensor
