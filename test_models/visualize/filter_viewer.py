import streamlit as st
import torch
import numpy as np
from torchvision.utils import make_grid
from PIL import Image


def _extract_conv_weights(module):
    """
    Extract convolution weights from a TorchScript module.
    Returns a list of (layer_name, weight_tensor).
    """
    conv_layers = []

    for name, sub in module.named_modules():
        # TorchScript Conv2d modules still expose .weight
        if hasattr(sub, "weight") and isinstance(sub.weight, torch.Tensor):
            w = sub.weight.detach().cpu()
            if w.ndim == 4:  # Conv2d weights: (out_channels, in_channels, H, W)
                conv_layers.append((name, w))

    return conv_layers


def _tensor_to_image(tensor):
    """
    Convert a 3D tensor (C,H,W) or 4D tensor (N,C,H,W) into a PIL image grid.
    """
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    # Normalize for visualization
    t = tensor.clone()
    t -= t.min()
    if t.max() > 0:
        t /= t.max()

    grid = make_grid(t, nrow=8, padding=1)
    np_img = (grid.numpy() * 255).astype(np.uint8)
    np_img = np.transpose(np_img, (1, 2, 0))  # CHW → HWC

    return Image.fromarray(np_img)


def show_filters(model, mode):
    """
    Display convolution filters from the TorchScript model.

    Args:
        model: TorchScript model
        mode: "Simplified (first layer only)" or "Complete (all conv layers)"
    """

    st.subheader("Convolution Filters")

    conv_layers = _extract_conv_weights(model)

    if not conv_layers:
        st.info("No convolution layers found in this model.")
        return

    if mode.startswith("Simplified"):
        # Show only the first conv layer
        name, weights = conv_layers[0]
        st.write(f"**First Conv Layer:** `{name}`")
        img = _tensor_to_image(weights)
        st.image(img, caption=f"Filters of {name}", use_column_width=True)
        return

    # Complete mode: show all conv layers
    for name, weights in conv_layers:
        st.write(f"**Layer:** `{name}` — shape {tuple(weights.shape)}")
        img = _tensor_to_image(weights)
        st.image(img, caption=f"Filters of {name}", use_column_width=True)
