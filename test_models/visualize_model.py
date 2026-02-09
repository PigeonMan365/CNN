import sys
from pathlib import Path
import math

import numpy as np
import streamlit as st
import torch
from PIL import Image
import plotly.graph_objects as go

# ------------------------------------------------------------
# Resolve project structure
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
TEST_MODELS_DIR = THIS_FILE.parent                     # CNN/test_models/
PROJECT_ROOT = TEST_MODELS_DIR.parent                  # CNN/
TRAINING_DIR = PROJECT_ROOT / "training"               # CNN/training/
MODEL_DIR = TEST_MODELS_DIR / "model"                  # CNN/test_models/model/

# Ensure training/ is importable
sys.path.insert(0, str(TRAINING_DIR))

# Import model architecture
from model import MalNetFocusAug

# Visualization modules
from visualize.utils import (
    preprocess_for_visualization,
    load_yaml_config,
)
from visualize.activation_capture import capture_activations


# ------------------------------------------------------------
# Session-state helpers
# ------------------------------------------------------------
def init_session_state():
    defaults = {
        "model_name": None,
        "complexity_mode": "Simplified",
        "uploaded_file": None,
        "rendered": False,
        "layer_idx": 0,
        "activations": None,
        "layer_shapes": None,
        "layer_order": None,
        "view_order": None,
        "conv_weights": None,
        "cfg_path": "config.yaml",
        "precomputed_2d": None,
        "precomputed_arch_3d": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_layer_index():
    view_order = st.session_state.view_order
    if view_order is None or len(view_order) == 0:
        return 0
    n = len(view_order)
    st.session_state.layer_idx = max(0, min(st.session_state.layer_idx, n - 1))
    return st.session_state.layer_idx


def step_layer(delta: int):
    view_order = st.session_state.view_order
    if view_order is None or len(view_order) == 0:
        return
    n = len(view_order)
    idx = get_layer_index() + delta
    idx = max(0, min(idx, n - 1))
    st.session_state.layer_idx = idx


# ------------------------------------------------------------
# Load eager model (.pt) from CNN/test_models/model/
# ------------------------------------------------------------
def load_eager_model(cfg, model_name: str):
    if not MODEL_DIR.exists():
        st.error(f"Model directory does not exist: {MODEL_DIR}")
        st.stop()

    pt_path = MODEL_DIR / model_name
    if not pt_path.exists():
        st.error(f"Selected model not found: {pt_path}")
        st.stop()

    model = MalNetFocusAug(
        input_size=cfg["inference"]["resize_target_size"][0],
        attention=True
    )

    state = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    return model


# ------------------------------------------------------------
# Complexity filter: simplified vs complete
# ------------------------------------------------------------
def build_view_order(layer_order, layer_shapes, mode: str):
    if mode == "Complete":
        return list(layer_order)

    simplified = []
    last_shape = None
    for name in layer_order:
        shape = layer_shapes.get(name)
        if shape is None:
            continue
        if last_shape is None or shape != last_shape:
            simplified.append(name)
        last_shape = shape
    return simplified


# ------------------------------------------------------------
# Extract conv weights by layer name
# ------------------------------------------------------------
def extract_conv_weights(model):
    conv_weights = {}
    for name, module in model.named_modules():
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            w = module.weight.data.cpu()
            if w.ndim == 4:  # conv-like: (out_channels, in_channels, kH, kW)
                conv_weights[name] = w
    return conv_weights


# ------------------------------------------------------------
# Color mapping: red (-1) -> white (0) -> blue (1)
# ------------------------------------------------------------
def array_to_rwb_image(arr: np.ndarray) -> Image.Image:
    if arr.size == 0:
        return Image.new("RGB", (1, 1), (255, 255, 255))

    max_abs = np.max(np.abs(arr))
    if max_abs == 0:
        norm = np.zeros_like(arr, dtype=np.float32)
    else:
        norm = arr.astype(np.float32) / max_abs
        norm = np.clip(norm, -1.0, 1.0)

    h, w = norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    pos = norm > 0
    neg = norm < 0

    # Positive: white -> blue
    t_pos = norm[pos]
    rgb[pos, 0] = (255 * (1 - t_pos)).astype(np.uint8)
    rgb[pos, 1] = (255 * (1 - t_pos)).astype(np.uint8)
    rgb[pos, 2] = 255

    # Negative: red -> white
    t_neg = np.abs(norm[neg])
    rgb[neg, 0] = 255
    rgb[neg, 1] = (255 * t_neg).astype(np.uint8)
    rgb[neg, 2] = (255 * t_neg).astype(np.uint8)

    zero = norm == 0
    rgb[zero] = np.array([255, 255, 255], dtype=np.uint8)

    return Image.fromarray(rgb)


# ------------------------------------------------------------
# Header / footer rendering
# ------------------------------------------------------------
def render_header(position: str):
    activations = st.session_state.activations
    view_order = st.session_state.view_order
    layer_shapes = st.session_state.layer_shapes

    if activations is None or view_order is None or len(view_order) == 0:
        return

    idx = get_layer_index()
    layer_name = view_order[idx]
    shape = layer_shapes.get(layer_name)

    if shape is not None and len(shape) == 4:
        _, C, H, W = shape
        st.markdown(
            f"### {position} — Current layer: `{layer_name}`  "
            f"&nbsp;&nbsp; Channels: {C}, Height: {H}, Width: {W}"
        )
    else:
        st.markdown(f"### {position} — Current layer: `{layer_name}`")


# ------------------------------------------------------------
# Precompute 2D images (activations + filters)
# ------------------------------------------------------------
def precompute_2d_images(activations, view_order, conv_weights):
    precomputed = {}
    for lname in view_order:
        vol = activations[lname]
        if vol.ndim != 4:
            continue
        _, C, H, W = vol.shape
        feat = vol[0]  # (C, H, W)

        w = conv_weights.get(lname) if conv_weights is not None else None

        layer_data = []
        for c in range(C):
            fmap = feat[c]
            act_img = array_to_rwb_image(fmap)

            filt_img = None
            if w is not None and c < w.shape[0]:
                filt = w[c]
                if filt.ndim == 3:
                    filt_vis = filt.mean(dim=0).numpy()
                else:
                    filt_vis = filt.numpy()
                filt_img = array_to_rwb_image(filt_vis)

            layer_data.append((act_img, filt_img))
        precomputed[lname] = layer_data
    return precomputed


# ------------------------------------------------------------
# Precompute architecture 3D data (cake-layer blocks)
# ------------------------------------------------------------
def precompute_architecture_3d(activations, view_order):
    """
    For each layer, build a 'block' with C horizontal slices (one per feature map),
    each slice a flat HxW plane at z = layer_offset + channel_index.
    Color per slice = mean activation (red->white->blue).
    """
    arch_data = []
    z_offset = 0

    for lname in view_order:
        vol = activations[lname]
        if vol.ndim != 4:
            continue
        _, C, H, W = vol.shape
        vol0 = vol[0]  # (C, H, W)

        means = vol0.reshape(C, -1).mean(axis=1)
        max_abs = np.max(np.abs(means)) if np.max(np.abs(means)) != 0 else 1.0
        norm_means = np.clip(means / max_abs, -1.0, 1.0)

        layer_surfaces = []
        for c_idx in range(C):
            m = norm_means[c_idx]
            if m > 0:
                t = m
                r = int(255 * (1 - t))
                g = int(255 * (1 - t))
                b = 255
            elif m < 0:
                t = abs(m)
                r = 255
                g = int(255 * t)
                b = int(255 * t)
            else:
                r = g = b = 255
            color = f"rgb({r},{g},{b})"

            # Plane coordinates: x in [0, W-1], y in [0, H-1], z constant
            x = np.linspace(0, W - 1, W)
            y = np.linspace(0, H - 1, H)
            X, Y = np.meshgrid(x, y)
            Z = np.full_like(X, z_offset + c_idx, dtype=float)

            layer_surfaces.append(
                dict(
                    x=X,
                    y=Y,
                    z=Z,
                    color=color,
                    name=f"{lname} | map {c_idx}",
                )
            )

        arch_data.append(
            dict(
                layer_name=lname,
                z_offset=z_offset,
                C=C,
                H=H,
                W=W,
                surfaces=layer_surfaces,
            )
        )

        z_offset += C

    return arch_data


# ------------------------------------------------------------
# 2D view: feature maps + corresponding filters (using precomputed images)
# ------------------------------------------------------------
def render_2d_view():
    precomputed_2d = st.session_state.precomputed_2d
    view_order = st.session_state.view_order

    if precomputed_2d is None or view_order is None or len(view_order) == 0:
        return

    idx = get_layer_index()
    layer_name = view_order[idx]

    if layer_name not in precomputed_2d:
        st.warning("No precomputed 2D data for this layer.")
        return

    layer_data = precomputed_2d[layer_name]  # list of (act_img, filt_img)
    C = len(layer_data)

    st.markdown("#### 2D Feature Maps and Filters")

    c = 0
    while c < C:
        cols = st.columns(4)

        # First feature map
        act_img_1, filt_img_1 = layer_data[c]
        with cols[0]:
            st.image(act_img_1, caption=f"Feature map {c}", width="stretch")

        with cols[1]:
            if filt_img_1 is not None:
                st.image(filt_img_1, caption=f"Filter {c}", width="stretch")
            else:
                if c + 1 < C:
                    act_img_extra, _ = layer_data[c + 1]
                    st.image(act_img_extra, caption=f"Feature map {c+1}", width="stretch")
                else:
                    st.write("No filter available.")

        # Second feature map (if exists)
        if c + 1 < C:
            act_img_2, filt_img_2 = layer_data[c + 1]
            with cols[2]:
                st.image(act_img_2, caption=f"Feature map {c+1}", width="stretch")

            with cols[3]:
                if filt_img_2 is not None:
                    st.image(filt_img_2, caption=f"Filter {c+1}", width="stretch")
                else:
                    if c + 2 < C:
                        act_img_extra2, _ = layer_data[c + 2]
                        st.image(act_img_extra2, caption=f"Feature map {c+2}", width="stretch")
                    else:
                        st.write("No filter available.")
        c += 2


# ------------------------------------------------------------
# 3D view: model architecture (cake-layer blocks)
# ------------------------------------------------------------
def render_3d_architecture():
    arch_data = st.session_state.precomputed_arch_3d
    view_order = st.session_state.view_order

    if arch_data is None or view_order is None or len(view_order) == 0:
        return

    idx_current = get_layer_index()
    visible_layers = [d for d in arch_data if d["layer_name"] in view_order[: idx_current + 1]]

    fig = go.Figure()

    for layer in visible_layers:
        for surf in layer["surfaces"]:
            fig.add_surface(
                x=surf["x"],
                y=surf["y"],
                z=surf["z"],
                surfacecolor=np.zeros_like(surf["x"], dtype=float),
                colorscale=[[0, surf["color"]], [1, surf["color"]]],
                showscale=False,
                name=surf["name"],
                hoverinfo="name",
                opacity=0.9,
            )

    fig.update_layout(
        scene=dict(
            xaxis_title="Width",
            yaxis_title="Height",
            zaxis_title="Channel / Layer stack",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    st.plotly_chart(fig, width="stretch")


# ------------------------------------------------------------
# Main UI
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="CNN Visualization", layout="wide")
    init_session_state()

    st.title("🧠 CNN Visualization Demo")

    # --------------------------------------------------------
    # Load config
    # --------------------------------------------------------
    params = st.query_params
    cfg_path = params.get("config", [st.session_state.cfg_path])[0]
    st.session_state.cfg_path = cfg_path
    cfg = load_yaml_config(cfg_path)

    # --------------------------------------------------------
    # Sidebar: menu flow
    # --------------------------------------------------------
    st.sidebar.header("Configuration")

    # 1) Model selection
    if not MODEL_DIR.exists():
        st.sidebar.error(f"Model directory does not exist: {MODEL_DIR}")
        return

    pt_files = [
        p for p in MODEL_DIR.glob("*.pt")
        if not p.name.endswith(".ts.pt")
    ]
    if not pt_files:
        st.sidebar.error(f"No eager .pt models found in: {MODEL_DIR}")
        return

    model_names = [p.name for p in pt_files]
    st.session_state.model_name = st.sidebar.selectbox(
        "Model",
        model_names,
        index=model_names.index(st.session_state.model_name)
        if st.session_state.model_name in model_names
        else 0,
    )

    # 2) Complexity filter selection
    st.session_state.complexity_mode = st.sidebar.selectbox(
        "Complexity filter",
        ["Simplified", "Complete"],
        index=0 if st.session_state.complexity_mode == "Simplified" else 1,
    )

    # 3) File upload
    uploaded = st.sidebar.file_uploader("Upload file", type=None)
    st.session_state.uploaded_file = uploaded

    # 4) Render button
    render_clicked = st.sidebar.button("Render")

    # --------------------------------------------------------
    # Handle render
    # --------------------------------------------------------
    if render_clicked:
        if st.session_state.uploaded_file is None:
            st.warning("Please upload a file before rendering.")
            return

        with st.spinner("Rendering visualization (precomputing all views)..."):
            model = load_eager_model(cfg, st.session_state.model_name)

            input_tensor = preprocess_for_visualization(
                st.session_state.uploaded_file,
                cfg
            )

            activations, layer_shapes, layer_order = capture_activations(
                model,
                input_tensor
            )

            view_order = build_view_order(
                layer_order,
                layer_shapes,
                st.session_state.complexity_mode,
            )

            conv_weights = extract_conv_weights(model)

            precomputed_2d = precompute_2d_images(
                activations,
                view_order,
                conv_weights,
            )

            precomputed_arch_3d = precompute_architecture_3d(
                activations,
                view_order,
            )

            st.session_state.activations = activations
            st.session_state.layer_shapes = layer_shapes
            st.session_state.layer_order = layer_order
            st.session_state.view_order = view_order
            st.session_state.conv_weights = conv_weights
            st.session_state.precomputed_2d = precomputed_2d
            st.session_state.precomputed_arch_3d = precomputed_arch_3d
            st.session_state.layer_idx = 0
            st.session_state.rendered = True

        st.success("Render complete.")

    # --------------------------------------------------------
    # If not rendered yet, stop here
    # --------------------------------------------------------
    if not st.session_state.rendered:
        st.info("Configure options in the sidebar and click **Render**.")
        return

    # --------------------------------------------------------
    # Main page: navigation + 2D + 3D
    # --------------------------------------------------------
    render_header("Header")

    col_prev_top, col_next_top = st.columns(2)
    with col_prev_top:
        if st.button("⬅️ Previous layer", key="prev_top"):
            step_layer(-1)
    with col_next_top:
        if st.button("Next layer ➡️", key="next_top"):
            step_layer(+1)

    render_2d_view()

    st.subheader("🏗️ Model Architecture 3D View (Layer Blocks, Cake-Style)")
    render_3d_architecture()

    render_header("Footer")

    col_prev_bot, col_next_bot = st.columns(2)
    with col_prev_bot:
        if st.button("⬅️ Previous layer", key="prev_bot"):
            step_layer(-1)
    with col_next_bot:
        if st.button("Next layer ➡️", key="next_bot"):
            step_layer(+1)


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
