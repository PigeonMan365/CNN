import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
import plotly.graph_objects as go
import plotly.io as pio

# ------------------------------------------------------------
# Resolve project structure
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
TEST_MODELS_DIR = THIS_FILE.parent                     # CNN/test_models/
PROJECT_ROOT = TEST_MODELS_DIR.parent                  # CNN/
TRAINING_DIR = PROJECT_ROOT / "training"               # CNN/training/
MODEL_DIR = TEST_MODELS_DIR / "model"                  # CNN/test_models/model/

sys.path.insert(0, str(TRAINING_DIR))

from model import MalNetFocusAug
from visualize.utils import preprocess_for_visualization, load_yaml_config
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
        "view_mode": "2D View",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_layer_index():
    view_order = st.session_state.view_order
    if not view_order:
        return 0
    n = len(view_order)
    st.session_state.layer_idx = max(0, min(st.session_state.layer_idx, n - 1))
    return st.session_state.layer_idx


def step_layer(delta: int):
    view_order = st.session_state.view_order
    if not view_order:
        return
    n = len(view_order)
    idx = get_layer_index() + delta
    st.session_state.layer_idx = max(0, min(idx, n - 1))


# ------------------------------------------------------------
# Model loading
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
        attention=True,
    )
    state = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ------------------------------------------------------------
# Complexity filter
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

    # Ensure head.1 is included
    for name in layer_order:
        if "head.1" in name and name not in simplified:
            simplified.append(name)

    return simplified


# ------------------------------------------------------------
# Conv weights
# ------------------------------------------------------------
def extract_conv_weights(model):
    conv_weights = {}
    for name, module in model.named_modules():
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            w = module.weight.data.cpu()
            if w.ndim == 4:
                conv_weights[name] = w
    return conv_weights


# ------------------------------------------------------------
# Color mapping
# ------------------------------------------------------------
def array_to_rwb_image(arr: np.ndarray) -> Image.Image:
    if arr.size == 0:
        return Image.new("RGB", (1, 1), (255, 255, 255))

    max_abs = np.max(np.abs(arr)) or 1.0
    norm = np.clip(arr.astype(np.float32) / max_abs, -1.0, 1.0)

    h, w = norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    pos = norm > 0
    neg = norm < 0

    t_pos = norm[pos]
    rgb[pos, 0] = (255 * (1 - t_pos)).astype(np.uint8)
    rgb[pos, 1] = (255 * (1 - t_pos)).astype(np.uint8)
    rgb[pos, 2] = 255

    t_neg = np.abs(norm[neg])
    rgb[neg, 0] = 255
    rgb[neg, 1] = (255 * t_neg).astype(np.uint8)
    rgb[neg, 2] = (255 * t_neg).astype(np.uint8)

    zero = norm == 0
    rgb[zero] = np.array([255, 255, 255], dtype=np.uint8)

    return Image.fromarray(rgb)


def scalar_to_rwb_color(val: float) -> str:
    val = float(np.clip(val, -1.0, 1.0))
    if val > 0:
        t = val
        r = int(255 * (1 - t))
        g = int(255 * (1 - t))
        b = 255
    elif val < 0:
        t = abs(val)
        r = 255
        g = int(255 * t)
        b = int(255 * t)
    else:
        r = g = b = 255
    return f"rgb({r},{g},{b})"


# ------------------------------------------------------------
# Current layer info
# ------------------------------------------------------------
def render_current_layer_info():
    activations = st.session_state.activations
    view_order = st.session_state.view_order
    layer_shapes = st.session_state.layer_shapes

    if activations is None or not view_order:
        return "No layer"

    idx = get_layer_index()
    lname = view_order[idx]
    shape = layer_shapes.get(lname)

    if shape is not None and len(shape) == 4:
        _, C, H, W = shape
        return f"`{lname}` — C:{C} H:{H} W:{W}"
    return f"`{lname}`"


# ------------------------------------------------------------
# Precompute 2D images
# ------------------------------------------------------------
def precompute_2d_images(activations, view_order, conv_weights):
    pre = {}
    for lname in view_order:
        vol = activations[lname]

        # Force head layers into dense-like
        if "head" in lname:
            arr = vol
            if arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            pre[lname] = {"type": "dense_like", "data": arr[0]}
            continue

        if vol.ndim == 4:
            _, C, H, W = vol.shape
            feat = vol[0]
            w = conv_weights.get(lname)

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

            pre[lname] = {"type": "conv_like", "data": layer_data}
        else:
            arr = vol
            if arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            pre[lname] = {"type": "dense_like", "data": arr[0]}
    return pre


# ------------------------------------------------------------
# 3D helpers
# ------------------------------------------------------------
def downsample_2d(arr: np.ndarray, max_size: int = 16) -> np.ndarray:
    """Aggressive downsampling for efficiency."""
    h, w = arr.shape
    if h <= max_size and w <= max_size:
        return arr
    sh = max(1, h // max_size)
    sw = max(1, w // max_size)
    return arr[::sh, ::sw]


def precompute_architecture_3d(activations, view_order):
    """
    For each 4D layer (N,C,H,W), store per-channel per-pixel activations.
    We keep x/y grids and normalized values; Z will be assigned densely later.
    """
    arch_data = []

    for lname in view_order:
        vol = activations[lname]
        if vol.ndim != 4:
            continue

        _, C, H, W = vol.shape
        vol0 = vol[0]

        max_abs = np.max(np.abs(vol0)) or 1.0
        norm = np.clip(vol0 / max_abs, -1.0, 1.0)

        layer_surfaces = []
        for c in range(C):
            fmap = norm[c]
            fmap_ds = downsample_2d(fmap, max_size=16)
            Hds, Wds = fmap_ds.shape

            x = np.linspace(0, W - 1, Wds)
            y = np.linspace(0, H - 1, Hds)
            X, Y = np.meshgrid(x, y)

            layer_surfaces.append(
                dict(
                    x=X,
                    y=Y,
                    values=fmap_ds,
                    name=f"{lname} | map {c}",
                )
            )

        arch_data.append(
            dict(
                layer_name=lname,
                C=C,
                H=H,
                W=W,
                surfaces=layer_surfaces,
            )
        )

    return arch_data


# ------------------------------------------------------------
# 2D view
# ------------------------------------------------------------
def render_2d_view():
    pre = st.session_state.precomputed_2d
    view_order = st.session_state.view_order
    if pre is None or not view_order:
        return

    idx = get_layer_index()
    lname = view_order[idx]
    if lname not in pre:
        st.warning("No 2D data for this layer.")
        return

    layer_info = pre[lname]
    ltype = layer_info["type"]

    if ltype == "conv_like":
        st.markdown("#### 2D Feature Maps and Filters")
        layer_data = layer_info["data"]
        C = len(layer_data)

        has_filters = any(f is not None for _, f in layer_data)
        if has_filters:
            c = 0
            while c < C:
                cols = st.columns(4)
                act1, filt1 = layer_data[c]
                with cols[0]:
                    st.image(act1, caption=f"Feature map {c}", width="stretch")
                with cols[1]:
                    if filt1 is not None:
                        st.image(filt1, caption=f"Filter {c}", width="stretch")
                    else:
                        st.image(act1, caption=f"Feature map {c} (no filter)", width="stretch")

                if c + 1 < C:
                    act2, filt2 = layer_data[c + 1]
                    with cols[2]:
                        st.image(act2, caption=f"Feature map {c+1}", width="stretch")
                    with cols[3]:
                        if filt2 is not None:
                            st.image(filt2, caption=f"Filter {c+1}", width="stretch")
                        else:
                            st.image(act2, caption=f"Feature map {c+1} (no filter)", width="stretch")
                c += 2
        else:
            st.markdown("#### 2D Feature Maps (No Filters)")
            idx_map = 0
            while idx_map < C:
                cols = st.columns(4)
                for i in range(4):
                    if idx_map >= C:
                        break
                    act_img, _ = layer_data[idx_map]
                    with cols[i]:
                        st.image(act_img, caption=f"Feature map {idx_map}", width="stretch")
                    idx_map += 1

    else:
        st.markdown("#### Dense / Head Layer Activations")
        vec = layer_info["data"].astype(np.float32)
        max_abs = np.max(np.abs(vec)) or 1.0
        norm = np.clip(vec / max_abs, -1.0, 1.0)

        cols_per_row = 16
        idx_n = 0
        while idx_n < len(norm):
            cols = st.columns(cols_per_row)
            for i in range(cols_per_row):
                if idx_n >= len(norm):
                    break
                val = norm[idx_n]
                color = scalar_to_rwb_color(val)
                with cols[i]:
                    st.markdown(
                        f"<div style='width:20px;height:20px;background:{color};"
                        f"border-radius:4px;border:1px solid #555;margin:auto;'></div>"
                        f"<div style='text-align:center;font-size:10px;color:#ffffff;'>n{idx_n}</div>",
                        unsafe_allow_html=True,
                    )
                idx_n += 1


# ------------------------------------------------------------
# 3D view (static image, dense Z, progressive build)
# ------------------------------------------------------------
def render_3d_architecture():
    arch_data = st.session_state.precomputed_arch_3d
    view_order = st.session_state.view_order
    if arch_data is None or not view_order:
        return

    idx = get_layer_index()

    # All layers up to current
    visible_names = set(view_order[: idx + 1])
    visible_layers = [d for d in arch_data if d["layer_name"] in visible_names]

    if not visible_layers:
        st.info("No 3D representation for these layers.")
        return

    fig = go.Figure()

    z_cursor = 0

    for layer_i, layer_entry in enumerate(visible_layers):
        # More aggressive downsampling for earlier layers
        if layer_i < len(visible_layers) - 3:
            local_downsample = 8
        else:
            local_downsample = 16

        for surf in layer_entry["surfaces"]:
            X = surf["x"]
            Y = surf["y"]
            V = surf["values"]

            # Downsample again if needed
            if local_downsample != 16:
                H, W = V.shape
                sh = max(1, H // local_downsample)
                sw = max(1, W // local_downsample)
                V = V[::sh, ::sw]
                x = np.linspace(0, X.max(), V.shape[1])
                y = np.linspace(0, Y.max(), V.shape[0])
                X, Y = np.meshgrid(x, y)

            Z = np.full_like(X, z_cursor, dtype=float)

            fig.add_surface(
                x=X,
                y=Y,
                z=Z,
                surfacecolor=V,
                colorscale=[
                    [0.0, "rgb(255,0,0)"],
                    [0.5, "rgb(255,255,255)"],
                    [1.0, "rgb(0,0,255)"],
                ],
                cmin=-1.0,
                cmax=1.0,
                showscale=False,
                hoverinfo="name",
                opacity=0.75,
            )

            z_cursor += 1

    fig.update_layout(
        scene=dict(
            xaxis_title="Width",
            yaxis_title="Height",
            zaxis_title="Channel index (dense stack)",
            xaxis=dict(backgroundcolor="#1a1a1a", gridcolor="#444444", zerolinecolor="#666666", color="white"),
            yaxis=dict(backgroundcolor="#1a1a1a", gridcolor="#444444", zerolinecolor="#666666", color="white"),
            zaxis=dict(backgroundcolor="#1a1a1a", gridcolor="#444444", zerolinecolor="#666666", color="white"),
        ),
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(color="white"),
        uirevision="constant",
    )

    # Static image with fallback
    try:
        img_bytes = pio.to_image(fig, format="png")
        st.image(img_bytes, caption="3D Architecture (static)")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# Main UI
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="CNN Visualization", layout="wide")
    init_session_state()

    st.markdown(
        """
        <style>
        body, .stApp {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        header[data-testid="stHeader"] {
            background-color: #1a1a1a;
        }
        header[data-testid="stHeader"] * {
            color: #ffffff !important;
        }
        button[kind="header"] {
            color: #ffffff !important;
        }
        button[kind="header"]:hover {
            background-color: #555555 !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #3a3a3a !important;
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        .stButton>button {
            background-color: #444444;
            color: #ffffff;
            border-radius: 4px;
            border: 1px solid #666666;
        }
        .stButton>button:hover {
            background-color: #555555;
            border-color: #888888;
        }
        [data-testid="stFileUploader"] section {
            background-color: #444444 !important;
            border: 1px solid #666666 !important;
            color: #ffffff !important;
        }
        [data-testid="stFileUploader"] label {
            color: #ffffff !important;
        }
        div[data-baseweb="select"] > div {
            background-color: #444444;
            color: #ffffff;
        }
        .stRadio > div {
            background-color: #444444;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
        }
        .stRadio label {
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🧠 CNN Visualization Demo")

    # Config
    params = st.query_params
    cfg_path = params.get("config", [st.session_state.cfg_path])[0]
    st.session_state.cfg_path = cfg_path
    cfg = load_yaml_config(cfg_path)

    # Sidebar
    st.sidebar.header("Configuration")

    if not MODEL_DIR.exists():
        st.sidebar.error(f"Model directory does not exist: {MODEL_DIR}")
        return

    pt_files = [p for p in MODEL_DIR.glob("*.pt") if not p.name.endswith(".ts.pt")]
    if not pt_files:
        st.sidebar.error(f"No eager .pt models found in: {MODEL_DIR}")
        return

    model_names = [p.name for p in pt_files]
    st.session_state.model_name = st.sidebar.selectbox(
        "Model",
        model_names,
        index=model_names.index(st.session_state.model_name)
        if st.session_state.model_name in model_names else 0,
    )

    st.session_state.complexity_mode = st.sidebar.selectbox(
        "Complexity filter",
        ["Simplified", "Complete"],
        index=0 if st.session_state.complexity_mode == "Simplified" else 1,
    )

    uploaded = st.sidebar.file_uploader("Upload file")
    st.session_state.uploaded_file = uploaded

    if st.sidebar.button("Render"):
        if st.session_state.uploaded_file is None:
            st.warning("Please upload a file before rendering.")
            return

        with st.spinner("Rendering..."):
            model = load_eager_model(cfg, st.session_state.model_name)
            inp = preprocess_for_visualization(st.session_state.uploaded_file, cfg)
            activations, layer_shapes, layer_order = capture_activations(model, inp)

            view_order = build_view_order(
                layer_order,
                layer_shapes,
                st.session_state.complexity_mode,
            )
            conv_w = extract_conv_weights(model)

            st.session_state.activations = activations
            st.session_state.layer_shapes = layer_shapes
            st.session_state.layer_order = layer_order
            st.session_state.view_order = view_order
            st.session_state.conv_weights = conv_w
            st.session_state.precomputed_2d = precompute_2d_images(
                activations, view_order, conv_w
            )
            st.session_state.precomputed_arch_3d = precompute_architecture_3d(
                activations, view_order
            )
            st.session_state.layer_idx = 0
            st.session_state.rendered = True

        st.success("Done.")

    if not st.session_state.rendered:
        st.info("Upload a file and click Render.")
        return

    # Navigation + view mode
    st.subheader(render_current_layer_info())

    col_prev, col_next, col_view = st.columns([1, 1, 2])
    with col_prev:
        if st.button("⬅️ Previous layer"):
            step_layer(-1)
            st.rerun()
    with col_next:
        if st.button("Next layer ➡️"):
            step_layer(+1)
            st.rerun()
    with col_view:
        st.session_state.view_mode = st.radio(
            "View",
            ["2D View", "3D View"],
            index=0 if st.session_state.view_mode == "2D View" else 1,
            horizontal=True,
        )

    if st.session_state.view_mode == "2D View":
        render_2d_view()
    else:
        st.subheader("🏗️ Model Architecture 3D View (Per-Pixel Cake Layers)")
        render_3d_architecture()


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
