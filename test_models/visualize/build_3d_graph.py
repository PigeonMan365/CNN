import plotly.graph_objects as go
import numpy as np


def _cube(x, y, z, dx, dy, dz, color, opacity=0.25):
    """
    Create a rectangular prism (cube) for Plotly.
    (x, y, z) is the origin corner.
    dx, dy, dz are dimensions.
    """
    return go.Mesh3d(
        x=[x, x+dx, x+dx, x, x, x+dx, x+dx, x],
        y=[y, y, y+dy, y+dy, y, y, y+dy, y+dy],
        z=[z, z, z, z, z+dz, z+dz, z+dz, z+dz],
        color=color,
        opacity=opacity,
        flatshading=True
    )


def _layer_color(index):
    """Generate a visually distinct color per layer."""
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    return colors[index % len(colors)]


def _estimate_params(shape_in, shape_out):
    """
    Estimate parameter count from shapes.
    This is approximate because TorchScript hides kernel sizes.
    But it's good enough for visualization.
    """
    if len(shape_in) == 4 and len(shape_out) == 4:
        cin = shape_in[1]
        cout = shape_out[1]
        # Assume 3x3 conv for visualization purposes
        return cin * cout * 3 * 3
    return 0


def build_3d_architecture(layer_shapes, layer_order, show_names=True, param_mode="none"):
    """
    Build a 3D Plotly figure representing the CNN architecture.

    Args:
        layer_shapes: dict[layer_name] = shape tuple
        layer_order: list of layer names in execution order
        show_names: bool
        param_mode: "none", "total", "per_layer", "running_total"

    Returns:
        fig: Plotly Figure
    """

    fig = go.Figure()

    x_offset = 0
    spacing = 1.5  # space between layers

    running_params = 0

    for idx, layer_name in enumerate(layer_order):
        shape = layer_shapes[layer_name]

        # Expect shape like (1, C, H, W)
        if len(shape) != 4:
            continue

        _, C, H, W = shape

        # Normalize dimensions for visualization
        scale = 0.02
        dy = H * scale
        dz = W * scale
        dx = max(0.5, C * scale * 0.5)

        color = _layer_color(idx)

        # Add cube
        fig.add_trace(
            _cube(
                x_offset, 0, 0,
                dx, dy, dz,
                color=color,
                opacity=0.35
            )
        )

        # Add label
        label = ""
        if show_names:
            label += layer_name

        # Parameter display
        if param_mode != "none":
            # Estimate params using previous layer if possible
            if idx > 0:
                prev_shape = layer_shapes.get(layer_order[idx - 1], None)
                if prev_shape:
                    params = _estimate_params(prev_shape, shape)
                else:
                    params = 0
            else:
                params = 0

            running_params += params

            if param_mode == "per_layer":
                label += f"<br>params: {params:,}"
            elif param_mode == "running_total":
                label += f"<br>total: {running_params:,}"
            elif param_mode == "total":
                # Only show total on last layer
                if idx == len(layer_order) - 1:
                    label += f"<br>total params: {running_params:,}"

        if label:
            fig.add_trace(go.Scatter3d(
                x=[x_offset + dx/2],
                y=[dy + 0.1],
                z=[dz + 0.1],
                mode="text",
                text=[label],
                textposition="top center",
                showlegend=False
            ))

        x_offset += dx + spacing

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis_title="Layer index",
            yaxis_title="Height",
            zaxis_title="Width",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=False
    )

    return fig
