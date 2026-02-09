import time
import streamlit as st
import plotly.graph_objects as go


def _highlight_cube(fig, cube_trace, color="yellow", opacity=0.45):
    """
    Create a translucent highlight cube over an existing layer cube.
    cube_trace is a Mesh3d trace from the architecture figure.
    """
    # Extract cube geometry
    xs = cube_trace.x
    ys = cube_trace.y
    zs = cube_trace.z

    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        color=color,
        opacity=opacity,
        flatshading=True,
        showscale=False
    )


def animate_forward_pass(fig, layer_order, activations, autoplay=True, speed=1.0):
    """
    Animate the forward pass through the CNN.

    Args:
        fig: Plotly figure from build_3d_architecture()
        layer_order: list of layer names in execution order
        activations: dict[layer_name] = numpy activation
        autoplay: bool
        speed: seconds per layer
    """

    st.subheader("Forward Pass Controls")

    # Streamlit session state
    if "anim_index" not in st.session_state:
        st.session_state.anim_index = 0
    if "anim_running" not in st.session_state:
        st.session_state.anim_running = autoplay

    # UI controls
    cols = st.columns([1, 1, 1, 3])

    if cols[0].button("⏮ Prev"):
        st.session_state.anim_index = max(0, st.session_state.anim_index - 1)
        st.session_state.anim_running = False

    if cols[1].button("⏯ Play/Pause"):
        st.session_state.anim_running = not st.session_state.anim_running

    if cols[2].button("⏭ Next"):
        st.session_state.anim_index = min(len(layer_order) - 1,
                                          st.session_state.anim_index + 1)
        st.session_state.anim_running = False

    # Extract all cube traces from the architecture figure
    cube_traces = [t for t in fig.data if isinstance(t, go.Mesh3d)]

    # Safety check
    if len(cube_traces) < len(layer_order):
        st.warning("Warning: fewer cubes than layers — skip connections or labels may be interfering.")
        return

    # Determine active layer
    idx = st.session_state.anim_index
    active_layer = layer_order[idx]

    st.write(f"**Active Layer:** `{active_layer}`")

    # Build a new figure with highlight overlay
    new_fig = go.Figure(fig)  # shallow copy

    # Add highlight cube
    highlight = _highlight_cube(cube_traces[idx])
    new_fig.add_trace(highlight)

    # Display updated figure
    st.plotly_chart(new_fig, use_container_width=True)

    # Autoplay logic
    if st.session_state.anim_running:
        time.sleep(speed)
        if st.session_state.anim_index < len(layer_order) - 1:
            st.session_state.anim_index += 1
        else:
            st.session_state.anim_running = False

        # Force rerun
        st.experimental_rerun()
