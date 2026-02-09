import plotly.graph_objects as go


def _arc_points(x1, y1, z1, x2, y2, z2, height=0.5, steps=20):
    """
    Generate a smooth 3D arc between two points.
    The arc rises above the architecture by `height`.
    """
    xs = []
    ys = []
    zs = []

    for t in range(steps + 1):
        u = t / steps

        # Linear interpolation
        x = x1 + u * (x2 - x1)
        y = y1 + u * (y2 - y1)
        z = z1 + u * (z2 - z1)

        # Add arc height (parabolic)
        y += height * (1 - (2*u - 1)**2)

        xs.append(x)
        ys.append(y)
        zs.append(z)

    return xs, ys, zs


def add_skip_connections(fig, layer_order, max_gap=4):
    """
    Add simple arc-style skip connections to the 3D architecture.

    Args:
        fig: Plotly figure returned by build_3d_architecture()
        layer_order: list of layer names in execution order
        max_gap: how many layers ahead to search for matching shapes

    Returns:
        fig with skip connections added
    """

    # Extract cube positions from the figure
    cube_positions = []
    for trace in fig.data:
        if isinstance(trace, go.Mesh3d):
            # Mesh3d stores 8 vertices; take the first as origin
            x0 = trace.x[0]
            y0 = trace.y[0]
            z0 = trace.z[0]

            # Compute approximate center
            dx = max(trace.x) - min(trace.x)
            dy = max(trace.y) - min(trace.y)
            dz = max(trace.z) - min(trace.z)

            cx = x0 + dx / 2
            cy = y0 + dy / 2
            cz = z0 + dz / 2

            cube_positions.append((cx, cy, cz))

    # Add arcs between matching layers
    for i in range(len(cube_positions)):
        x1, y1, z1 = cube_positions[i]

        # Look ahead for matching shapes
        for j in range(i + 1, min(i + max_gap + 1, len(cube_positions))):
            x2, y2, z2 = cube_positions[j]

            # Draw arc
            xs, ys, zs = _arc_points(x1, y1, z1, x2, y2, z2, height=0.4)

            fig.add_trace(go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color="orange", width=6),
                showlegend=False
            ))

    return fig
