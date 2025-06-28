import argparse
import json
import logging

import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
import numpy as np  # exposed for tests  # noqa: F401

logger = logging.getLogger(__name__)


def load_telemetry(log_path):
    """Load UAV position data from a CSV log file.

    Args:
        log_path: Path to the telemetry ``.csv`` file.

    Returns:
        A tuple ``(data, x_col, y_col, z_col)`` where ``data`` is an ``Nx3``
        array of positions and the remaining values are the column names used
        for ``x``, ``y`` and ``z``.
    """
    df = pd.read_csv(log_path)
    required = {'pos_x', 'pos_y', 'pos_z'}
    missing = required - set(df.columns)
    if missing:
        cols = ', '.join(sorted(missing))
        raise ValueError(f"Missing columns in telemetry file: {cols}")

    x_col, y_col, z_col = 'pos_x', 'pos_y', 'pos_z'
    return df[[x_col, y_col, z_col]].values, x_col, y_col, z_col


def load_obstacles(obstacle_path):
    """Read obstacle definitions from a JSON file.

    Args:
        obstacle_path: Path to the obstacle description ``.json`` file.

    Returns:
        A list of obstacle dictionaries describing the scene.
    """
    with open(obstacle_path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Obstacle data must be a list of objects")

    required = {'name', 'location', 'dimensions', 'rotation'}
    for idx, obj in enumerate(data):
        missing = required - obj.keys()
        if missing:
            cols = ', '.join(sorted(missing))
            raise ValueError(f"Obstacle index {idx} missing fields: {cols}")

    return data


def find_alignment_marker(obstacles, marker_name="PlayerStart_3"):
    """Locate a reference object used to align the UAV path.

    Args:
        obstacles: Iterable of obstacle dictionaries.
        marker_name: Name of the marker object to search for.

    Returns:
        Numpy array containing the marker's ``(x, y, z)`` position.
    """
    for obj in obstacles:
        if obj['name'] == marker_name:
            return np.array(obj['location'])
    raise ValueError(f"Marker '{marker_name}' not found in obstacle data.")


def compute_offset(uav_start, marker_position, scale=1.0):
    """Calculate how much to translate the UAV path for alignment.

    Args:
        uav_start: The first UAV position from the telemetry log.
        marker_position: Position of the alignment marker in obstacle space.
        scale: Multiplicative scale applied to the UAV data.

    Returns:
        ``(x, y, z)`` offset to add to the UAV coordinates.
    """
    return marker_position - uav_start * scale


def draw_box(center, dimensions, rotation_euler):
    """Create line traces representing a 3‑D box.

    Args:
        center: The box center coordinates ``(x, y, z)``.
        dimensions: Box side lengths ``(dx, dy, dz)``.
        rotation_euler: Euler rotation applied to the box.

    Returns:
        A list of ``go.Scatter3d`` objects outlining the box edges.
    """
    cx, cy, cz = center
    dx, dy, dz = [d / 2.0 for d in dimensions]
    corners = np.array([
        [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
        [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz],
    ])
    rot = R.from_euler('xyz', rotation_euler).as_matrix()
    rotated = np.dot(corners, rot.T) + np.array(center)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    lines = []
    for start, end in edges:
        xs, ys, zs = zip(rotated[start], rotated[end])
        lines.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='lines',
                line=dict(color='red', width=2),
                showlegend=False,
            )
        )
    return lines


def build_plot(uav_path, obstacles, offset, scale):
    """Assemble a plotly visualization of the flight path and obstacles.

    Args:
        uav_path: ``Nx3`` array of UAV positions.
        obstacles: List of obstacle dictionaries.
        offset: Translation offset for aligning the UAV path.
        scale: Scale multiplier applied to the UAV positions.

    Returns:
        A ``plotly.graph_objects.Figure`` containing the scene.
    """
    uav_scaled = np.array(uav_path) * scale
    uav_flipped = np.column_stack(
        (uav_scaled[:, 0], -uav_scaled[:, 1], uav_scaled[:, 2])
    )
    uav_aligned = uav_flipped + offset

    path_trace = go.Scatter3d(
        x=uav_aligned[:, 0],
        y=uav_aligned[:, 1],
        z=uav_aligned[:, 2],
        mode='lines',
        line=dict(color='blue', width=4),
        name='UAV Path'
    )

    box_lines = []
    for obs in obstacles:
        name = obs['name']
        loc = np.array(obs['location'], dtype=float)
        dims = np.array(obs['dimensions'], dtype=float)

        if name.startswith("UCX_"):
            continue
        if float(np.sqrt(float((loc ** 2).sum()))) < 1.0:
            # skip objects at origin
            continue
        if float(dims.max()) > 1000:
            continue
        if bool((dims > 200).any()) and dims[1] > 100:
            continue

        box_lines.extend(
            draw_box(
                obs['location'],
                obs['dimensions'],
                obs['rotation'],
            )
        )

    fig = go.Figure(data=[path_trace] + box_lines)
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        title="UAV Path vs Obstacle Course",
        showlegend=True
    )
    return fig


def main():
    """CLI entry point for generating a flight visualization.

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log',
        required=True,
        help="Path to UAV flight log CSV",
    )
    parser.add_argument(
        '--obstacles',
        default="analysis/obstacles.json",
        help="Path to obstacles JSON",
    )
    parser.add_argument(
        '--output',
        default="uav_debug_view.html",
        help="Output HTML file",
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
        help="Scale multiplier for UAV data",
    )
    args = parser.parse_args()

    try:
        telemetry, *_ = load_telemetry(args.log)
    except Exception as e:
        logger.error("Error loading telemetry: %s", e)
        return

    try:
        obstacles = load_obstacles(args.obstacles)
    except Exception as e:
        logger.error("Error loading obstacles: %s", e)
        return

    try:
        marker = find_alignment_marker(obstacles)
    except Exception as e:
        logger.error("Error finding alignment marker: %s", e)
        return

    if len(telemetry) == 0:
        logger.warning("No telemetry data available")
        return

    offset = compute_offset(telemetry[0], marker, scale=args.scale)

    fig = build_plot(telemetry, obstacles, offset, scale=args.scale)
    fig.write_html(args.output)
    logger.info("\u2705 Visualization saved to %s", args.output)


if __name__ == '__main__':
    main()
