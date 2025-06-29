# uav/utils.py
"""Utility helpers for AirSim drone state and image processing."""
import math
import os
import fnmatch
import logging
import cv2
import numpy as np
import airsim
from analysis.utils import retain_recent_files
from datetime import datetime
from uav.config import FLOW_STD_MAX

logger = logging.getLogger(__name__)

def apply_clahe(gray_image):
    """Improve contrast of a grayscale image using CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    return clahe.apply(gray_image)


def get_yaw(orientation):
    """Return yaw angle in degrees from an AirSim quaternion."""
    return math.degrees(airsim.to_eularian_angles(orientation)[2])


def get_speed(velocity):
    """Compute the speed magnitude of a velocity vector."""
    return np.linalg.norm([velocity.x_val, velocity.y_val, velocity.z_val])


def get_drone_state(client):
    """Fetch position, yaw and speed from the AirSim client."""
    try:
        state = client.getMultirotorState()
    except Exception as e:
        logger.warning("State fetch error: %s", e)
        return airsim.Vector3r(0, 0, 0), 0.0, 0.0
    pos = state.kinematics_estimated.position
    ori = state.kinematics_estimated.orientation
    yaw = get_yaw(ori)
    vel = state.kinematics_estimated.linear_velocity
    speed = get_speed(vel)
    return pos, yaw, speed


def _timestamp_from_name(path: str) -> float:
    """Return a UNIX timestamp parsed from ``full_log_YYYYMMDD_HHMMSS.csv``.

    Falls back to the file's modification time if parsing fails.
    """
    name = os.path.basename(path)
    ts = name[len("full_log_"):-len(".csv")]
    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        return dt.timestamp()
    except Exception:
        return os.path.getmtime(path)


def retain_recent_logs(log_dir: str, keep: int = 5) -> None:
    """Keep only the ``keep`` most recent ``full_log_*.csv`` files."""
    try:
        files = [
            os.path.join(log_dir, f)
            for f in os.listdir(log_dir)
            if fnmatch.fnmatch(f, "full_log_*.csv")
        ]
    except FileNotFoundError:
        logger.warning("\u26A0\uFE0F Log directory '%s' not found.", log_dir)
        return

    files.sort(key=_timestamp_from_name, reverse=True)
    logger.info("\U0001F9F9 Found %d logs, keeping %d most recent.", len(files), keep)

    for old_file in files[keep:]:
        try:
            logger.info("\U0001F5D1\uFE0F Deleting old log: %s", old_file)
            os.remove(old_file)
        except OSError as e:
            logger.warning("\u26A0\uFE0F Could not delete %s: %s", old_file, e)

def should_flat_wall_dodge(
    center_mag: float,
    probe_mag: float,
    probe_count: int,
    min_probe_features: int = 5,
    flow_std: float = 0.0,
    std_threshold: float = FLOW_STD_MAX,
) -> bool:
    """Return True when probe flow is low but has enough features to
    confidently interpret a flat wall straight ahead.

    Parameters
    ----------
    center_mag : float
        Averaged flow magnitude in the central region.
    probe_mag : float
        Flow magnitude in the upper center "probe" band.
    probe_count : int
        Number of tracked features in the probe region.
    min_probe_features : int, optional
        Required feature count to consider the probe reliable.
    flow_std : float, optional
        Standard deviation of all tracked flow magnitudes for the
        current frame.
    std_threshold : float, optional
        Maximum allowed standard deviation before the flow is deemed
        unreliable.
    """

    if flow_std >= std_threshold:
        # Optical flow is noisy – skip this heuristic
        return False

    if probe_count < min_probe_features:
        return False

    return probe_mag < 0.5 and center_mag > 0.7
