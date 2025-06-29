import numpy as np

def compute_region_stats(magnitudes: np.ndarray, good_old: np.ndarray, image_width: int):
    h = 720  # Fixed height based on 1280x720 resolution
    good_old = good_old.reshape(-1, 2)
    x_coords = good_old[:, 0]
    y_coords = good_old[:, 1]

    # Define bands
    left_mask = x_coords < image_width // 3
    center_mask = (x_coords >= image_width // 3) & (x_coords < 2 * image_width // 3)
    right_mask = x_coords >= 2 * image_width // 3
    probe_band = y_coords < h // 3

    # Magnitudes
    left_mag = np.mean(magnitudes[left_mask]) if np.any(left_mask) else 0
    center_mag = np.mean(magnitudes[center_mask]) if np.any(center_mask) else 0
    right_mag = np.mean(magnitudes[right_mask]) if np.any(right_mask) else 0
    probe_mag = np.mean(magnitudes[center_mask & probe_band]) if np.any(center_mask & probe_band) else 0
    probe_count = int(np.sum(center_mask & probe_band))

    # Feature counts
    left_count = int(np.sum(left_mask))
    center_count = int(np.sum(center_mask))
    right_count = int(np.sum(right_mask))

    return (
        left_mag, center_mag, right_mag,
        probe_mag, probe_count,
        left_count, center_count, right_count
    )
