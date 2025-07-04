def compute_thresholds(speed):
    """Compute dynamic brake and dodge thresholds based on current speed."""
    brake_thres = 6 + 4 * speed
    dodge_thres = 2 + 1.5 * speed
    return brake_thres, dodge_thres
