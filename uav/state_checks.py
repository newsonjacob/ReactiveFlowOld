def in_grace_period(current_time, navigator):
    """Check if the UAV is still within its post-resume grace period."""
    return navigator.just_resumed and current_time < navigator.resume_grace_end_time
