def format_log_line(
    frame_count, time_now, good_old, smooth_L, smooth_C, smooth_R, flow_std,
    pos, yaw, speed, state_str, collided, obstacle_detected, side_safe,
    brake_thres, dodge_thres, probe_req, actual_fps,
    simgetimage_s, decode_s, processing_s, loop_elapsed
):
    return (
        f"{frame_count},{time_now:.2f},{len(good_old)},"
        f"{smooth_L:.3f},{smooth_C:.3f},{smooth_R:.3f},{flow_std:.3f},"
        f"{pos.x_val:.2f},{pos.y_val:.2f},{pos.z_val:.2f},{yaw:.2f},{speed:.2f},{state_str},{collided},{obstacle_detected},{int(side_safe)},"
        f"{brake_thres:.2f},{dodge_thres:.2f},{probe_req:.2f},{actual_fps:.2f},"
        f"{simgetimage_s:.3f},{decode_s:.3f},{processing_s:.3f},{loop_elapsed:.3f}\n"
    )
