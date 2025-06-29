# === Standard Library Imports ===
import os
import cv2
import time
import subprocess
import numpy as np
import logging
from datetime import datetime
from queue import Queue
from threading import Thread

# === AirSim Imports ===
import airsim
from airsim import ImageRequest, ImageType

# === Internal Module Imports ===
from uav.overlay import draw_overlay
from uav.navigation_rules import compute_thresholds
from uav.video_utils import start_video_writer_thread
from uav.logging_utils import format_log_line
from uav.perception import OpticalFlowTracker, FlowHistory
from uav.navigation import Navigator
from uav.state_checks import in_grace_period
from uav.scoring import compute_region_stats
from uav.utils import (
    FLOW_STD_MAX, get_drone_state, retain_recent_logs, should_flat_wall_dodge
)
from analysis.utils import retain_recent_views
from uav import config

logger = logging.getLogger(__name__)

# === Perception Processing ===

def perception_loop(tracker, image):
    """Process a single image for optical flow."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if tracker.prev_gray is None:
        tracker.initialize(gray)
        return np.array([]), np.array([]), 0.0
    return tracker.process_frame(gray, time.time())

# === Navigation Step ===

def navigation_step(
    client, navigator, flow_history, good_old, flow_vectors, flow_std,
    smooth_L, smooth_C, smooth_R, probe_mag, probe_count,
    left_count, center_count, right_count, frame_queue, vis_img,
    time_now, frame_count, prev_state, state_history, pos_history, param_refs
):
    """
    Decide and execute navigation action based on perception and state.
    Returns: state_str, obstacle_detected, side_safe, brake_thres, dodge_thres, probe_req
    """
    state_str = "none"
    brake_thres = 0.0
    dodge_thres = 0.0
    probe_req = 0.0
    side_safe = False

    # --- Feature Validity ---
    valid_L = left_count >= config.MIN_FEATURES_PER_ZONE
    valid_C = center_count >= config.MIN_FEATURES_PER_ZONE
    valid_R = right_count >= config.MIN_FEATURES_PER_ZONE

    logger.debug(
        "Flow Magnitudes — L: %.2f, C: %.2f, R: %.2f",
        smooth_L, smooth_C, smooth_R,
    )

    # --- Grace Period Handling ---
    if in_grace_period(time_now, navigator):
        param_refs['state'][0] = "🕒 grace"
        obstacle_detected = 0
        try:
            frame_queue.get_nowait()
        except Exception:
            pass
        frame_queue.put(vis_img)
        return state_str, obstacle_detected, side_safe, brake_thres, dodge_thres, probe_req

    navigator.just_resumed = False

    # --- Navigation Logic ---
    if len(good_old) < 5:
        if smooth_L > 1.5 and smooth_R > 1.5 and smooth_C < 0.2:
            state_str = navigator.brake()
        else:
            state_str = navigator.blind_forward()
    else:
        pos, yaw, speed = get_drone_state(client)
        brake_thres, dodge_thres = compute_thresholds(speed)

        center_high = valid_C and (smooth_C > dodge_thres or smooth_C > 2 * min(smooth_L, smooth_R))
        side_safe = (
            valid_L and valid_R and abs(smooth_L - smooth_R) > 0.3 * smooth_C and (smooth_L < 100 or smooth_R < 100)
        )

        probe_reliable = probe_count > config.MIN_PROBE_FEATURES and probe_mag > 0.05

        # --- Obstacle Handling ---
        if smooth_C > (brake_thres * 1.5):
            state_str = navigator.brake()
            navigator.grace_period_end_time = time_now + 1.5
        elif smooth_C > brake_thres:
            state_str = navigator.brake()
            navigator.grace_period_end_time = time_now + 1.5
        elif center_high and side_safe:
            logger.debug(
                "Flow Magnitudes — L: %.2f, C: %.2f, R: %.2f",
                smooth_L, smooth_C, smooth_R,
            )
            if smooth_L < smooth_R:
                state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction='left')
            else:
                state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction='right')
            navigator.grace_period_end_time = time_now + 1.5
        elif probe_mag < 0.5 and smooth_C > 0.7:
            if should_flat_wall_dodge(smooth_C, probe_mag, probe_count, config.MIN_PROBE_FEATURES, flow_std, FLOW_STD_MAX):
                logger.warning("Flat wall detected — attempting fallback dodge")
                state_str = navigator.dodge(smooth_L, smooth_C, smooth_R)
                navigator.grace_period_end_time = time_now + 1.5

    # --- Recovery/State Maintenance ---
    if state_str == "none":
        if (
            navigator.dodging
            and smooth_C < dodge_thres * 0.9
            and time_now >= navigator.grace_period_end_time
            and not navigator.settling
        ):
            logger.info("Dodge ended — resuming forward at frame %s", frame_count)
            state_str = navigator.resume_forward()
        elif (
            navigator.braked
            and smooth_C < brake_thres * 0.8
            and smooth_L < brake_thres * 0.8
            and smooth_R < brake_thres * 0.8
            and time_now >= navigator.grace_period_end_time
        ):
            logger.info("Brake released — resuming forward at frame %s", frame_count)
            state_str = navigator.resume_forward()
        elif not navigator.braked and not navigator.dodging and time_now - navigator.last_movement_time > 2:
            state_str = navigator.reinforce()
        elif (navigator.braked or navigator.dodging) and speed < 0.2 and smooth_C < 5 and smooth_L < 5 and smooth_R < 5:
            state_str = navigator.nudge()
        elif time_now - navigator.last_movement_time > 4:
            state_str = navigator.timeout_recover()

    # --- Maintain Dodge State if Needed ---
    if (
        state_str == "none"
        and navigator.dodging
        and time_now < navigator.grace_period_end_time
        and isinstance(prev_state, str)
        and prev_state.startswith("dodge")
    ):
        state_str = prev_state

    # --- Update State and History ---
    param_refs['state'][0] = state_str
    obstacle_detected = int('dodge' in state_str or state_str == 'brake')

    pos_hist, _, _ = get_drone_state(client)
    state_history.append(state_str)
    pos_history.append((pos_hist.x_val, pos_hist.y_val))
    if len(state_history) == state_history.maxlen:
        if all(s == state_history[-1] for s in state_history) and state_history[-1].startswith("dodge"):
            dx = pos_history[-1][0] - pos_history[0][0]
            dy = pos_history[-1][1] - pos_history[0][1]
            if abs(dx) < 0.5 and abs(dy) < 1.0:
                logger.warning("Repeated dodges detected — extending dodge")
                state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, duration=3.0)
                state_history[-1] = state_str
                param_refs['state'][0] = state_str

    return state_str, obstacle_detected, side_safe, brake_thres, dodge_thres, probe_req


def log_frame_data(log_file, log_buffer, line):
    """Buffer log lines and periodically flush to disk."""
    log_buffer.append(line)
    if len(log_buffer) >= config.LOG_INTERVAL:
        log_file.writelines(log_buffer)
        log_buffer.clear()

def write_video_frame(queue, frame):
    """Queue a video frame for asynchronous writing."""
    try: queue.put_nowait(frame)
    except Exception: pass

def process_perception_data(
    client, args, data, frame_count, frame_queue, flow_history, navigator, param_refs, time_now, max_flow_mag
):
    """
    Process perception output and update histories.
    Returns processed perception data and region statistics.
    """
    vis_img, good_old, flow_vectors, flow_std, simgetimage_s, decode_s, processing_s = data
    gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
    if frame_count == 1 and len(good_old) == 0:
        frame_queue.put(vis_img)
        return None
    if args.manual_nudge and frame_count == 5:
        logger.info("Manual nudge forward for test")
        client.moveByVelocityAsync(2, 0, 0, 2)
    if flow_vectors.size == 0:
        magnitudes = np.array([])
    else:
        if flow_vectors.ndim == 1: flow_vectors = flow_vectors.reshape(-1, 2)
        magnitudes = np.linalg.norm(flow_vectors, axis=1)
    num_clamped = np.sum(magnitudes > max_flow_mag)
    if num_clamped > 0:
        logger.warning("Clamped %d large flow magnitudes to %s", num_clamped, max_flow_mag)
    magnitudes = np.clip(magnitudes, 0, max_flow_mag)
    good_old = good_old.reshape(-1, 2)
    left_mag, center_mag, right_mag, probe_mag, probe_count, left_count, center_count, right_count = compute_region_stats(magnitudes, good_old, gray.shape[1])
    flow_history.update(left_mag, center_mag, right_mag)
    smooth_L, smooth_C, smooth_R = flow_history.average()
    param_refs['L'][0], param_refs['C'][0], param_refs['R'][0] = smooth_L, smooth_C, smooth_R
    if navigator.just_resumed and time_now < navigator.resume_grace_end_time:
        cv2.putText(vis_img, "GRACE", (1100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    in_grace = navigator.just_resumed and time_now < navigator.resume_grace_end_time
    return (
        vis_img, good_old, flow_vectors, flow_std, simgetimage_s, decode_s, processing_s,
        smooth_L, smooth_C, smooth_R, probe_mag, probe_count, left_count, center_count, right_count, in_grace,
    )

def apply_navigation_decision(
    client, navigator, flow_history, good_old, flow_vectors, flow_std,
    smooth_L, smooth_C, smooth_R, probe_mag, probe_count,
    left_count, center_count, right_count, frame_queue, vis_img,
    time_now, frame_count, prev_state, state_history, pos_history, param_refs,
):
    """Wrapper around navigation_step for clarity."""
    return navigation_step(
        client, navigator, flow_history, good_old, flow_vectors, flow_std,
        smooth_L, smooth_C, smooth_R, probe_mag, probe_count,
        left_count, center_count, right_count, frame_queue, vis_img,
        time_now, frame_count, prev_state, state_history, pos_history, param_refs,
    )

def write_frame_output(
    client, vis_img, frame_queue, loop_start, frame_duration, fps_list, start_time,
    smooth_L, smooth_C, smooth_R, left_count, center_count, right_count,
    good_old, flow_vectors, in_grace, frame_count, time_now, param_refs,
    log_file, log_buffer, state_str, obstacle_detected, side_safe,
    brake_thres, dodge_thres, probe_req, simgetimage_s, decode_s, processing_s, flow_std,
):
    """
    Write video frame, overlay, and log data for the current step.
    Returns updated loop_start time.
    """
    pos, yaw, speed = get_drone_state(client)
    collision = client.simGetCollisionInfo()
    collided = int(getattr(collision, "has_collided", False))
    vis_img = draw_overlay(
        vis_img, frame_count, speed, param_refs['state'][0], time_now - start_time,
        smooth_L, smooth_C, smooth_R, left_count, center_count, right_count,
        good_old, flow_vectors, in_grace=in_grace,
    )
    write_video_frame(frame_queue, vis_img)
    elapsed = time.time() - loop_start
    if elapsed < frame_duration: time.sleep(frame_duration - elapsed)
    loop_elapsed = time.time() - loop_start
    actual_fps = 1 / max(loop_elapsed, 1e-6)
    loop_start = time.time()
    fps_list.append(actual_fps)
    log_line = format_log_line(
        frame_count, time_now, good_old, smooth_L, smooth_C, smooth_R, flow_std,
        pos, yaw, speed, state_str, collided, obstacle_detected, side_safe,
        brake_thres, dodge_thres, probe_req, actual_fps,
        simgetimage_s, decode_s, processing_s, loop_elapsed,
    )
    log_frame_data(log_file, log_buffer, log_line)
    logger.debug("Actual FPS: %.2f", actual_fps)
    logger.debug("Features detected: %d", len(good_old))
    return loop_start

def handle_reset(client, ctx, frame_count):
    """
    Reset simulation and restart logging/video.
    Returns reset frame_count.
    """
    param_refs, flow_history, navigator = ctx['param_refs'], ctx['flow_history'], ctx['navigator']
    frame_queue, video_thread, out = ctx['frame_queue'], ctx['video_thread'], ctx['out']
    log_file, log_buffer, fourcc = ctx['log_file'], ctx['log_buffer'], ctx['fourcc']
    logger.info("Resetting simulation...")
    try:
        client.landAsync().join(); client.reset(); client.enableApiControl(True)
        client.armDisarm(True); client.takeoffAsync().join(); client.moveToPositionAsync(0, 0, -2, 2).join()
    except Exception as e: logger.error("Reset error: %s", e)
    ctx['flow_history'], ctx['navigator'], frame_count = FlowHistory(), Navigator(client), 0
    param_refs['reset_flag'][0] = False
    if log_buffer: log_file.writelines(log_buffer); log_buffer.clear()
    log_file.close()
    ctx['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamp = ctx['timestamp']
    log_file = open(f"flow_logs/full_log_{timestamp}.csv", 'w')
    ctx['log_file'] = log_file
    log_file.write(
        "frame,time,features,flow_left,flow_center,flow_right,"
        "flow_std,pos_x,pos_y,pos_z,yaw,speed,state,collided,obstacle,side_safe,"
        "brake_thres,dodge_thres,probe_req,fps,simgetimage_s,decode_s,processing_s,loop_s\n"
    )
    retain_recent_logs("flow_logs")
    frame_queue.put(None)
    video_thread.join()
    out.release()
    out = cv2.VideoWriter(config.VIDEO_OUTPUT, fourcc, config.VIDEO_FPS, config.VIDEO_SIZE)
    ctx['out'] = out
    video_thread = start_video_writer_thread(frame_queue, out, ctx['exit_flag'])
    ctx['video_thread'] = video_thread
    return frame_count

def setup_environment(args, client):
    """Initialize the navigation environment and return a context dict."""
    from uav.interface import exit_flag, start_gui
    from uav.utils import retain_recent_logs
    param_refs = {'L': [0.0], 'C': [0.0], 'R': [0.0], 'state': [''], 'reset_flag': [False]}
    start_gui(param_refs)
    logger.info("Available vehicles: %s", client.listVehicles())
    client.enableApiControl(True); client.armDisarm(True)
    client.takeoffAsync().join(); client.moveToPositionAsync(0, 0, -2, 2).join()
    feature_params = dict(maxCorners=150, qualityLevel=0.05, minDistance=5, blockSize=5)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    tracker, flow_history, navigator = OpticalFlowTracker(lk_params, feature_params), FlowHistory(), Navigator(client)
    from collections import deque
    state_history, pos_history = deque(maxlen=3), deque(maxlen=3)
    start_time = time.time()
    GOAL_X, MAX_SIM_DURATION = args.goal_x, args.max_duration
    logger.info("Config:\n  Goal X: %sm\n  Max Duration: %ss", GOAL_X, MAX_SIM_DURATION)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs("flow_logs", exist_ok=True)
    log_file = open(f"flow_logs/full_log_{timestamp}.csv", 'w')
    log_file.write(
        "frame,time,features,flow_left,flow_center,flow_right,"
        "flow_std,pos_x,pos_y,pos_z,yaw,speed,state,collided,obstacle,side_safe,"
        "brake_thres,dodge_thres,probe_req,fps,simgetimage_s,decode_s,processing_s,loop_s\n"
    )
    retain_recent_logs("flow_logs")
    try: fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    except AttributeError: fourcc = cv2.FOURCC(*'MJPG')
    out = cv2.VideoWriter(config.VIDEO_OUTPUT, fourcc, config.VIDEO_FPS, config.VIDEO_SIZE)
    frame_queue = Queue(maxsize=20)
    video_thread = start_video_writer_thread(frame_queue, out, exit_flag)
    ctx = {
        'exit_flag': exit_flag, 'param_refs': param_refs, 'tracker': tracker, 'flow_history': flow_history,
        'navigator': navigator, 'state_history': state_history, 'pos_history': pos_history,
        'frame_queue': frame_queue, 'video_thread': video_thread, 'out': out, 'log_file': log_file,
        'log_buffer': [], 'timestamp': timestamp, 'start_time': start_time, 'fps_list': [], 'fourcc': fourcc,
    }
    return ctx

def start_perception_thread(ctx):
    """Launch background perception thread."""
    exit_flag, tracker = ctx['exit_flag'], ctx['tracker']
    perception_queue = Queue(maxsize=1)
    last_vis_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    def perception_worker():
        nonlocal last_vis_img
        local_client = airsim.MultirotorClient()
        local_client.confirmConnection()
        request = [ImageRequest("oakd_camera", ImageType.Scene, False, True)]
        while not exit_flag.is_set():
            t0 = time.time()
            responses = local_client.simGetImages(request, vehicle_name="UAV")
            t_fetch_end = time.time()
            response = responses[0]
            if (response.width == 0 or response.height == 0 or len(response.image_data_uint8) == 0):
                data = (last_vis_img, np.array([]), np.array([]), 0.0, t_fetch_end - t0, 0.0, 0.0)
            else:
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8).copy()
                img = cv2.imdecode(img1d, cv2.IMREAD_COLOR)
                t_decode_end = time.time()
                if img is None: continue
                img = cv2.resize(img, config.VIDEO_SIZE)
                vis_img = img.copy()
                last_vis_img = vis_img
                t_proc_start = time.time()
                good_old, flow_vectors, flow_std = perception_loop(tracker, img)
                processing_s = time.time() - t_proc_start
                data = (
                    vis_img, good_old, flow_vectors, flow_std,
                    t_fetch_end - t0, t_decode_end - t_fetch_end, processing_s,
                )
            try: perception_queue.put(data, block=False)
            except Exception: pass
    perception_thread = Thread(target=perception_worker, daemon=True)
    perception_thread.start()
    ctx['perception_queue'] = perception_queue
    ctx['perception_thread'] = perception_thread

def navigation_loop(args, client, ctx):
    """Main navigation loop processing perception results."""
    exit_flag, flow_history, navigator = ctx['exit_flag'], ctx['flow_history'], ctx['navigator']
    param_refs, state_history, pos_history = ctx['param_refs'], ctx['state_history'], ctx['pos_history']
    perception_queue, frame_queue, video_thread = ctx['perception_queue'], ctx['frame_queue'], ctx['video_thread']
    out, log_file, log_buffer = ctx['out'], ctx['log_file'], ctx['log_buffer']
    start_time, timestamp, fps_list, fourcc = ctx['start_time'], ctx['timestamp'], ctx['fps_list'], ctx['fourcc']
    MAX_FLOW_MAG, MAX_VECTOR_COMPONENT = config.MAX_FLOW_MAG, config.MAX_VECTOR_COMPONENT
    GRACE_PERIOD_SEC, MAX_SIM_DURATION = config.GRACE_PERIOD_SEC, args.max_duration
    GOAL_X, GOAL_RADIUS = args.goal_x, config.GOAL_RADIUS
    frame_count, target_fps, frame_duration = 0, config.TARGET_FPS, 1.0 / config.TARGET_FPS
    grace_logged, startup_grace_over = False, False
    try:
        loop_start = time.time()
        while not exit_flag.is_set():
            frame_count += 1
            time_now = time.time()
            if not startup_grace_over:
                if time_now - start_time < GRACE_PERIOD_SEC:
                    param_refs['state'][0] = "startup_grace"
                    if not grace_logged:
                        logger.info("Startup grace period active — waiting to start perception and nav")
                        grace_logged = True
                    time.sleep(0.05)
                    continue
                else:
                    startup_grace_over = True
                    logger.info("Startup grace period complete — beginning full nav logic")
            try: data = perception_queue.get(timeout=1.0)
            except Exception: continue
            prev_state = param_refs['state'][0]
            if navigator.settling and time_now >= navigator.settle_end_time:
                logger.info("Settle period over — resuming evaluation")
                navigator.settling = False
            if time_now - start_time >= MAX_SIM_DURATION:
                logger.info("Time limit reached — landing and stopping."); break
            pos_goal, _, _ = get_drone_state(client)
            if pos_goal.x_val >= GOAL_X - GOAL_RADIUS:
                logger.info("Goal reached — landing."); break
            processed = process_perception_data(
                client, args, data, frame_count, frame_queue, flow_history, navigator, param_refs, time_now, MAX_FLOW_MAG,
            )
            if processed is None: continue
            (
                vis_img, good_old, flow_vectors, flow_std, simgetimage_s, decode_s, processing_s,
                smooth_L, smooth_C, smooth_R, probe_mag, probe_count, left_count, center_count, right_count, in_grace,
            ) = processed
            (
                state_str, obstacle_detected, side_safe, brake_thres, dodge_thres, probe_req,
            ) = apply_navigation_decision(
                client, navigator, flow_history, good_old, flow_vectors, flow_std,
                smooth_L, smooth_C, smooth_R, probe_mag, probe_count,
                left_count, center_count, right_count, frame_queue, vis_img,
                time_now, frame_count, prev_state, state_history, pos_history, param_refs,
            )
            if param_refs['reset_flag'][0]:
                frame_count = handle_reset(client, ctx, frame_count)
                flow_history, navigator, log_file, video_thread, out = ctx['flow_history'], ctx['navigator'], ctx['log_file'], ctx['video_thread'], ctx['out']
                continue
            loop_start = write_frame_output(
                client, vis_img, frame_queue, loop_start, frame_duration, fps_list, start_time,
                smooth_L, smooth_C, smooth_R, left_count, center_count, right_count,
                good_old, flow_vectors, in_grace, frame_count, time_now, param_refs,
                log_file, log_buffer, state_str, obstacle_detected, side_safe,
                brake_thres, dodge_thres, probe_req, simgetimage_s, decode_s, processing_s, flow_std,
            )
    except KeyboardInterrupt:
        logger.info("Interrupted.")

def cleanup(client, sim_process, ctx):
    """Clean up resources and land the drone."""
    exit_flag, frame_queue, video_thread = ctx['exit_flag'], ctx['frame_queue'], ctx['video_thread']
    perception_thread, out, log_file, log_buffer = ctx.get('perception_thread'), ctx['out'], ctx['log_file'], ctx['log_buffer']
    timestamp, fps_list = ctx['timestamp'], ctx['fps_list']
    logger.info("Landing...")
    if log_buffer: log_file.writelines(log_buffer); log_buffer.clear()
    log_file.close()
    exit_flag.set()
    frame_queue.put(None)
    video_thread.join()
    if perception_thread: perception_thread.join()
    out.release()
    try:
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
    except Exception as e:
        logger.error("Landing error: %s", e)
    try:
        html_output = f"analysis/flight_view_{timestamp}.html"
        subprocess.run(["python3", "-m", "analysis.flight_path_viewer", html_output])
        logger.info("Flight path analysis saved to %s", html_output)
    except Exception as e:
        logger.error("Error generating flight path analysis: %s", e)
    try:
        retain_recent_views("analysis", 5)
    except Exception as e:
        logger.error("Error retaining recent views: %s", e)
    if sim_process:
        sim_process.terminate()
        try:
            sim_process.wait(timeout=5)
        except Exception:
            sim_process.kill()
        logger.info("UE4 simulation closed.")
