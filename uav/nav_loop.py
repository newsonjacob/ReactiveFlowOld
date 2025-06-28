# Standard libs
import os
import cv2
import time
import subprocess
import numpy as np
from datetime import datetime
from queue import Queue
from threading import Thread
# AirSim
import airsim
from airsim import ImageRequest, ImageType
# Internal modules
from uav.overlay import draw_overlay
from uav.navigation_rules import compute_thresholds
from uav.video_utils import start_video_writer_thread
from uav.logging_utils import format_log_line
from uav.perception import OpticalFlowTracker
from uav.state_checks import in_grace_period
from uav.scoring import get_weighted_scores, compute_region_stats
from uav.utils import FLOW_STD_MAX

def run_navigation(args, client, sim_process, SETTINGS_PATH):
    # === Configurable Constants ===
    MAX_FLOW_MAG = 10.0
    MAX_VECTOR_COMPONENT = 20.0
    MIN_FEATURES_PER_ZONE = 10
    GRACE_PERIOD_SEC = 1.0
    MAX_SIM_DURATION = 60  # seconds
    GOAL_X = 29  # meters
    GOAL_RADIUS = 1.0

    from uav.interface import exit_flag, start_gui
    from uav.perception import FlowHistory
    from uav.navigation import Navigator
    from uav.utils import get_drone_state, retain_recent_logs, should_flat_wall_dodge
    from analysis.utils import retain_recent_views

    # GUI parameter and status holders
    param_refs = {
            'L': [0.0],
            'C': [0.0],
            'R': [0.0],
            'state': [''],
            'reset_flag': [False]
        }
    start_gui(param_refs)

    print("Available vehicles:", client.listVehicles())
    client.enableApiControl(True)
    client.armDisarm(True)

    # After takeoff
    client.takeoffAsync().join()
    client.moveToPositionAsync(0, 0, -2, 2).join()

    # Tune feature detection to pick up more corners even on smooth surfaces
    feature_params = dict(maxCorners=150, qualityLevel=0.05, minDistance=5, blockSize=5)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    tracker = OpticalFlowTracker(lk_params, feature_params)
    flow_history = FlowHistory()
    navigator = Navigator(client)
    from collections import deque
    state_history = deque(maxlen=3)
    pos_history = deque(maxlen=3)

    frame_count = 0
    start_time = time.time()
    GOAL_X = args.goal_x
    MAX_SIM_DURATION = args.max_duration
    print(f"Config:\n  Goal X: {GOAL_X}m\n  Max Duration: {MAX_SIM_DURATION}s")
    GOAL_RADIUS = 1.0  # meters
    MIN_PROBE_FEATURES = 5
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs("flow_logs", exist_ok=True)
    log_file = open(f"flow_logs/full_log_{timestamp}.csv", 'w')
    log_file.write(
        "frame,time,features,flow_left,flow_center,flow_right,"
        "flow_std,pos_x,pos_y,pos_z,yaw,speed,state,collided,obstacle,side_safe,"
        "brake_thres,dodge_thres,probe_req,fps,simgetimage_s,decode_s,processing_s,loop_s\n"
    )
    retain_recent_logs("flow_logs")

    # Video writer setup
    try:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') # type: ignore
    except AttributeError:
        fourcc = cv2.FOURCC(*'MJPG') # type: ignore
    # Capture at 720p for better optical flow tracking
    out = cv2.VideoWriter('flow_output.avi', fourcc, 8.0, (1280, 720))

    # Offload video writing to a background thread
    frame_queue: Queue = Queue(maxsize=20)
    
    video_thread = start_video_writer_thread(frame_queue, out, exit_flag)

    # Perception thread for image capture and optical flow
    perception_queue: Queue = Queue(maxsize=1)
    last_vis_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    def perception_worker() -> None:
        nonlocal last_vis_img
        # Use a dedicated RPC client to avoid cross-thread issues
        local_client = airsim.MultirotorClient()
        local_client.confirmConnection()
        request = [ImageRequest("oakd_camera", ImageType.Scene, False, True)]
        while not exit_flag.is_set():
            t0 = time.time()
            responses = local_client.simGetImages(request, vehicle_name="UAV")
            t_fetch_end = time.time()
            response = responses[0]
            if (
                response.width == 0
                or response.height == 0
                or len(response.image_data_uint8) == 0
            ):
                data = (
                    last_vis_img,
                    np.array([]),
                    np.array([]),
                    0.0,
                    t_fetch_end - t0,
                    0.0,
                    0.0,
                )
            else:
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8).copy()
                img = cv2.imdecode(img1d, cv2.IMREAD_COLOR)
                t_decode_end = time.time()
                if img is None:
                    continue
                img = cv2.resize(img, (1280, 720))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                vis_img = img.copy()
                last_vis_img = vis_img
                if tracker.prev_gray is None:
                    tracker.initialize(gray)
                    data = (
                        vis_img,
                        np.array([]),
                        np.array([]),
                        0.0,
                        t_fetch_end - t0,
                        t_decode_end - t_fetch_end,
                        0.0,
                    )
                else:
                    t_proc_start = time.time()
                    good_old, flow_vectors, flow_std = tracker.process_frame(gray, t0)
                    processing_s = time.time() - t_proc_start
                    data = (
                        vis_img,
                        good_old,
                        flow_vectors,
                        flow_std,
                        t_fetch_end - t0,
                        t_decode_end - t_fetch_end,
                        processing_s,
                    )

            try:
                perception_queue.put(data, block=False)
            except Exception:
                # Drop frame if queue already contains an item
                pass

    perception_thread = Thread(target=perception_worker, daemon=True)
    perception_thread.start()

    # Buffer log lines to throttle disk writes
    log_buffer = []
    LOG_INTERVAL = 5  # flush every 5 frames


    target_fps = 20
    frame_duration = 1.0 / target_fps

    fps_list = []
    img = None  # Add this before your main loop
    
    grace_logged = False
    startup_grace_over = False

    try:
        loop_start = time.time()
        while not exit_flag.is_set():
            frame_count += 1
            time_now = time.time()

            # === Grace period (SKIPS perception AND nav logic entirely) ===
            if not startup_grace_over:
                if time_now - start_time < GRACE_PERIOD_SEC:
                    param_refs['state'][0] = "startup_grace"
                    if not grace_logged:
                        print("⏳ Startup grace period active — waiting to start perception and nav")
                        grace_logged = True
                    time.sleep(0.05)
                    continue
                else:
                    startup_grace_over = True
                    print("🚀 Startup grace period complete — beginning full nav logic")

            # === Retrieve perception results AFTER grace ===
            try:
                (
                    vis_img,
                    good_old,
                    flow_vectors,
                    flow_std,
                    simgetimage_s,
                    decode_s,
                    processing_s,
                ) = perception_queue.get(timeout=1.0)
            except Exception:
                continue


            prev_state = param_refs['state'][0]
            # Handle brief settle period after dodge
            if navigator.settling and time_now >= navigator.settle_end_time:
                print("✅ Settle period over — resuming evaluation")
                navigator.settling = False

            if time_now - start_time >= MAX_SIM_DURATION:
                print("⏱️ Time limit reached — landing and stopping.")
                break

            pos_goal, _, _ = get_drone_state(client)
            if pos_goal.x_val >= GOAL_X - GOAL_RADIUS:
                print("\U0001F3C1 Goal reached — landing.")
                break

            # --- Retrieve perception results ---
            try:
                (
                    vis_img,
                    good_old,
                    flow_vectors,
                    flow_std,
                    simgetimage_s,
                    decode_s,
                    processing_s,
                ) = perception_queue.get(timeout=1.0)
            except Exception:
                continue

            gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)

            if frame_count == 1 and len(good_old) == 0:
                frame_queue.put(vis_img)
                continue

            if args.manual_nudge and frame_count == 5:
                print("🔧 Manual nudge forward for test")
                client.moveByVelocityAsync(2, 0, 0, 2)

            magnitudes = np.linalg.norm(flow_vectors, axis=1)

            # Clamp extreme flow magnitudes (e.g., from sensor noise or motion blur)
            num_clamped = np.sum(magnitudes > MAX_FLOW_MAG)
            if num_clamped > 0:
                print(f"⚠️ Clamped {num_clamped} large flow magnitudes to {MAX_FLOW_MAG}")
            magnitudes = np.clip(magnitudes, 0, MAX_FLOW_MAG)

            good_old = good_old.reshape(-1, 2)  # Ensure proper shape

            (
                left_mag, center_mag, right_mag,
                probe_mag, probe_count,
                left_count, center_count, right_count
            ) = compute_region_stats(magnitudes, good_old, gray.shape[1])

            flow_history.update(left_mag, center_mag, right_mag)
            smooth_L, smooth_C, smooth_R = flow_history.average()
            param_refs['L'][0] = smooth_L
            param_refs['C'][0] = smooth_C
            param_refs['R'][0] = smooth_R

            # Grace indicator
            if navigator.just_resumed and time_now < navigator.resume_grace_end_time:
                cv2.putText(vis_img, "GRACE", (1100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            # Draw flow arrows for visualization
            for i, (p1, vec) in enumerate(zip(good_old, flow_vectors)):
                if i > 50:
                    break
                x1, y1 = int(p1[0]), int(p1[1])
                vec = np.ravel(vec)  # Ensure vec is 1D
                if vec.shape[0] >= 2:
                    dx = float(np.clip(vec[0], -MAX_VECTOR_COMPONENT, MAX_VECTOR_COMPONENT))
                    dy = float(np.clip(vec[1], -MAX_VECTOR_COMPONENT, MAX_VECTOR_COMPONENT))
                else:
                    dx, dy = 0.0, 0.0
                x2, y2 = int(x1 + dx), int(y1 + dy)
                cv2.arrowedLine(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)

            in_grace = navigator.just_resumed and time_now < navigator.resume_grace_end_time

            # === Navigation logic ===
            state_str = "none"
            brake_thres = 0.0
            dodge_thres = 0.0
            probe_req = 0.0
            side_safe = False

            # Minimum features per zone to consider valid

            valid_L = left_count >= MIN_FEATURES_PER_ZONE
            valid_C = center_count >= MIN_FEATURES_PER_ZONE
            valid_R = right_count >= MIN_FEATURES_PER_ZONE

            # === Weighted scoring (magnitude × count) ===
            left_score, center_score, right_score = get_weighted_scores(
                smooth_L, smooth_C, smooth_R,
                left_count, center_count, right_count
            )

            print(f"📊 Weighted Scores — L: {left_score:.2f}, C: {center_score:.2f}, R: {right_score:.2f}")


            # Skip obstacle logic during grace period after resuming
            if in_grace_period(time_now, navigator):
                    param_refs['state'][0] = "🕒 grace"
                    obstacle_detected = 0

                    # Always store frame first, even if queue is full
                    try:
                        frame_queue.get_nowait()
                    except Exception:
                        pass
                    frame_queue.put(vis_img)

                    continue  # Skip nav logic, but perception + video continues
            else:
                    navigator.just_resumed = False
                    print("🟢 Grace period ended — navigation activated")

            if len(good_old) < 5: #If the number of "good" feature points tracked by the optical flow algorithm is less than 5, then...
                if smooth_L > 1.5 and smooth_R > 1.5 and smooth_C < 0.2:
                    state_str = navigator.brake()
                else:
                    state_str = navigator.blind_forward()
            else:
                pos, yaw, speed = get_drone_state(client)

                # Adaptive thresholds tuned for quicker reactions
                brake_thres, dodge_thres = compute_thresholds(speed)

                center_high = valid_C and (smooth_C > dodge_thres or smooth_C > 2 * min(smooth_L, smooth_R))
                side_diff = abs(smooth_L - smooth_R)
                side_safe = (
                    valid_L and valid_R and
                    abs(smooth_L - smooth_R) > 0.3 * smooth_C and
                    (smooth_L < 100 or smooth_R < 100)
                )

                probe_reliable = probe_count > MIN_PROBE_FEATURES and probe_mag > 0.05
                in_grace_period_flag = in_grace_period(time_now, navigator)

                # === Priority 1: Severe brake override (ignores grace period)
                if smooth_C > (brake_thres * 1.5):
                    state_str = navigator.brake()
                    navigator.grace_period_end_time = time_now + 1.5

                elif not in_grace_period:
                    # === Brake Logic
                    if smooth_C > brake_thres:
                        state_str = navigator.brake()
                        navigator.grace_period_end_time = time_now + 1.5
                    # elif not probe_reliable and center_mag > 0.7:
                    #     print("⚠️ No probe flow but high center flow — possible wall ahead")
                    #     state_str = navigator.brake()
                    #     navigator.grace_period_end_time = time_now + 2.5

                    # === Dodge Logic
                    elif center_high and side_safe:
                        print(f"📊 Hybrid Scores — L: {left_score:.2f}, C: {center_score:.2f}, R: {right_score:.2f}")
                        if left_score < right_score:
                            state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction='left')
                        else:
                            state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction='right')
                        navigator.grace_period_end_time = time_now + 1.5

                    elif probe_mag < 0.5 and center_mag > 0.7:
                        if should_flat_wall_dodge(center_mag, probe_mag, probe_count, MIN_PROBE_FEATURES, flow_std, FLOW_STD_MAX):
                            print("🟥 Flat wall detected — attempting fallback dodge")
                            state_str = navigator.dodge(smooth_L, smooth_C, smooth_R)
                            navigator.grace_period_end_time = time_now + 1.5
                        else:
                            print("🔬 Insufficient probe features — ignoring fallback")

                # === Recovery / Maintenance States (always allowed)
                if state_str == "none":
                    if (
                        navigator.dodging
                        and smooth_C < dodge_thres * 0.9
                        and time_now >= navigator.grace_period_end_time
                        and not navigator.settling
                    ):
                        print(f"🔄 Dodge ended — resuming forward at frame {frame_count}")
                        state_str = navigator.resume_forward()

                    elif (
                        navigator.braked
                        and smooth_C < brake_thres * 0.8
                        and smooth_L < brake_thres * 0.8
                        and smooth_R < brake_thres * 0.8
                        and time_now >= navigator.grace_period_end_time
                    ):
                        print(f"🟢 Brake released — resuming forward at frame {frame_count}")
                        state_str = navigator.resume_forward()
                    elif not navigator.braked and not navigator.dodging and time_now - navigator.last_movement_time > 2:
                        state_str = navigator.reinforce()
                    elif (navigator.braked or navigator.dodging) and speed < 0.2 and smooth_C < 5 and smooth_L < 5 and smooth_R < 5:
                        state_str = navigator.nudge()
                    elif time_now - navigator.last_movement_time > 4:
                        state_str = navigator.timeout_recover()

            if (
                state_str == "none"
                and navigator.dodging
                and time_now < navigator.grace_period_end_time
                and isinstance(prev_state, str)
                and prev_state.startswith("dodge")
            ):
                state_str = prev_state
            param_refs['state'][0] = state_str
            obstacle_detected = int('dodge' in state_str or state_str == 'brake')

            # === Detect repeated dodges with minimal progress ===
            pos_hist, _, _ = get_drone_state(client)
            state_history.append(state_str)
            pos_history.append((pos_hist.x_val, pos_hist.y_val))
            if len(state_history) == state_history.maxlen:
                if all(s == state_history[-1] for s in state_history) and state_history[-1].startswith("dodge"):
                    dx = pos_history[-1][0] - pos_history[0][0]
                    dy = pos_history[-1][1] - pos_history[0][1]
                    if abs(dx) < 0.5 and abs(dy) < 1.0:
                        print("♻️ Repeated dodges detected — extending dodge")
                        state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, duration=3.0)
                        state_history[-1] = state_str
                        param_refs['state'][0] = state_str

            # === Reset logic from GUI ===
            if param_refs['reset_flag'][0]:
                print("🔄 Resetting simulation...")
                try:
                    client.landAsync().join()
                    client.reset()
                    client.enableApiControl(True)
                    client.armDisarm(True)
                    client.takeoffAsync().join()
                    client.moveToPositionAsync(0, 0, -2, 2).join()
                except Exception as e:
                    print("Reset error:", e)

                flow_history = FlowHistory()
                navigator = Navigator(client)
                frame_count = 0
                param_refs['reset_flag'][0] = False

                # === Reset log file ===
                if log_buffer:
                    log_file.writelines(log_buffer)
                    log_buffer.clear()
                log_file.close()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file = open(f"flow_logs/full_log_{timestamp}.csv", 'w')
                log_file.write(
                    "frame,time,features,flow_left,flow_center,flow_right,"
                    "flow_std,pos_x,pos_y,pos_z,yaw,speed,state,collided,obstacle,side_safe,"
                    "brake_thres,dodge_thres,probe_req,fps,simgetimage_s,decode_s,processing_s,loop_s\n"
                )
                retain_recent_logs("flow_logs")

                # === Reset video writer ===
                frame_queue.put(None)
                video_thread.join()
                out.release()
                out = cv2.VideoWriter('flow_output.avi', fourcc, 8.0, (1280, 720))
                video_thread = Thread(target=video_worker, daemon=True)
                video_thread.start()
                continue

            # Queue frame for async video writing
            try:
                frame_queue.put_nowait(vis_img)
            except Exception:
                pass

            # Throttle loop to target FPS
            elapsed = time.time() - loop_start
            if elapsed < frame_duration:
                time.sleep(frame_duration - elapsed)
            loop_elapsed = time.time() - loop_start
            actual_fps = 1 / max(loop_elapsed, 1e-6)
            loop_start = time.time()

            fps_list.append(actual_fps)

            pos, yaw, speed = get_drone_state(client)
            collision = client.simGetCollisionInfo()
            collided = int(getattr(collision, "has_collided", False))
            vis_img = draw_overlay(
                vis_img, 
                frame_count, 
                speed, 
                param_refs['state'][0], 
                time_now - start_time, 
                smooth_L, smooth_C, smooth_R, 
                left_count, center_count, right_count, 
                good_old,
                flow_vectors,
                in_grace=in_grace
            )

            log_line = format_log_line(
                frame_count, time_now, good_old, smooth_L, smooth_C, smooth_R, flow_std,
                pos, yaw, speed, state_str, collided, obstacle_detected, side_safe,
                brake_thres, dodge_thres, probe_req, actual_fps,
                simgetimage_s, decode_s, processing_s, loop_elapsed
            )
            log_buffer.append(log_line)

            if frame_count % LOG_INTERVAL == 0:
                log_file.writelines(log_buffer)
                log_buffer.clear()

            print(f"Actual FPS: {actual_fps:.2f}")
            print(f"Features detected: {len(good_old)}")

    except KeyboardInterrupt:
        print("Interrupted.")

    finally:
        print("Landing...")
        if log_buffer:
            log_file.writelines(log_buffer)
            log_buffer.clear()
        log_file.close()
        exit_flag.set()
        frame_queue.put(None)
        video_thread.join()
        perception_thread.join()
        out.release()
        try:
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
        except Exception as e:
            print("Landing error:", e)
        # === Auto-generate 3D flight visualisation ===
        try:
            html_output = f"analysis/flight_view_{timestamp}.html"
            subprocess.run([
                "python", "analysis/visualize_flight.py",
                "--log", f"flow_logs/full_log_{timestamp}.csv",
                "--obstacles", "analysis/obstacles.json",
                "--output", html_output,
                "--scale", "1.0"
            ])
            print(f"✅ 3D visualisation saved to {html_output}")
            retain_recent_views("analysis")
            retain_recent_logs("flow_logs")
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")

        if sim_process:
            sim_process.terminate()
            try:
                sim_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("UE4 did not terminate gracefully, killing process...")
                sim_process.kill()
            print("UE4 simulation closed.")

        # Re-encode video at median FPS using OpenCV
        import statistics
        if len(fps_list) > 0:
            median_fps = statistics.median(fps_list)
            print(f"Median FPS: {median_fps:.2f}")

            input_video = 'flow_output.avi'
            output_video = 'flow_output_fixed.avi'

            cap = cv2.VideoCapture(input_video)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out_fixed = cv2.VideoWriter(output_video, fourcc, median_fps, (1280, 720))

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out_fixed.write(frame)
                frame_count += 1

            cap.release()
            out_fixed.release()
            print(f"Re-encoded {frame_count} frames at {median_fps:.2f} FPS to {output_video}")

