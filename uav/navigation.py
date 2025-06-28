# uav/navigation.py
"""Navigation utilities for issuing motion commands to an AirSim drone."""
import time
import math
import airsim


class Navigator:
    """Issue high level movement commands and track state."""
    def __init__(self, client):
        self.client = client
        self.braked = False
        self.dodging = False
        self.settling = False
        self.last_movement_time = time.time()
        self.grace_used = False  # add in __init__
        self.grace_period_end_time: float = 0.0
        self.settle_end_time = 0
        self.just_resumed = False
        self.resume_grace_end_time = 0

    def get_state(self):
        """Return the drone position, yaw angle and speed."""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        ori = state.kinematics_estimated.orientation
        yaw = math.degrees(airsim.to_eularian_angles(ori)[2])
        vel = state.kinematics_estimated.linear_velocity
        speed = math.sqrt(vel.x_val ** 2 + vel.y_val ** 2 + vel.z_val ** 2)
        return pos, yaw, speed

    def brake(self):
        """Stop the drone immediately."""
        print("🛑 Braking")
        self.client.moveByVelocityAsync(0, 0, 0, 1)
        self.braked = True
        return "brake"

    def dodge(self, smooth_L, smooth_C, smooth_R, duration: float = 2.0, direction: str = None):
        print(
            f"🔍 Dodge Decision — L: {smooth_L:.1f}, "
            f"C: {smooth_C:.1f}, R: {smooth_R:.1f}"
        )

        # Allow external override of dodge direction (used in hybrid scoring)
        if direction is None:
            left_safe = smooth_L < 0.8 * smooth_C
            right_safe = smooth_R < 0.8 * smooth_C

            if left_safe and not right_safe:
                direction = "left"
            elif right_safe and not left_safe:
                direction = "right"
            elif left_safe and right_safe:
                direction = "left" if smooth_L <= smooth_R else "right"
                print(f"⚠️ Both sides okay — picking {direction}")
            else:
                direction = "left" if smooth_L <= smooth_R else "right"
                print(f"⚠️ No safe sides — forcing {direction}")
        else:
            print(f"📣 Dodge direction forced by caller: {direction}")

        lateral = 1.0 if direction == "right" else -1.0
        strength = 0.5 if max(smooth_L, smooth_R) > 100 else 1.0
        forward_speed = 0.0

        # Stop before dodging
        self.brake()

        print(
            f"🔀 Dodging {direction} (strength {strength:.1f}, "
            f"forward {forward_speed:.1f})"
        )
        self.client.moveByVelocityBodyFrameAsync(
            forward_speed,
            lateral * strength,
            0,
            duration
        )

        self.dodging = True
        self.braked = False
        self.settling = True
        self.settle_end_time = time.time() + 0.1
        self.last_movement_time = time.time()
        return f"dodge_{direction}"

    def resume_forward(self):
        """Resume normal forward velocity."""
        print("✅ Resuming forward motion")
        self.client.moveByVelocityAsync(2, 0, 0, duration=3,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0))
        self.braked = False
        self.dodging = False
        self.just_resumed = True
        self.resume_grace_end_time = time.time() + 0.75  # 0.75 second grace
        self.last_movement_time = time.time()
        return "resume"

    def blind_forward(self):
        """Move forward when no features are detected."""
        print("⚠️ No features — continuing blind forward motion")
        self.client.moveByVelocityAsync(
            2,
            0,
            0,
            duration=2,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0),
        )
        self.last_movement_time = time.time()
        return "blind_forward"

    def nudge(self):
        """Gently push the drone forward when stalled."""
        print("⚠️ Low flow + zero velocity — nudging forward")
        self.client.moveByVelocityAsync(0.5, 0, 0, 1)
        self.last_movement_time = time.time()
        return "nudge"

    def reinforce(self):
        """Reissue the forward command to reinforce motion."""
        print("🔁 Reinforcing forward motion")
        self.client.moveByVelocityAsync(
            2, 0, 0,
            duration=3,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0),
        )
        self.braked = False
        self.dodging = False
        self.last_movement_time = time.time()

        if not self.grace_used:
            self.just_resumed = True
            self.resume_grace_end_time = time.time() + 1.0
            self.grace_used = True
            print("🕒 Grace period started (first movement only)")

        return "resume_reinforce"

    def timeout_recover(self):
        """Move slowly forward after a command timeout."""
        print("⏳ Timeout — forcing recovery motion")
        self.client.moveByVelocityAsync(0.5, 0, 0, 1)
        self.last_movement_time = time.time()
        return "timeout_nudge"
