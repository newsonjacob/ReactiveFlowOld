# uav/navigation.py
"""Navigation utilities for issuing motion commands to an AirSim drone."""
import time
import math
import logging
import airsim

logger = logging.getLogger(__name__)


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
        """Stop the drone immediately with a reverse velocity proportional to current speed."""
        try:
            # Get current velocity
            state = self.client.getMultirotorState()
            vel = state.kinematics_estimated.linear_velocity
            speed = vel.x_val
            # Apply reverse velocity proportional to current forward speed (clamp if needed)
            reverse_speed = -min(speed, 3.0)  # Limit max reverse speed for safety
            self.client.moveByVelocityAsync(reverse_speed, 0, 0, 0.5)
        except AttributeError:
            self.client.moveByVelocityAsync(0, 0, 0, 0.5)
        self.braked = True
        return "brake"

    def dodge(self, smooth_L, smooth_C, smooth_R, duration: float = 2.0, direction: str = None): # type: ignore
        lateral = 1.0 if direction == "right" else -1.0
        # strength = 0.75 if max(smooth_L, smooth_R) > 100 else 1.0
        strength = 1.0
        forward_speed = 0.0

        # Stop before dodging
        self.brake()
        time.sleep(0.5)  # Allow time for braking to take effect
        self.client.moveByVelocityBodyFrameAsync(forward_speed,lateral * strength,0,duration)

        self.dodging = True
        self.braked = False
        self.last_movement_time = time.time()
        self.dodge_direction = direction
        self.dodge_strength = strength
        return f"dodge_{direction}"

    def maintain_dodge(self):
        """Maintain the dodge movement."""
        if self.dodging:
            lateral = 1.0 if self.dodge_direction == "right" else -1.0
            self.client.moveByVelocityBodyFrameAsync(0.0, lateral * self.dodge_strength, 0, duration=0.3)

    def resume_forward(self):
        """Resume normal forward velocity."""

        # Stop before resuming forward motion
        self.client.moveByVelocityAsync(0, 0, 0, 0)
        time.sleep(0.2)  # Allow time for braking to take effect
        
        state = self.client.getMultirotorState()
        z = state.kinematics_estimated.position.z_val  # NED: z is negative up
        self.client.moveByVelocityZAsync(2, 0, 0, duration=3,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0))
        self.braked = False
        self.dodging = False
        self.just_resumed = True
        self.resume_grace_end_time = time.time() + 0 # 0 second grace
        self.last_movement_time = time.time()
        return "resume"

    def blind_forward(self):
        """Move forward when no features are detected."""
        logger.warning(
            "\u26A0\uFE0F No features — continuing blind forward motion")
        state = self.client.getMultirotorState()
        z = state.kinematics_estimated.position.z_val  # NED: z is negative up
        self.client.moveByVelocityZAsync(2,0,0,duration=2,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0),)
        self.last_movement_time = time.time()
        if not self.grace_used:
            self.just_resumed = True
            self.resume_grace_end_time = time.time() + 1.0
            self.grace_used = True
        return "blind_forward"

    def nudge_forward(self):
        """Gently push the drone forward when stalled."""
        logger.warning(
            "\u26A0\uFE0F Low flow + zero velocity — nudging forward"
        )
        state = self.client.getMultirotorState()
        z = state.kinematics_estimated.position.z_val  # NED: z is negative up
        self.client.moveByVelocityZAsync(0.5, 0, 0, 1)
        self.last_movement_time = time.time()
        return "nudge"

    def reinforce(self):
        """Reissue the forward command to reinforce motion."""
        logger.info("\U0001F501 Reinforcing forward motion")
        state = self.client.getMultirotorState()
        z = state.kinematics_estimated.position.z_val  # NED: z is negative up
        self.client.moveByVelocityZAsync(
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
            logger.info("\U0001F552 Grace period started (first movement only)")

        return "resume_reinforce"

    def timeout_recover(self):
        """Move slowly forward after a command timeout."""
        logger.warning("\u23F3 Timeout — forcing recovery motion")
        self.client.moveByVelocityAsync(0.5, 0, 0, 1)
        self.last_movement_time = time.time()
        return "timeout_nudge"
