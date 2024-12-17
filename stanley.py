import math
import numpy as np

class StanleyController:
    def __init__(self, k, l_f, max_steer, k_he, k_ctc):
        """
        Initialize the Stanley Controller.
        
        Args:
            k (float): Gain for cross-track error correction.
            l_f (float): Distance from the CoG to the front axle.
            max_steer (float): Maximum steering angle in radians.
        """
        self.k = k
        self.l_f = l_f
        self.max_steer = max_steer
        self.k_he = k_he
        self.k_ctc = k_ctc

    def compute_steering_angle(self, current_pose, target_pose, speed):
        """
        Compute the steering angle using the Stanley Controller algorithm.
        
        Args:
            current_pose (tuple): Current pose of the vehicle (x, y, yaw).
            target_pose (tuple): Target pose on the path (x, y, yaw).
            speed (float): Current vehicle speed (m/s).
        
        Returns:
            float: Steering angle (delta) in radians.
        """
        # Extract current and target states
        cog_x, cog_y, current_yaw = current_pose
        target_x, target_y, target_yaw = target_pose

        # Adjust CoG position to the front axle position
        front_x = cog_x + self.l_f * math.cos(current_yaw)
        front_y = cog_y + self.l_f * math.sin(current_yaw)

        # Compute heading error (yaw difference)
        heading_error = target_yaw - current_yaw
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        # Compute cross-track error
        dx = front_x - target_x
        dy = front_y - target_y
        cross_track_error = (dx * math.sin(target_yaw) - dy * math.cos(target_yaw))

        # Compute cross-track correction
        effective_speed = max(speed, 1.0)  # Prevent division by very small speeds
        cross_track_correction = math.atan2(self.k * cross_track_error, effective_speed)

        # Total steering angle
        # TO-DO: Adjust the gains in order to improve the controller's tracking
        # right now they are both 1.0
        delta = self.k_he * heading_error + self.k_ctc * cross_track_correction


        # Saturate the steering angle within the maximum limits
        delta = max(-self.max_steer, min(delta, self.max_steer))
        
        return delta
