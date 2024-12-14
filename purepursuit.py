import math

class PurePursuitController:
    def __init__(self, wheelbase, max_steer):
        """
        Initialize the Pure Pursuit controller.
        
        Args:
            wheelbase (float): Distance between the front and rear axles (L).
            max_steer (float): Maximum steering angle in radians.
        """
        self.wheelbase = wheelbase
        self.max_steer = max_steer

    def compute_steering_angle(self, target, actual_heading, Lf):
        """
        Compute the steering angle using the Pure Pursuit algorithm.
        
        Args:
            target (tuple): Coordinates of the target point (x, y) in the vehicle frame.
            actual_heading (float): Current heading of the vehicle in radians.
        
        Returns:
            float: Steering angle (delta) in radians.
        """
        # Calculate the heading error (alpha)
        alpha = math.atan2(target[1], target[0]) - actual_heading
        
        # Compute the steering angle (delta)
        delta = 0.0#TO-DO: complete as in the PurePursuit slides
        
        # Saturate the steering angle within the maximum limits
        delta = max(-self.max_steer, min(delta, self.max_steer))
        
        return delta
