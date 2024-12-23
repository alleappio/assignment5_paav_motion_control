class SimulationParameters:
    dt = 0.001                  # Time step (s)
    ax = 0.0                    # Constant longitudinal acceleration (m/s^2)
    steer = 0.0                 # Constant steering angle (rad)
    sim_time = 65.0             # Simulation duration in seconds
    steps = int(sim_time / dt)  # Simulation steps (30 seconds)
    target_speed = 20.0
    controller = 'mpc'
    figures_path = 'figures/general'
    vehicle_model = [
        ("rk4", "kinematic"),
    ]


class VehicleParameters:
    lf = 1.156                # Distance from COG to front axle (m)
    lr = 1.42                 # Distance from COG to rear axle (m)
    wheelbase = lf + lr       # Wheelbase of vehicle
    mass = 1200               # Vehicle mass (kg)
    Iz = 1792                 # Yaw moment of inertia (kg*m^2)
    max_steer = 3.14          # Maximum steering angle in radians
    min_steer = -3.14         # Maximum steering angle in radians


class PIDParameters:
    kp = 1.7                  # Proportional gain
    ki = 0.7                  # Integrative gain
    kd = 0.1                  # Derivative gain
    output_limits = (-2, 2)   # Saturation limits


class PurepursuitParameters:
    k_v = 0.13                # Speed proportional gain for Pure Pursuit
    k_c = 0.05                # Curve proportional gain for Pure Pursuit
    limit_curvature = 0.05    # Minimum heading angle for adding curve proportional gain
    look_ahead = 1.0          # Minimum look-ahead distance for Pure Pursuit


class StanleyParameters:
    k_stanley = 2.9           # Gain for cross-track error for Stanley
    k_he = 1.1                # Gain for heading error
    k_ctc = 2.9               # Gain for cross-trac correction 


class MpcParameters:
    gain_mult = 1.0  
    k_x = 100.0
    k_y = 100.0
    k_theta = 10.0
    k_j = 10000.0
    T =  1.5                    # Horizon length in seconds
    dt = 0.15        # Horizon timesteps
    N = int(T/dt)             # Horizon total points
