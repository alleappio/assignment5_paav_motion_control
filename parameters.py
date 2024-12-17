class SimulationParameters:
    dt = 0.001                  # Time step (s)
    ax = 0.0                    # Constant longitudinal acceleration (m/s^2)
    steer = 0.0                 # Constant steering angle (rad)
    sim_time = 200.0             # Simulation duration in seconds
    steps = int(sim_time / dt)  # Simulation steps (30 seconds)
    target_speed = 20.0
    controller = 'stanley'


class VehicleParameters:
    lf = 1.156                  # Distance from COG to front axle (m)
    lr = 1.42                   # Distance from COG to rear axle (m)
    wheelbase = lf + lr         # Wheelbase of vehicle
    mass = 1200                 # Vehicle mass (kg)
    Iz = 1792                   # Yaw moment of inertia (kg*m^2)
    max_steer = 3.14            # Maximum steering angle in radians


class PIDParameters:
    kp = 2                      # Proportional gain
    ki = 1.2                    # Integrative gain
    kd = 0.2                    # Derivative gain
    output_limits = (-2, 2)     # Saturation limits


class PurepursuitParameters:
    k_v = 1                     # Speed proportional gain for Pure Pursuit
    k_c = 0.1                   # Curve proportional gain for Pure Pursuit
    limit_theta = 0.1           # Minimum heading angle for adding curve proportional gain
    look_ahead = 1.0            # Minimum look-ahead distance for Pure Pursuit


class StanleyParameters:
    k_stanley = 0.8           # Gain for cross-track error for Stanley
    k_he = 0.8                # Gain for heading error
    k_ctc = 1.5               # Gain for cross-trac correction 
