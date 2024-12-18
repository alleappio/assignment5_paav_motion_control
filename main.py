import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simulation import Simulation
import pid
import purepursuit
import stanley
import mpc
import cubic_spline_planner
import math
from parameters import SimulationParameters as sim_params
from parameters import VehicleParameters as vehicle_params
from parameters import PIDParameters as PID_params
from parameters import PurepursuitParameters as PP_params
from parameters import StanleyParameters as stanley_params
from parameters import MpcParameters as MPC_params

# Create instance of PID for Longitudinal Control
long_control_pid = pid.PIDController(kp=PID_params.kp, ki=PID_params.ki, kd=PID_params.kd, output_limits=PID_params.output_limits)

# Create instance of PurePursuit, Stanley and MPC for Lateral Control
pp_controller = purepursuit.PurePursuitController(vehicle_params.wheelbase, vehicle_params.max_steer)
stanley_controller = stanley.StanleyController(stanley_params.k_stanley, vehicle_params.lf, vehicle_params.max_steer, stanley_params.k_he, stanley_params.k_ctc)
mpc_controller = mpc.MPC(MPC_params.T, MPC_params.dt, MPC_params.N, vehicle_params.max_steer, vehicle_params.min_steer)

def load_path(file_path):
    file = open(file_path, "r")
    
    xs = []
    ys = []
    
    cur_point = [0.0,0.0]
    prev_point = [0.0,0.0]

    total_path = 0.0

    while(file.readline()):
        line = file.readline()
        cur_point[0] = float(line.split(",")[0])
        cur_point[1] = float(line.split(",")[1])
        total_path += math.dist(cur_point, prev_point)
        xs.append( cur_point[0] )
        ys.append( cur_point[1] )
        prev_point[0] = cur_point[0]
        prev_point[1] = cur_point[1]
    print(total_path)
    return xs, ys, total_path

# Load path and create a spline
xs, ys, total_path = load_path("oval_trj.txt")
path_spline = cubic_spline_planner.Spline2D(xs, ys)

def point_transform(trg, pose, yaw):

    local_trg = [trg[0] - pose[0], trg[1] - pose[1]]

    return local_trg

def plot_comparison(results, labels, title, xlabel, ylabel):
    """ Plot comparison of results for a specific state variable. """
    plt.figure(figsize=(10, 6))
    for i, result in enumerate(results):
        plt.plot(result, label=labels[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_trajectory(x_vals, y_vals, labels, path_spline):
    """ Plot 2D trajectory (x vs y) for all simulation configurations and path_spline trajectory. """
    plt.figure(figsize=(10, 6))
    
    # Plot the simulation trajectories
    for i in range(len(x_vals)):
        plt.plot(x_vals[i], y_vals[i], label=labels[i])
    
    # Plot the path_spline trajectory
    spline_x = [path_spline.calc_position(s)[0] for s in np.linspace(0, path_spline.s[-1], 1000)]
    spline_y = [path_spline.calc_position(s)[1] for s in np.linspace(0, path_spline.s[-1], 1000)]
    plt.plot(spline_x, spline_y, label="Path Spline", linestyle="--", color="red")
    
    # Customize plot
    plt.title("2D Trajectory Comparison")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def run_simulation(ax, steer, dt, integrator, model, steps=500):
    """ Run a simulation with the given parameters and return all states. """
    total_path_done = 0.0
    cur_position = (0.0, 0.0)
    prev_position = (0.0, 0.0)
    lap_counter = 0
    # Initialize the simulation
    sim = Simulation(vehicle_params.lf, vehicle_params.lr, vehicle_params.mass, vehicle_params.Iz, dt, integrator=integrator, model=model)

    # Storage for state variables and slip angles
    x_vals, y_vals, theta_vals, vx_vals, vy_vals, r_vals = [], [], [], [], [], []
    alpha_f_vals, alpha_r_vals = [], []  # Slip angles
    lat_error_vals, vel_error_vals = [], []
    
    # casadi_model() #for MPC... TO-DO
    mpc_controller.casadi_model()
    prev_time=0
    for step in range(steps):
        time = step*dt
        # Print time
        if(int(time)!=int(prev_time)):
            print(f"time: {int(time)} seconds")
            prev_time = time

        cur_position = (sim.x, sim.y)
        total_path_done += math.dist(cur_position, prev_position)
    
        if(abs(total_path_done - total_path) < 0.01):
            total_path_done = 0
            lap_counter+=1
            print(f"done {lap_counter} lap at time {time}")
        # Calculate ax to track speed
        ax = long_control_pid.compute(sim_params.target_speed, sim.vx, dt) # Exercise 1
        steer = 0

        # Update actual frenet-frame position in the spline
        # aka longitudinal position and actual lateral error
        actual_position = sim.x, sim.y
        actual_pose = sim.x, sim.y, sim.theta
        path_spline.update_current_s(actual_position)

        # get actual position projected on the path/spline
        position_projected = path_spline.calc_position(path_spline.cur_s)
        prj = [ position_projected[0], position_projected[1] ]
        local_error = point_transform(prj, actual_position, sim.theta)

        if(abs(local_error[1]) > 1.0):
            print("Lateral error is higher than 1.0... ending the simulation")
            print("Lateral error: ", local_error[1])
            break

        # Calculate lookahead including:
        # - base lookahead
        # - speed
        # - curvature
        Lf = PP_params.k_v * sim.vx + PP_params.look_ahead 
        if(abs(path_spline.calc_curvature(path_spline.cur_s)) > PP_params.limit_curvature):
            Lf += PP_params.k_c / abs(path_spline.calc_curvature(path_spline.cur_s))

        s_pos = path_spline.cur_s + Lf

        trg = path_spline.calc_position(s_pos)
        trg = [ trg[0], trg[1] ]
        pp_position = actual_position
        # Adjust CoG position to the rear axle position for PP
        pp_position = actual_position[0] + vehicle_params.lr * math.cos(sim.theta), actual_position[1] + vehicle_params.lr * math.sin(sim.theta)
        loc_trg = point_transform(trg, pp_position, sim.theta)

        # Calculate steer to track path
        
        ####### Pure Pursuit 1
        # Compute the look-ahead distance
        if(sim_params.controller == 'purepursuit'):
            steer = pp_controller.compute_steering_angle(loc_trg, sim.theta, Lf)
        
        ###### Stanley 
        #TO-DO: Move actual position (CoG) to the front axle for stanley
        if(sim_params.controller == 'stanley'):
            stanley_position = actual_position[0] + vehicle_params.lf * math.cos(sim.theta), actual_position[1] + vehicle_params.lf * math.sin(sim.theta)
            actual_pose = stanley_position[0], stanley_position[1], sim.theta
            stanley_target = position_projected[0], position_projected[1], path_spline.calc_yaw(path_spline.cur_s)
            steer = stanley_controller.compute_steering_angle(actual_pose, stanley_target, sim.vx)

        ###### MPC
        if(sim_params.controller == 'mpc'):
            # get future horizon targets pose
            targets = [ ]
            s_pos = path_spline.cur_s
            for i in range(mpc_controller.N):
                step_increment = (sim.vx)*dt
                trg = point_transform(path_spline.calc_position(s_pos), actual_pose, sim.theta)
                trg = [ trg[0], trg[1] ]
                targets.append(trg)
                s_pos += step_increment

            steer = mpc_controller.opt_step(targets, sim)

        prev_position = cur_position
        # Make one step simulation via model integration
        sim.integrate(ax, float(steer))
        
        # Append each state to corresponding list
        x_vals.append(sim.x)
        y_vals.append(sim.y)
        theta_vals.append(sim.theta)
        vx_vals.append(sim.vx)
        vy_vals.append(sim.vy)
        r_vals.append(sim.r)

        # Calculate slip angles for front and rear tires
        alpha_f = steer - np.arctan((sim.vy + sim.l_f * sim.r) / max(0.5, sim.vx))  # Front tire slip angle
        alpha_r = -(np.arctan(sim.vy - sim.l_r * sim.r) / max(0.5, sim.vx))         # Rear tire slip angle
        
        alpha_f_vals.append(alpha_f)
        alpha_r_vals.append(alpha_r)

        #vel_error = (abs(sim.vx-sim_params.target_speed)/sim_params.target_speed)*100
        vel_error = long_control_pid.previous_error
        prj = [ position_projected[0], position_projected[1] ]
        local_error = point_transform(prj, actual_position, sim.theta)
        lat_error = math.sin(abs(local_error[1]))

        vel_error_vals.append(vel_error)
        lat_error_vals.append(lat_error)
    return x_vals, y_vals, theta_vals, vx_vals, vy_vals, r_vals, alpha_f_vals, alpha_r_vals, vel_error_vals, lat_error_vals

def main():

    # List of configurations
    configs = [
        ("rk4", "nonlinear"),
    ]

    # Run each simulation and store the results
    all_results = []
    actual_state = []
    labels = []
    for integrator, model in configs:
        actual_state = run_simulation(sim_params.ax, sim_params.steer, sim_params.dt, integrator, model, sim_params.steps)
        all_results.append(actual_state)
        labels.append(f"{integrator.capitalize()} - {model.capitalize()}")

    # Separate each state for plotting
    x_results = [result[0] for result in all_results]
    y_results = [result[1] for result in all_results]
    theta_results = [result[2] for result in all_results]
    vx_results = [result[3] for result in all_results]
    vy_results = [result[4] for result in all_results]
    r_results = [result[5] for result in all_results]
    alpha_f_results = [result[6] for result in all_results]
    alpha_r_results = [result[7] for result in all_results]
    vel_error_results = [result[8] for result in all_results]
    lat_error_results = [result[9] for result in all_results]

    # Plot comparisons for each state variable
    plot_trajectory(x_results, y_results, labels, path_spline)
    #plot_comparison(theta_results, labels, "Heading Angle Comparison", "Time Step", "Heading Angle (rad)")
    plot_comparison(vx_results, labels, "Longitudinal Velocity Comparison", "Time Step", "Velocity (m/s)")
    #plot_comparison(vy_results, labels, "Lateral Velocity Comparison", "Time Step", "Lateral Velocity (m/s)")
    #plot_comparison(r_results, labels, "Yaw Rate Comparison", "Time Step", "Yaw Rate (rad/s)")
    #plot_comparison(alpha_f_results, labels, "Front Slip Angle Comparison", "Time Step", "Slip Angle (rad) - Front")
    #plot_comparison(alpha_r_results, labels, "Rear Slip Angle Comparison", "Time Step", "Slip Angle (rad) - Rear")
    plot_comparison(vel_error_results, labels, "velocity error comparison", "Time Step", "Velocity Error (%)")
    plot_comparison(lat_error_results, labels, "lateral error comparison", "Time Step", "Lateral Error (m)")

if __name__ == "__main__":
    main()
