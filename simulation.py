import numpy as np


class Simulation:
    def __init__(self, lf, lr, mass, Iz, dt, integrator="euler", model="kinematic"):
        """
        Initialize the simulation parameters.
        """
        self.l_f = lf                   # Distance to front axle (m)
        self.l_r = lr                   # Distance to rear axle (m)
        self.l_wb = lf + lr
        self.mass = mass                # Vehicle mass (kg)
        self.I_z = Iz                   # Yaw moment of inertia (kg*m^2)
        self.dt = dt                    # Time step (s)
        self.integrator = integrator    # Integrator choice
        self.model = model              # Model choice

        # Aerodynamic and rolling resistance parameters
        self.rho = 1.225               # Air density (kg/m^3)
        self.C_d = 0.3                 # Drag coefficient (typical for cars)
        self.A = 2.2                   # Frontal area (m^2)
        self.C_rr = 0.015              # Rolling resistance coefficient

        # Initialize states
        self.x = 0                     # X position (m)
        self.y = 0                      # Y position (m)
        self.theta = 0                  # Heading angle (rad)
        self.vx = 0.001                    # Longitudinal velocity (m/s)
        self.vy = 0                     # Lateral velocity (m/s)
        self.r = 0                      # Yaw rate (rad/s)
        self.beta = 0

        # Pacejka's Magic Formula coefficients
        self.B, self.C, self.D, self.E = 7.1433, 1.3507, 1.0489, -0.0074722
        self.B_f, self.C_f, self.D_f, self.E_f = self.B, self.C, self.D, self.E
        self.B_r, self.C_r, self.D_r, self.E_r = self.B, self.C, self.D, self.E

        self.Cf, self.Cr = self.B_f*self.C_f*self.D_f, self.B_r*self.C_r*self.D_r  # Cornering stiffness front/rear (N/rad)

    def kinematic_model(self, ax, delta):
        """ Kinematic single-track model equations of motion. """

        # Aerodynamic drag and rolling resistance forces
        v = self.vx * self.vx + self.vy * self.vy
        F_aero = (1/2) * self.rho * self.C_d * self.A * v
        F_roll = self.C_rr * self.mass * 9.81

        # TODO: slide 18 
        dx = np.array([
            self.vx*np.cos(self.theta)-self.vy*np.sin(self.theta),
            self.vx*np.sin(self.theta)+self.vy*np.cos(self.theta),
            (self.vx/self.l_wb)*np.tan(delta),
            ax+self.r*self.vy,
            0,
            0
        ])
        return dx

    # TODO: slide 29
    def linear_single_track_model(self, ax, delta):
        """ Linear single-track model with aerodynamic and rolling resistance. """

        # Tire slip angles
        alpha_f = delta - ((self.vy+self.l_f*self.r)/self.vx)  # Front tire slip angle
        alpha_r = -((self.vy-self.l_r*self.r)/self.vx)         # Rear tire slip angle

        # Vertical forces (nominal vertical load)
        Fz_f_nominal = (self.l_r/self.l_wb)*self.mass*9.81
        Fz_r_nominal = (self.l_f/self.l_wb)*self.mass*9.81

        # Front and rear lateral forces
        Fyf = Fz_f_nominal * self.Cf * alpha_f
        Fyr = Fz_r_nominal * self.Cr * alpha_r

        # Aerodynamic drag and rolling resistance forces
        v = self.vx * self.vx + self.vy * self.vy
        # v=self.vx*self.vx
        F_aero = (1/2)*self.rho*self.C_d*self.A*v
        F_roll = self.C_rr * self.mass * 9.81
        F_x = (self.mass * ax) - F_aero - F_roll

        # Dynamics equations
        dx = np.array([
            self.vx*np.cos(self.theta)-self.vy*np.sin(self.theta),           # dx/dt
            self.vx*np.sin(self.theta)+self.vy*np.cos(self.theta),           # dy/dt
            self.r,                                                          # dtheta/dt
            (1/self.mass)*(F_x-Fyf*np.sin(delta)+self.mass*self.vy*self.r),  # dvx/dt with resistive forces
            (1/self.mass)*(Fyr+Fyf*np.cos(delta)-self.mass*self.vx*self.r),  # dvy/dt
            (1/self.I_z)*(Fyf*self.l_f*np.cos(delta)-Fyr*self.l_r)           # dr/dt
        ])

        return dx

    def nonlinear_single_track_model(self, ax, delta):
        """ Nonlinear single-track model with aerodynamic and rolling resistance. """

        # Tire slip angles
        alpha_f = delta - np.arctan((self.vy+self.l_f*self.r)/self.vx)  # Front tire slip angle
        alpha_r = - np.arctan((self.vy-self.l_r*self.r)/self.vx)         # Rear tire slip angle

        # Vertical forces (nominal vertical load)
        Fz_f_nominal = (self.l_r/self.l_wb)*self.mass*9.81
        Fz_r_nominal = (self.l_f/self.l_wb)*self.mass*9.81

        # Front and rear lateral forces
        Fyf = Fz_f_nominal * self.D * np.sin(self.C*np.arctan(self.B*alpha_f - self.E*(self.B * alpha_f - np.arctan(self.B*alpha_f))))
        Fyr = Fz_r_nominal * self.D * np.sin(self.C*np.arctan(self.B*alpha_r - self.E*(self.B * alpha_r - np.arctan(self.B*alpha_r))))

        # Aerodynamic drag and rolling resistance forces
        v = self.vx * self.vx + self.vy * self.vy
        # v=self.vx*self.vx
        F_aero = (1/2)*self.rho*self.C_d*self.A*v
        F_roll = self.C_rr * self.mass * 9.81
        # F_x = (self.mass * ax) - F_aero - F_roll
        F_x = (self.mass * ax) - F_aero - F_roll

        # Dynamics equations
        dx = np.array([
            self.vx*np.cos(self.theta)-self.vy*np.sin(self.theta),           # dx/dt
            self.vx*np.sin(self.theta)+self.vy*np.cos(self.theta),           # dy/dt
            self.r,                                                          # dtheta/dt
            (1/self.mass)*(F_x-Fyf*np.sin(delta)+self.mass*self.vy*self.r),  # dvx/dt with resistive forces
            (1/self.mass)*(Fyr+Fyf*np.cos(delta)-self.mass*self.vx*self.r),  # dvy/dt
            (1/self.I_z)*(Fyf*self.l_f*np.cos(delta)-Fyr*self.l_r)           # dr/dt
        ])

        return dx

    def integrate(self, ax, delta):
        """ Select the integrator method and apply it to update the state. """
        if self.integrator == "euler":
            self.euler_step(ax, delta)
        elif self.integrator == "rk4":
            self.rk4_step(ax, delta)

    def euler_step(self, ax, delta):
        """ Euler integration method. """
        dx = self.compute_dx(ax, delta)
        self.update_state(dx)

    def rk4_step(self, ax, delta):
        """ Runge-Kutta 4th order integration method. """
        k1 = self.compute_dx(ax, delta)
        self.update_state(k1, scale=0.5)

        k2 = self.compute_dx(ax, delta)
        self.update_state(k2, scale=0.5, revert=k1)

        k3 = self.compute_dx(ax, delta)
        self.update_state(k3, scale=1, revert=k2)

        k4 = self.compute_dx(ax, delta)

        # Combine k1, k2, k3, k4 for RK4 update
        dx = (k1 + 2*k2 + 2*k3 + k4) / 6
        self.update_state(dx)

    def compute_dx(self, ax, delta):
        """ Compute the state derivatives using the chosen model. """
        if self.model == "kinematic":
            return self.kinematic_model(ax, delta)
        elif self.model == "linear":
            return self.linear_single_track_model(ax, delta)
        elif self.model == "nonlinear":
            return self.nonlinear_single_track_model(ax, delta)

    def update_state(self, dx, scale=1, revert=None):
        """ Update state with scaled dx. Optionally revert previous state for RK4. """
        if revert is not None:
            self.x -= revert[0] * self.dt
            self.y -= revert[1] * self.dt
            self.theta -= revert[2] * self.dt
            self.vx -= revert[3] * self.dt
            self.vy -= revert[4] * self.dt
            self.r -= revert[5] * self.dt

        self.x += dx[0] * self.dt * scale
        self.y += dx[1] * self.dt * scale
        self.theta += dx[2] * self.dt * scale
        self.vx += dx[3] * self.dt * scale
        self.vy += dx[4] * self.dt * scale
        self.r += dx[5] * self.dt * scale
