from cubic_spline_planner import *
from casadi import *
from casadi.tools import *

class MPC:
    def __init__(self, T, dt, N, max_steer, min_steer):
        # MPC time
        self.T = T  # Horizon length in seconds
        self.dt = dt # Horizon timesteps
        self.N = N  # Horizon total points

        self.max_steer = max_steer # Maximum steering angle in radians
        self.min_steer = min_steer  # Minimum steering angle in radians

        self.F = 0
    def casadi_model(self):
        # Control
        # Create 1r-1c matrix containing control inputs. 
        # Set steer as element
        u = MX.sym("u",1)
        steer = u[0]

        # Constants - Model parameters
        Lr = 1.42 
        Lf = 1.156
        L = Lf + Lr

        # State
        x = MX.sym("x",4)
        sx    = x[0]  # position x
        sy    = x[1]  # position y
        yaw   = x[2]  # yaw
        speed = x[3]

        # ODE right hand side
        sxdot    = speed*cos(yaw)
        sydot    = speed*sin(yaw)
        yawdot   = (speed/L)*tan(steer)
        speeddot = 0.0

        # Concatenate vertically the expressions creating a row vector
        xdot = vertcat(sxdot, sydot, yawdot, speeddot)

        # ODE right hand side function
        # as input are used the state and control inputs,
        # and as output the row vector containing the model expressions
        f = Function('f', [x,u],[xdot])

        # Integrate the step with Explicit Euler
        IN = 1
        xj = x
        for i in range(IN):
            fj = f(xj,u)
            xj += self.dt*fj/IN

        # Discrete time dynamics function
        self.F = Function('F', [x,u],[xj])

    def opt_step(self, target, state):
        

        # Control for all segments
        nu = self.N #number of states
        Us = MX.sym("U",nu) #steer control input

        # Initial conditions
        x0 = [state.x, state.y, state.theta, state.vx]
        X0 = MX(x0) # vector containing the initial state

        J = 0 # objective function that should be minimized by the nlp solver

        # build graph
        X=X0
        G = None
        lbg = []
        ubg = []

        # For every temporal step
        for k in range(nu):
            X = self.F(X, vertcat(Us[k])) #Integrate the step
            gain_mult = 1
            # give more importance to the last step using a bigger gain
            if(k == nu-1):
                gain_mult=1 #You can use this multiplier as terminal cost
            J += 100.0*gain_mult*(X[0]-target[k][0])**2 #x error cost 
            J += 100.0*gain_mult*(X[1]-target[k][1])**2 #y error cost
            #J += 10.0*gain_mult*(X[2] - target[k][2])**2 #heading error cost
        # G = X[index] #if you want to set a state to constrain in arg["lbg"] and arg["ubg"]. It can be ignored


        # Objective function and constraints
        J += mtimes(Us.T,Us)*100000.0 #Consider to set this weight dependent on speed

        # NLP
        nlp = {'x':vertcat(Us), 'f':J}
        
        # Allocate an NLP solver
        opts = {
        "print_time": False,
        "verbose": False,
        "ipopt.tol": 1e-3,
        "ipopt.acceptable_tol": 1e-2,
        "ipopt.acceptable_iter": 5,
        "ipopt.max_iter": 20,
        "ipopt.print_level": 0,
        "expand": True,
        }

        solver = nlpsol("solver", "ipopt", nlp, opts)
        arg = {}

        # Bounds on u and initial condition
        arg["lbx"] =  vertcat(self.min_steer*np.ones(nu)) # lower bound for steer
        arg["ubx"] =  vertcat(self.max_steer*np.ones(nu)) # upper bound for steer
        arg["x0"] =    0.0 # first guess 

        #These can be ingnored 
        # # Bounds on g
        # arg["lbg"] = lower_bound on a state
        # arg["ubg"] = inf

        # Solve the problem
        res = solver(**arg)
        #print "f:", res["f"]
        ctrls = reshape(res["x"], (nu,1)).T #reshape to have a row for each step

        return ctrls[0][0]
