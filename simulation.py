import numpy as np
from planar_quadrotor import GRAVITY
import numpy as np
import matplotlib.pyplot as plt


def simulate_mpc_regulation(N, N_sim, x0, x_eq, u_eq,
                            Ad, Bd, Q, R, P_inf,
                            lb_u, ub_u,
                            lb_x, ub_x,
                            A_inf, b_inf,
                            dt=0.01):
    """
    Simulate MPC control loop over N_sim time steps.

    Parameters:
        N               : Prediction horizon
        N_sim           : Simulation time steps
        x0              : Initial state (absolute)
        x_eq, u_eq      : Equilibrium state and input (absolute)
        Ad, Bd          : Discrete-time system matrices
        Q, R, P_inf     : Cost matrices
        lb_x_tilde etc. : Constraint bounds (relative to equilibrium)
        A_inf, b_inf    : Terminal constraint set

    Returns:
        x_hist, u_hist  : Simulated state and input trajectories (absolute)
    """
    from controllers import solve_mpc_condensed

    dim_u = Bd.shape[1]
    dim_x = Ad.shape[0]


    lb_x_tilde = lb_x - x_eq
    ub_x_tilde = ub_x - x_eq
    lb_u_tilde = lb_u - u_eq
    ub_u_tilde = ub_u - u_eq

    stage_costs_mpc = np.zeros(N_sim)
    x_hist = np.zeros((N_sim + 1, dim_x))
    u_hist = np.zeros((N_sim, dim_u))

    x_hist[0, :] = x0 - x_eq  # convert to delta x for MPC

    for t in range(N_sim):
        x_bar, u_bar = solve_mpc_condensed(
            Ad, Bd, Q, R, P_inf, x_hist[t, :], N,
            lb_u_tilde, ub_u_tilde, lb_x_tilde, ub_x_tilde, A_inf, b_inf
        )
        u0 = u_bar[0, :]  # Use first input in optimal sequence
        x0 = x_hist[t, :]

        u_hist[t, :] = u0
        x_next = Ad @ x_hist[t, :] + Bd @ (u0 + u_eq) - x_eq  # delta x forward
        x_next[3] -= GRAVITY * dt
        x_next[2] -= 0.0004905  # custom vertical correction
        stage_costs_mpc[t] = x0.T @ Q @ x0 + u0.T @ R @ u0

        x_hist[t + 1, :] = x_next

    # Convert back to absolute trajectories
    x_hist = x_hist + x_eq
    u_hist = u_hist + u_eq
    cumulative_stage_costs = np.cumsum(stage_costs_mpc)

    return x_hist, u_hist,cumulative_stage_costs


def plot_mpc_regulation(x_hist, u_hist, N_sim, dt):
    """
    Plot state and control trajectories over time from MPC simulation.
    
    Parameters:
        x_hist : (N_sim+1, n_x) array of state trajectories
        u_hist : (N_sim, n_u) array of control inputs
        N_sim  : Number of simulation steps
        dt     : Time step
    """
    time1 = np.arange(N_sim + 1) * dt
    time2 = np.arange(N_sim) * dt

    fig, axs = plt.subplots(4, 1, figsize=(8, 4), sharex=True)

    axs[0].plot(time1, x_hist[:, 0], label='x', linewidth=1)
    axs[0].plot(time1, x_hist[:, 2], label='y', linewidth=1)
    axs[0].set_ylabel('Position \n [m]', fontsize=8)
    axs[0].set_title('States over Time')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(time1, x_hist[:, 1], label='vx', linewidth=1)
    axs[1].plot(time1, x_hist[:, 3], label='vy', linewidth=1)
    axs[1].set_ylabel('Velocity \n [m/s]', fontsize=8)
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(time1, x_hist[:, 4], label='theta', linewidth=1)
    axs[2].set_ylabel('Angle \n [rad]', fontsize=8)
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(time1, x_hist[:, 5], label='omega', linewidth=1)
    axs[3].set_xlabel('Time [s]')
    axs[3].set_ylabel('Angular Velocity \n [rad/s]', fontsize=8)
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()

    # Plot control inputs
    plt.figure(figsize=(8, 2))
    for i in range(u_hist.shape[1]):
        plt.plot(time2, u_hist[:, i], label=f'u_{i+1}', linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Thrust [N]')
    plt.title('Control Inputs Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def compare_regulation_by_Q(N, N_sim, x0, Q_list, labels, colors,
                               x_eq, u_eq,
                               Ad, Bd, R,
                               lb_u, ub_u,
                               lb_x, ub_x,
                               dt=0.01):
    """
    Compare MPC responses for different Q matrices.

    Parameters:
        N, N_sim         : Horizon length, simulation steps
        x0               : Initial state (absolute)
        Q_list           : List of state cost matrices
        labels, colors   : For plotting
        x_eq, u_eq       : Equilibrium point
        Ad, Bd, R        : Discrete system matrices and input cost
        lb_u, ub_u       : Absolute input bounds
        lb_x, ub_x       : Absolute state bounds
        dt               : Time step (default 0.01)
    """
    from terminal_set import compute_terminal_set


    lb_x_tilde = lb_x - x_eq
    ub_x_tilde = ub_x - x_eq
    lb_u_tilde = lb_u - u_eq
    ub_u_tilde = ub_u - u_eq

    state_r, state_x, state_beta = [], [], []

    for Q in Q_list:
        A_inf, b_inf, P_inf, K = compute_terminal_set(
            Ad, Bd,
            lb_x_tilde, ub_x_tilde,
            lb_u_tilde, ub_u_tilde,
            Q, R,
            max_iter=50
        )

        x_hist, u_hist, _ = simulate_mpc_regulation(
            N, N_sim, x0, x_eq, u_eq,
            Ad, Bd, Q, R, P_inf,
            lb_u, ub_u,
            lb_x, ub_x,
            A_inf, b_inf,
            dt
        )

        state_r.append(x_hist[:, 0])       # x position
        state_x.append(x_hist[:, 2])       # y position
        state_beta.append(x_hist[:, 5])    # theta

    time = np.linspace(0, N_sim * dt, N_sim + 1)
    fig, axs = plt.subplots(3, 1, figsize=(8, 5), sharex=True)

    for i in range(len(Q_list)):
        axs[0].step(time, state_r[i], label=labels[i], where='post', color=colors[i], linewidth=0.8)
    axs[0].set_ylabel(r'$x$ [m]')
    axs[0].grid(True)

    for i in range(len(Q_list)):
        axs[1].step(time, state_x[i], label=labels[i], where='post', color=colors[i], linewidth=0.8)
    axs[1].set_ylabel(r'$y$ [m]')
    axs[1].grid(True)
    axs[1].legend()

    for i in range(len(Q_list)):
        axs[2].step(time, state_beta[i], label=labels[i], where='post', color=colors[i], linewidth=0.8)
    axs[2].set_ylabel(r'$\theta$ [rad]')
    axs[2].set_xlabel('Time [s]')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()




def compare_regulation_by_R(N, N_sim, x0, Q, R_list, labels, colors,
                              x_eq, u_eq,
                              Ad, Bd,
                              lb_u, ub_u,
                              lb_x, ub_x,
                              dt=0.01):
    """
    Compare MPC regulation performance under different R matrices.

    Parameters:
        N, N_sim         : Prediction horizon and simulation steps
        x0               : Initial state (absolute)
        Q                : Fixed state cost matrix
        R_list           : List of input cost matrices to test
        labels, colors   : For plot labeling and color
        x_eq, u_eq       : Equilibrium point
        Ad, Bd           : Discrete-time system matrices
        lb_u, ub_u       : Input bounds (absolute)
        lb_x, ub_x       : State bounds (absolute)
        dt               : Sampling time
    """

    from terminal_set import compute_terminal_set

    lb_x_tilde = lb_x - x_eq
    ub_x_tilde = ub_x - x_eq
    lb_u_tilde = lb_u - u_eq
    ub_u_tilde = ub_u - u_eq

    state_x, state_y, state_theta = [], [], []
    input_1, input_2 = [], []

    for R in R_list:
        A_inf, b_inf, P_inf, K = compute_terminal_set(
            Ad, Bd,
            lb_x_tilde, ub_x_tilde,
            lb_u_tilde, ub_u_tilde,
            Q, R,
            max_iter=50
        )

        x_hist, u_hist, _ = simulate_mpc_regulation(
            N, N_sim, x0, x_eq, u_eq,
            Ad, Bd, Q, R, P_inf,
            lb_u, ub_u,
            lb_x, ub_x,
            A_inf, b_inf,
            dt
        )

        state_x.append(x_hist[:, 0])
        state_y.append(x_hist[:, 2])
        state_theta.append(x_hist[:, 5])
        input_1.append(u_hist[:, 0])
        input_2.append(u_hist[:, 1])

    time_x = np.linspace(0, N_sim * dt, N_sim + 1)
    time_u = np.linspace(0, N_sim * dt, N_sim)

    # Plot state trajectories
    fig, axs = plt.subplots(3, 1, figsize=(8, 5), sharex=True)

    for i in range(len(R_list)):
        axs[0].step(time_x, state_x[i], label=labels[i], where='post', color=colors[i], linewidth=0.5)
    axs[0].set_ylabel(r'$x$ [m]')
    axs[0].grid(True)

    for i in range(len(R_list)):
        axs[1].step(time_x, state_y[i], label=labels[i], where='post', color=colors[i], linewidth=0.5)
    axs[1].set_ylabel(r'$y$ [m]')
    axs[1].grid(True)
    axs[1].legend()

    for i in range(len(R_list)):
        axs[2].step(time_x, state_theta[i], label=labels[i], where='post', color=colors[i], linewidth=0.5)
    axs[2].set_ylabel(r'$\theta$ [rad]')
    axs[2].set_xlabel('Time [s]')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Plot control inputs
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    for i in range(len(R_list)):
        axs[0].step(time_u, input_1[i], label=labels[i], where='post', color=colors[i], linewidth=0.5)
    axs[0].set_ylabel(r'$u_1$ [N]')
    axs[0].grid(True)

    for i in range(len(R_list)):
        axs[1].step(time_u, input_2[i], label=labels[i], where='post', color=colors[i], linewidth=0.5)
    axs[1].set_ylabel(r'$u_2$ [N]')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()




def simulate_lqr(Ad, Bd, Q, R, K, x0, x_eq, u_eq, 
                         N_sim, lb_u, ub_u, dt=0.01, 
                         apply_constraint=False):
    """
    Simulate LQR (with or without input constraints).
    
    Parameters:
        Ad, Bd           : Discrete system matrices
        Q, R             : Cost matrices
        K                : Feedback gain
        x0               : Initial state
        x_eq, u_eq       : Equilibrium state and input
        N_sim            : Simulation steps
        lb_u, ub_u       : Input bounds (absolute)
        dt               : Time step
        apply_constraint : Whether to clip input using bounds

    Returns:
        x_hist           : (N_sim+1, dim_x) state trajectory
        u_hist           : (N_sim, dim_u) control input
        cumulative_costs : (N_sim,) cumulative stage cost
    """
    dim_x = Ad.shape[0]
    dim_u = Bd.shape[1]

    x_hist = np.zeros((N_sim + 1, dim_x))
    u_hist = np.zeros((N_sim, dim_u))
    stage_costs = np.zeros(N_sim)

    x_hist[0, :] = x0

    for t in range(N_sim):
        u = K @ (x_hist[t, :] - x_eq) + u_eq

        if apply_constraint:
            u = np.clip(u, lb_u, ub_u)

        u_hist[t, :] = u

        x_t = x_hist[t, :]
        stage_costs[t] = x_t.T @ Q @ x_t + (u - u_eq).T @ R @ (u - u_eq)

        x_next = Ad @ x_t + Bd @ u
        x_next[3] -= 9.81 * dt
        x_next[2] -= 0.0004905

        x_hist[t + 1, :] = x_next

    cumulative_costs = np.cumsum(stage_costs)

    return x_hist, u_hist, cumulative_costs



import numpy as np
import matplotlib.pyplot as plt

def plot_mpc_vs_lqr_states(x_hist, u_hist,
                                x_hist_lqrc, u_hist_lqrc,
                                x_hist_lqru, u_hist_lqru,
                                N_sim, dt=0.01):
    """
    Plot trajectory and input comparison between MPC, constrained LQR, and unconstrained LQR.

    Parameters:
        x_hist, u_hist         : MPC trajectories
        x_hist_lqrc, u_hist_lqrc : Constrained LQR trajectories
        x_hist_lqru, u_hist_lqru : Unconstrained LQR trajectories
        N_sim                  : Simulation steps
        dt                     : Time step
    """

    time1 = np.arange(N_sim + 1) * dt
    time2 = np.arange(N_sim) * dt

    state_index = [0, 2, 1, 3, 4, 5]
    state_labels = [r'$x$ [m]', r'$y$ [m]', r'$v_x$ [m/s]', r'$v_y$ [m/s]', r'$\theta$ [rad]', r'$\omega$ [rad/s]']

    fig, axs = plt.subplots(6, 1, figsize=(8, 6), sharex=True)
    for i in range(6):
        axs[i].plot(time1, x_hist[:, state_index[i]], label='MPC', linewidth=1)
        axs[i].plot(time1, x_hist_lqrc[:, state_index[i]], label='Constrained LQR', linewidth=1, linestyle='--')
        axs[i].plot(time1, x_hist_lqru[:, state_index[i]], label='Unconstrained LQR', linewidth=1, linestyle='--')
        axs[i].set_ylabel(state_labels[i])
        axs[i].grid(True)

    axs[1].legend()
    axs[5].set_xlabel('Time [s]')
    fig.suptitle('Trajectory comparison: MPC vs Constrained and Unconstrained LQR', fontsize=14)
    plt.tight_layout()
    plt.show()

    input_labels = [r'$u_1$', r'$u_2$']

    fig, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    for i in range(2):
        axs[i].plot(time2, u_hist[:, i], label='MPC', linewidth=1)
        axs[i].plot(time2, u_hist_lqrc[:, i], label='Constrained LQR', linewidth=1, linestyle='--')
        axs[i].plot(time2, u_hist_lqru[:, i], label='Unconstrained LQR', linewidth=1, linestyle='--')
        axs[i].set_ylabel(input_labels[i])
        axs[i].legend()
        axs[i].grid(True)

    axs[1].set_xlabel('Time [s]')
    fig.suptitle('Control input comparison: MPC vs LQR', fontsize=14)
    plt.tight_layout()
    plt.show()



def generate_reference_circle_trajectory(N, dim_x):
    """
    :param N_sim
    :param n_x
    :return: x_ref_all (n_x, N_sim)
    """
    x_ref_all = np.zeros((dim_x, N))
    
    for t in range(N):
        t_norm = t / N

        x_ref_all[:, t] = [
            np.cos(2 * np.pi * t_norm),  # x
            0,                           # vx 
            np.sin(2 * np.pi * t_norm),  # y
            0,                           # vy 
            0,                
            0                            # omega
        ]
    return x_ref_all




def simulate_mpc_tracking(Ad, Bd, Q, R, N_sim, N, x0, x_eq, u_eq,
                           lb_u, ub_u, lb_x, ub_x,
                           x_ref_hist, u_ref_hist,
                           dt=0.01):
    """
    Simulate MPC tracking given full reference trajectories.

    Parameters:
        Ad, Bd           : Discrete system matrices
        Q, R             : Cost matrices
        N_sim            : Total simulation steps
        N                : Prediction horizon
        x0               : Initial state
        x_eq, u_eq       : Equilibrium point
        lb_u, ub_u       : Input bounds (absolute)
        lb_x, ub_x       : State bounds (absolute)
        x_ref_hist       : (N_sim+1, dim_x) reference state trajectory
        u_ref_hist       : (N_sim, dim_u) reference input trajectory
        dt               : Time step (used in c term)

    Returns:
        x_hist, u_hist   : Simulated full state and input trajectory
    """
    from controllers import solve_mpc_condensed_track
    from terminal_set import compute_terminal_set

    dim_x = Ad.shape[0]
    dim_u = Bd.shape[1]

    lb_x_tilde = lb_x - x_eq
    ub_x_tilde = ub_x - x_eq
    lb_u_tilde = lb_u - u_eq
    ub_u_tilde = ub_u - u_eq

    # Offset (e.g. gravity )
    c = np.array([0., 0., -0.0004905, -9.81 * dt, 0., 0.])

    # Terminal cost matrix
    _, _, P_inf, _ = compute_terminal_set(
        Ad, Bd, lb_x_tilde, ub_x_tilde, lb_u_tilde, ub_u_tilde, Q, R
    )

    x_hist = np.zeros((N_sim + 1, dim_x))
    u_hist = np.zeros((N_sim, dim_u))
    x_hist[0, :] = x0

    for t in range(N_sim):
        x_bar, u_bar = solve_mpc_condensed_track(
            Ad, Bd, Q, R, P_inf,
            x_hist[t, :], N,
            lb_u, ub_u, lb_x, ub_x,
            xref_vec=x_ref_hist[t:t + N + 1, :],
            uref_vec=u_ref_hist[t:t + N, :]
        )

        u0 = u_bar[0, :]

        x_next = Ad @ x_hist[t, :] + Bd @ u0 + c
        x_hist[t + 1, :] = x_next
        u_hist[t, :] = u0

    return x_hist, u_hist


def plot_xy_trajectory(x_hist, x_ref_all, xlabel='x', ylabel='y',
                       title='Trajectory in XY Plane',
                       legend_labels=('Actual Trajectory', 'Reference Trajectory')):
    """
    Plot the robot's actual vs. reference trajectory in the XY plane.

    Parameters:
        x_hist        : (N+1, dim_x) array of actual state trajectory
        x_ref_all     : (dim_x, N+1) array of reference trajectory
        xlabel        : Label for X-axis (default 'x')
        ylabel        : Label for Y-axis (default 'y')
        title         : Plot title
        legend_labels : Tuple of labels for actual and reference trajectory
    """
    plt.figure(figsize=(8, 6))

    # Actual trajectory: x vs y (from state vector)
    plt.plot(x_hist[:, 0], x_hist[:, 2], 'b-', label=legend_labels[0])

    # Reference trajectory: x vs y (assumes shape (dim_x, N+1))
    plt.plot(x_ref_all[0, :], x_ref_all[2, :], 'g--', label=legend_labels[1])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def simulate_mpc_disturbance(
    Ad, Bd, Q, R, P_inf, A_inf, b_inf,
    C, H,
    lb_x, ub_x, lb_u, ub_u,
    x0, yset, N_sim, N,
    Bp, ps,     # disturbance input matrix + profile
    Lx, Ld,     # observer gains
    Bd_dist, C_dist,  # disturbance models
    dt=0.01,
):
    
    from planar_quadrotor import GRAVITY
    from controllers import solve_OTS_disturbance,solve_mpc_condensed
    dim_x = Ad.shape[0]
    dim_u = Bd.shape[1]
    dim_d = Bd_dist.shape[1]

    # Initialize histories
    x_hist = np.zeros((N_sim + 1, dim_x))
    u_hist = np.zeros((N_sim , dim_u))
    xhat_hist = np.zeros((N_sim, dim_x))
    dhat_hist = np.zeros((N_sim, dim_d))
    xs_hist = np.zeros((N_sim, dim_x))
    us_hist = np.zeros((N_sim, dim_u))

    xhat_ = np.zeros(dim_x)
    dhat_ = np.zeros(dim_d)

    x_hist[0] = x0

    for k in range(N_sim):
        x = x_hist[k]

        # Measurement
        y = C @ x

        # state estimate（Luenberger observer correction）
        ey = y - C @ xhat_ - C_dist @ dhat_
        xhat = xhat_ + Lx @ ey
        dhat = dhat_ + Ld @ ey

        xhat_hist[k] = xhat
        dhat_hist[k] = dhat

        # Target selection
        xs, us = solve_OTS_disturbance(Ad, Bd, C, H, yset, dhat, Bd_dist, C_dist)
        xs_hist[k, :] = xs
        us_hist[k, :] = us

        # MPC solve for deviation
        x0_mpc = xhat - xs
        lb_x_tilde = lb_x -  xs
        ub_x_tilde = ub_x -  xs

        lb_u_tilde = lb_u - us
        ub_u_tilde = ub_u - us
    

        _, u_seq=solve_mpc_condensed(Ad, Bd, Q, R, P_inf, x0_mpc, N,
                                            lb_u_tilde, ub_u_tilde,
                                            lb_x_tilde, ub_x_tilde,
                                            A_inf, b_inf)

        u = u_seq[0] + us
        u_hist[k] = u

        # True state update with disturbance
        x_next = Ad @ x + Bd @ u + Bp @ ps[:, k]
        x_next[3] -= GRAVITY * dt
        x_next[2] -= 0.0004905
        x_hist[k + 1] = x_next

        # Observer prediction step
        xhat_ = Ad @ xhat + Bd @ u + Bd_dist @ dhat
        xhat_[3] -= GRAVITY * dt
        xhat_[2] -= 0.0004905
        dhat_ = dhat

    return x_hist, u_hist, xhat_hist, dhat_hist, xs_hist, us_hist


