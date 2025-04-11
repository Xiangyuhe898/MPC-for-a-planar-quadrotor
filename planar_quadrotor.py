import numpy as np
from scipy.linalg import expm
import matplotlib.animation as animation
import IPython
import matplotlib.pyplot as plt


# constant
MASS = 0.600
INERTIA = 0.15
LENGTH = 0.2
GRAVITY = 9.81

def get_linearized_matrices(x_eq, u_eq):
    theta = x_eq[4]
    u1, u2 = u_eq

    A = np.zeros((6, 6))
    B = np.zeros((6, 2))

    A[0, 1] = 1.0
    A[1, 4] = -(u1 + u2) * np.cos(theta) / MASS
    A[2, 3] = 1.0
    A[3, 4] = -(u1 + u2) * np.sin(theta) / MASS
    A[4, 5] = 1.0

    B[1, 0] = -np.sin(theta) / MASS
    B[1, 1] = -np.sin(theta) / MASS
    B[3, 0] =  np.cos(theta) / MASS
    B[3, 1] =  np.cos(theta) / MASS
    B[5, 0] =  LENGTH / INERTIA
    B[5, 1] = -LENGTH / INERTIA

    return A, B

def linearize_and_discretize(x_eq, u_eq, dt):
    A, B = get_linearized_matrices(x_eq, u_eq)
    dim_x = A.shape[0]
    dim_u = B.shape[1]

    ABc = np.zeros((dim_x + dim_u, dim_x + dim_u))
    ABc[:dim_x, :dim_x] = A
    ABc[:dim_x, dim_x:] = B

    expm_ABc = expm(ABc * dt)
    Ad = expm_ABc[:dim_x, :dim_x]
    Bd = expm_ABc[:dim_x, dim_x:]

    return Ad, Bd

def animate_robot_with_trail(x, u, dt=0.01, show_trail=True, trail_color='c', trail_lw=1.5,x_ref_all=None):
    '''Copyright (c) 2023 navoday01'''

    min_dt = 0.1
    if dt < min_dt:
        steps = int(min_dt / dt)
        use_dt = int(np.round(min_dt * 1000))
    else:
        steps = 1
        use_dt = int(np.round(dt * 1000))

    plotx = x[:, ::steps]
    plotx = plotx[:, :-1]
    plotu = u[:, ::steps]

    
    ref_x = x_ref_all[:, ::2*steps]  
    fig = plt.figure(figsize=[8.5, 8.5])
    ax = fig.add_subplot(111, autoscale_on=False, xlim=[-1.5, 1.5], ylim=[-1.5, 1.5])
    ax.grid()

    
    if x_ref_all is not None:
        ax.plot(ref_x[0, :], ref_x[2, :], color='gray', lw=1.5, linestyle='-', label='Reference')

    list_of_lines = []

    # Create robot parts
    line, = ax.plot([], [], 'k', lw=6)  # main frame
    list_of_lines.append(line)
    line, = ax.plot([], [], 'b', lw=4)  # left propeller
    list_of_lines.append(line)
    line, = ax.plot([], [], 'b', lw=4)  # right propeller
    list_of_lines.append(line)
    line, = ax.plot([], [], 'r', lw=1)  # left thrust
    list_of_lines.append(line)
    line, = ax.plot([], [], 'r', lw=1)  # right thrust
    list_of_lines.append(line)

    # Add trajectory trail line
    if show_trail:
        trail_line, = ax.plot([], [], trail_color, lw=trail_lw, linestyle='--', label='Trajectory')
        list_of_lines.append(trail_line)
    
   
    def _animate(i):
        for l in list_of_lines:
            l.set_data([], [])

        theta = plotx[4, i]
        x_pos = plotx[0, i]
        y_pos = plotx[2, i]
        trans = np.array([[x_pos, x_pos], [y_pos, y_pos]])
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        main_frame = np.array([[-LENGTH, LENGTH], [0, 0]])
        left_propeller = np.array([[-1.3 * LENGTH, -0.7 * LENGTH], [0.1, 0.1]])
        right_propeller = np.array([[1.3 * LENGTH, 0.7 * LENGTH], [0.1, 0.1]])
        left_thrust = np.array([[LENGTH, LENGTH], [0.1, 0.1 + plotu[0, i] * 0.04]])
        right_thrust = np.array([[-LENGTH, -LENGTH], [0.1, 0.1 + plotu[1, i] * 0.04]])

        main_frame = rot @ main_frame + trans
        left_propeller = rot @ left_propeller + trans
        right_propeller = rot @ right_propeller + trans
        left_thrust = rot @ left_thrust + trans
        right_thrust = rot @ right_thrust + trans

        list_of_lines[0].set_data(main_frame[0, :], main_frame[1, :])
        list_of_lines[1].set_data(left_propeller[0, :], left_propeller[1, :])
        list_of_lines[2].set_data(right_propeller[0, :], right_propeller[1, :])
        list_of_lines[3].set_data(left_thrust[0, :], left_thrust[1, :])
        list_of_lines[4].set_data(right_thrust[0, :], right_thrust[1, :])

        if show_trail:
            trail_x = plotx[0, :i+1]
            trail_y = plotx[2, :i+1]
            list_of_lines[5].set_data(trail_x, trail_y)

        if i == plotx.shape[1] - 1:
            fig.savefig("final_frame.png", dpi=300)  # save as PNG picture


        return list_of_lines

    def _init():
        return _animate(0)

    ani = animation.FuncAnimation(fig, _animate, np.arange(0, plotx.shape[1]),
                                  interval=use_dt, blit=True, init_func=_init)
    plt.close(fig)
    plt.close(ani._fig)
    IPython.display.display_html(IPython.core.display.HTML(ani.to_html5_video()))