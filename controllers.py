from qpsolvers import solve_qp # https://qpsolvers.github.io/qpsolvers/quadratic-programming.html#qpsolvers.solve_qp
import numpy as np





def gen_prediction_matrices(Ad, Bd, N):
    dim_x = Ad.shape[0]
    dim_u = Bd.shape[1]
    
    T = np.zeros(((dim_x * (N + 1), dim_x)))
    S = np.zeros(((dim_x * (N + 1), dim_u * N)))
    
    # Condensing
    power_matricies = []    # power_matricies = [I, A, A^2, ..., A^N]
    power_matricies.append(np.eye(dim_x))
    for k in range(N):
        power_matricies.append(power_matricies[k] @ Ad)#add elemrnts to the end of list, the list has no dimension, storing  successive powers of A
    
    for k in range(N + 1):
        T[k * dim_x: (k + 1) * dim_x, :] = power_matricies[k]# fill in T matrix, where each block stores kth power of A
        for j in range(N):
            if k > j:
                S[k * dim_x:(k + 1) * dim_x, j * dim_u:(j + 1) * dim_u] = power_matricies[k - j - 1] @ Bd
                
    return T, S

def gen_cost_matrices(Q, R, P, T, S, x0, N):
    dim_x = Q.shape[0]
    
    Q_bar = np.zeros(((dim_x * (N + 1), dim_x * (N + 1))))
    Q_bar[-dim_x:, -dim_x:] = P
    Q_bar[:dim_x * N, :dim_x * N] = np.kron(np.eye(N), Q) 

    R_bar = np.kron(np.eye(N), R)
    

    H = S.T @ Q_bar @ S + R_bar
    h = S.T @ Q_bar @ T @ x0
    H = 0.5 * (H + H.T) # Ensure symmetry!
    
    return H, h


def gen_constraint_matrices(A, B, T, S, D, c_lb, c_ub, u_lb, u_ub, N, x0):
    dim_x = A.shape[0]
    dim_u = B.shape[1]
    
    # Construct \tilde{D}, \tilde{E}, \tilde{b}
    D_tilde = np.block([
        [D @ A],  # D * A
        [-D @ A],  # -D * A
        [np.zeros((dim_u , dim_x))],  # For input bounds
        [np.zeros((dim_u , dim_x))]
    ])


    E_tilde = np.block([
        [D @ B],   # D * B
        [-D @ B],  # -D * B
        [np.eye(dim_u)],  # Identity for u upper bound
        [-np.eye(dim_u)]  # Negative identity for u lower bound
    ])

    
    b_tilde = np.concatenate([
        c_ub,    # Upper bound for D(Ax + Bu)
        -c_lb,   # Lower bound for D(Ax + Bu)
        u_ub,    # Upper bound for u
        -u_lb    # Lower bound for u
    ])
    
    # Construct \bar{D}, \bar{E}, \bar{b}
    D_bar = np.kron(np.eye(N), D_tilde)  # Extend over N steps
    E_bar = np.kron(np.eye(N), E_tilde)
    b_bar = np.kron(np.ones(N), b_tilde)  # Stack over N horizons


    S_tilde = S[:-dim_x, :]  # 删除最后 dim_x 行
    T_tilde = T[:-dim_x, :]  # 删除最后 dim_x 行


    # Compute G and g
    G = D_bar @ S_tilde + E_bar
    g = b_bar - D_bar @ T_tilde @ x0  

    return G, g

def extend_matrices(G, g, S_N, b_N, N, dim_x, dim_u,A_inf):
    
    """
    construct G_tilde and g_tilde:
    
    G_tilde = [[G, 0], 
               [0, A_inf @ S_N]]
    g_tilde = [[g], 
               [b_N]]
    """
    # dimensions
    # G_rows = 2 * N * (dim_x + dim_u)  # rows of G
    # G_cols = N * dim_u  # columns of G
    # S_cols = N * dim_u  # columns of S_N
    # A_rows = 2 * dim_x  # rows of A_inf
    # A_cols = dim_x  # columns A_inf
    # g_rows = 2 * N * (dim_x + dim_u)  # rows of g
    # b_rows = dim_x  # rows of b_N 

    # construct G_tilde
    G_tilde = np.block([
        [G],   
        [A_inf@S_N]  
    ])

    # construct g_tilde
    g = g.reshape(-1, 1) 
    b_N = b_N.reshape(-1, 1)  
    g_tilde = np.block([
        [g], 
        [b_N]
    ])


    return G_tilde, g_tilde

def get_S_N_b_N(S, dim_x,A_inf, b_inf,x0,N,Ad):
    """
    extract last dim_x rows of S，get S_N
    """
    S_N = S[-dim_x:, :]  
    # compute A^N
    A_powerN = np.linalg.matrix_power(Ad, N)  # A^N
    
    # conpute b_N
    b_N = b_inf - A_inf @ A_powerN @ x0

    return S_N, b_N




def solve_mpc_condensed(Ad, Bd, Q, R, P, x0, N, u_lb, u_ub,c_lb, c_ub, A_inf, b_inf):
    dim_u = Bd.shape[1]
    dim_x = Ad.shape[0]

    T, S = gen_prediction_matrices(Ad, Bd, N)

    H, h = gen_cost_matrices(Q, R, P, T, S, x0, N)
    
    D=np.eye(dim_x)
    G, g = gen_constraint_matrices(Ad, Bd, T, S, D, c_lb, c_ub, u_lb, u_ub, N,x0)
   
    S_N, b_N=get_S_N_b_N(S, dim_x,A_inf, b_inf,x0,N,Ad)

    G_tilde, g_tilde = extend_matrices(G, g, S_N, b_N, N, dim_x, dim_u, A_inf)


    try:
        u_bar = solve_qp(H, h, G_tilde, g_tilde, solver='quadprog')#H:quadratic cost matrix/h:linear cost vector/G:constraint matrix/g:constraint bound vector
        if u_bar is None:
            raise ValueError("solve_qp returned None.")
    except Exception as e:
        print("QP Solver Error:", str(e))
    
    
    x_bar = T @ x0 + S @ u_bar
    x_bar = x_bar.reshape((N + 1, dim_x))
    u_bar = u_bar.reshape((N, dim_u))
   
    
    return x_bar, u_bar



def gen_cost_matrices_track(Q, R, P, T, S, x0, N, xref_vec, uref_vec):
    dim_x = Q.shape[0]
    dim_u = R.shape[0]

    Q_bar = np.zeros(((dim_x * (N + 1), dim_x * (N + 1))))
    Q_bar[:dim_x * N, :dim_x * N] = np.kron(np.eye(N), Q)
    Q_bar[-dim_x:, -dim_x:] = P

    R_bar = np.kron(np.eye(N), R)

    xref_vec = xref_vec.reshape(-1, 1)  # ((N+1)*n, 1)
    uref_vec = uref_vec.reshape(-1, 1)  # (N*m, 1)

    H = S.T @ Q_bar @ S + R_bar
    h = S.T @ Q_bar @ (T @ x0.reshape(-1, 1) - xref_vec) - R_bar @ uref_vec

    H = 0.5 * (H + H.T)  
    return H, h.flatten()


def solve_mpc_condensed_track(Ad, Bd, Q, R, P, x0, N, u_lb, u_ub,c_lb, c_ub,xref_vec, uref_vec):
    dim_u = Bd.shape[1]
    dim_x = Ad.shape[0]

    T, S = gen_prediction_matrices(Ad, Bd, N)

    H, h = gen_cost_matrices_track(Q, R, P, T, S, x0, N, xref_vec, uref_vec)

    D=np.eye(dim_x)
    G, g = gen_constraint_matrices(Ad, Bd, T, S, D, c_lb, c_ub, u_lb, u_ub, N,x0)
    


    try:
        u_bar = solve_qp(H, h, G, g, solver='quadprog')#H:quadratic cost matrix/h:linear cost vector/G:constraint matrix/g:constraint bound vector
        if u_bar is None:
            raise ValueError("solve_qp returned None.")
    except Exception as e:
        print("QP Solver Error:", str(e))
    
    
   

    x_bar = T @ x0 + S @ u_bar
    x_bar = x_bar.reshape((N + 1, dim_x))
    u_bar = u_bar.reshape((N, dim_u))
    
    return x_bar, u_bar



def solve_OTS_tracking(H, C, Ad, Bd, yset):
    """
    Compute the reference state and input (x_ref_now, u_ref_now)
    from desired output yset using inverse dynamics.

    Parameters:
        H      : Output selection matrix (e.g., selects [x, y])
        C      : Output matrix (maps full state to y)
        Ad     : Discrete A matrix
        Bd     : Discrete B matrix
        yset   : Desired output (e.g., [x, vx, y, vy])
        c      : Offset vector (e.g., to compensate gravity)

    Returns:
        x_ref_now : Reference state (flat, shape: (dim_x,))
        u_ref_now : Reference input (flat, shape: (dim_u,))
    """
    from planar_quadrotor import GRAVITY
    dim_x = Ad.shape[0]
    dim_u = Bd.shape[1]
    dt=0.01
    c=np.array([0., 0., -0.0004905,-GRAVITY * dt, 0.,0.])

    if yset.ndim == 1:
        yset = yset.reshape(-1, 1)

    # 构造约束矩阵 G
    G = np.block([
        [np.eye(dim_x) - Ad, -Bd],
        [H @ C, np.zeros((H.shape[0], dim_u))]
    ])

    # 求逆
    Ginv = np.linalg.inv(G)

    # 构造右端项并求解
    rhs = np.vstack([c.reshape(-1, 1), H @ yset])
    qs = Ginv @ rhs

    x_ref_now = qs[:dim_x].flatten()
    u_ref_now = qs[dim_x:].flatten()

    return x_ref_now, u_ref_now








def construct_augmented_system_with_observability_check(Ad, Bd, C, dim_d):
    """
    Construct an augmented system with disturbance and check its observability.

    Parameters:
        Ad      : State transition matrix (n x n)
        Bd      : Control input matrix (n x m)
        C       : Output matrix (p x n)
        dim_d   : Disturbance dimension (e.g., 2 for wind on vx, vy)

    Returns:
        A_aug   : Augmented A matrix (n+dim_d x n+dim_d)
        B_aug   : Augmented B matrix (n+dim_d x m)
        C_aug   : Augmented C matrix (p x n+dim_d)
        observable : True if system is observable, False otherwise
    """
    dim_x = Ad.shape[0]
    dim_u = Bd.shape[1]
    dim_y = C.shape[0]

    # disturbance-to-state matrix
    Bd_dist = np.zeros((dim_x, dim_d))
    Bd_dist[1, 0] = 1.0  # d1 acts on vx
    Bd_dist[3, 1] = 1.0  # d2 acts on vy

    # disturbance-to-output matrix
    C_dist = np.zeros((dim_y, dim_d))  # assumes disturbance doesn't influence output directly

    # augmented system
    A_aug = np.block([
        [Ad, Bd_dist],
        [np.zeros((dim_d, dim_x)), np.eye(dim_d)]
    ])

    B_aug = np.vstack([Bd, np.zeros((dim_d, dim_u))])
    C_aug = np.hstack([C, C_dist])

    # observability check
    G_obs = np.vstack([
        np.hstack([np.eye(dim_x) - Ad, -Bd_dist]),
        np.hstack([C, C_dist])
    ])
    rank_G = np.linalg.matrix_rank(G_obs)
    expected_rank = dim_x + dim_d

    observable = (rank_G == expected_rank)

    print(f"Rank of observability matrix G: {rank_G}")
    print(f"Expected rank: {expected_rank}")
    print(f"System is {'observable' if observable else 'NOT observable'}.")

    return A_aug, B_aug, C_aug, observable


def design_observer_gain(A_aug, C_aug, eigenvalues, dim_x):
    """
    Design observer gain matrix L for the augmented system using pole placement.

    Parameters:
        A_aug       : Augmented A matrix (nx + nd, nx + nd)
        C_aug       : Augmented C matrix (ny, nx + nd)
        eigenvalues : Desired closed-loop eigenvalues for the observer (length nx+nd)
        nx          : Dimension of the original state (not including disturbance)

    Returns:
        L           : Full observer gain matrix (nx+nd x ny)
        Lx          : Observer gain on x part (nx x ny)
        Ld          : Observer gain on d part (nd x ny)
    """
    from scipy.signal import place_poles
    dim_d = A_aug.shape[0] - dim_x
    assert A_aug.shape[0] == C_aug.shape[1], "A_aug and C_aug size mismatch"

    # Place observer poles
    place_result = place_poles(A_aug.T, C_aug.T, eigenvalues)
    L = place_result.gain_matrix.T

    # Split into parts
    Lx = L[:dim_x, :]
    Ld = L[dim_x:, :]

    return L, Lx, Ld



import numpy as np

def solve_OTS_disturbance(Ad, Bd, C, H, yset, dhat, Bd_dist, C_dist, dt=0.01):
    """
    Solve for reference state and input (xs, us) with disturbance compensation.

    Parameters:
        A, B      : System matrices
        C         : Output matrix
        H         : Output selector (e.g., pick x and y)
        yset      : Desired output (shape (4,) or (4,1))
        dhat      : Disturbance estimate (shape (2,) or (2,1))
        Bd_dist   : Disturbance to state matrix
        C_dist        : Disturbance to output matrix
        dt        : Time step

    Returns:
        xs        : Solved reference state (dim_x,)
        us        : Solved reference input (dim_u,)
    """
    from planar_quadrotor import GRAVITY

    dim_x = Ad.shape[0]
    dim_u = Bd.shape[1]
    # Ensure proper shape
    yset = yset.reshape(-1, 1)
    dhat = dhat.reshape(-1, 1)

    # Offset (gravity + compensation)
    c = np.array([0., 0., -0.0004905, -GRAVITY * dt, 0., 0.]).reshape(-1, 1)

    # Construct G matrix
    G = np.block([
        [np.eye(dim_x) - Ad, -Bd],
        [H @ C, np.zeros((H.shape[0], dim_u))]
    ])

    # Solve inverse
    Ginv = np.linalg.inv(G)

    rhs = np.vstack([
        Bd_dist @ dhat + c,
        H @ (yset - C_dist @ dhat)
    ])

    qs = Ginv @ rhs
    xs = qs[:dim_x].reshape(-1)
    us = qs[dim_x:].reshape(-1)

    return xs, us



