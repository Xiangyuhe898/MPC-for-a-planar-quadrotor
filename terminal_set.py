import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from control import dare



def box_constraints(lb, ub):
    num_con = 2 * len(lb)
    
    A = np.kron(np.eye(len(lb)), [[1], [-1]])

    b = np.zeros(num_con)
    for i in range(num_con):
        b[i] = ub[i // 2] if i % 2 == 0 else -lb[i // 2]

    goodrows = np.logical_and(~np.isinf(b), ~np.isnan(b))
    A = A[goodrows]
    b = b[goodrows]
    
    return A, b

def compute_maximal_admissible_set(F, A, b, max_iter=100):
    '''
    Compute the maximal admissible set for the system x_{t+1} = F x_t subject to A x_t <= b.
    
    Note that if F is unstable, this procedure will not work.
    '''
    
    dim_con = A.shape[0]
    A_inf_hist = []
    b_inf_hist = []

    Ft = F
    A_inf = A
    b_inf = b
    A_inf_hist.append(A_inf)
    b_inf_hist.append(b_inf)

    for t in range(max_iter):
        #print(f"========== iteration {t} ==========")
        f_obj = A @ Ft
        stop_flag = True
        for i in range(dim_con):
            x = linprog(-f_obj[i], A_ub=A_inf, b_ub=b_inf, method="highs")["x"]
            # x = solve_qp(np.zeros((2, 2)), -f_obj[i], A_inf, b_inf, solver="") # Actually, this is not a QP, but a LP. It is better to use a LP solver.
            if f_obj[i] @ x > b[i]:
                stop_flag = False
                break

        if stop_flag:
            break
        
        A_inf = np.vstack((A_inf, A @ Ft))
        b_inf = np.hstack((b_inf, b))
        Ft = F @ Ft
        A_inf_hist.append(A_inf)
        b_inf_hist.append(b_inf)

    return A_inf_hist, b_inf_hist

def find_lqr_invariant_set(A, B, K, lb_x, ub_x, lb_u, ub_u,max_iter):
    A_x, b_x = box_constraints(lb_x, ub_x)
    A_u, b_u = box_constraints(lb_u, ub_u)

    A_lqr = A_u @ K
    b_lqr = b_u

    A_con = np.vstack((A_lqr, A_x))
    b_con = np.hstack((b_lqr, b_x))
    A_con = np.where(np.abs(A_con) < 1e-10, 0, A_con)
    

    F = A + B @ K
    eigvals = np.linalg.eigvals(F)
    
    A_inf_hist, b_inf_hist = compute_maximal_admissible_set(F, A_con, b_con,max_iter)

    return A_inf_hist, b_inf_hist 




def remove_redundant_constraints(A, b, x0=None, tol=1e-6):
    """
    Removes redundant constraints for the polyhedron Ax <= b.

    """
    # A = np.asarray(A)
    # b = np.asarray(b).flatten()
    
    if A.shape[0] != b.shape[0]:
        raise ValueError("A and b must have the same number of rows!")
    
    if tol is None:
        tol = 1e-8 * max(1, np.linalg.norm(b) / len(b))
    elif tol <= 0:
        raise ValueError("tol must be strictly positive!")
    
    A = np.where(np.abs(A) < tol, 0, A)
    b = np.where(np.abs(b) < tol, 0, b)

    # Remove zero rows in A
    Anorms = np.max(np.abs(A), axis=1)
    badrows = (Anorms == 0)
    if np.any(b[badrows] < 0):
        raise ValueError("A has infeasible trivial rows.")
        
    A = A[~badrows, :]
    b = b[~badrows]
    goodrows = np.concatenate(([0], np.where(~badrows)[0]))
        
    # Find an interior point if not supplied
    if x0 is None:
        if np.all(b > 0):
            x0 = np.zeros(A.shape[1])
        else:
            raise ValueError("Must supply an interior point!")
    else:
        x0 = np.asarray(x0).flatten()
        if x0.shape[0] != A.shape[1]:
            raise ValueError("x0 must have as many entries as A has columns.")
        if np.any(A @ x0 >= b - tol):
            raise ValueError("x0 is not in the strict interior of Ax <= b!")
            
    # Compute convex hull after projection
    # btilde = b - A @ x0
    # if np.any(btilde <= 0):
    #     print("Warning: Shifted b is not strictly positive. Convex hull may fail.")
    
    # Atilde = np.vstack((np.zeros((1, A.shape[1])), A / btilde[:, np.newaxis]))
    btilde = np.maximum(b - A @ x0, 1e-6)  # 防止爆炸
    if np.any(btilde <= 0):
        print("Warning: Shifted b is not strictly positive. Convex hull may fail.")
    
    Atilde = np.vstack((np.zeros((1, A.shape[1])), A / btilde[:, np.newaxis]))

    # hull = ConvexHull(Atilde) 
    hull =ConvexHull(Atilde, qhull_options='QJ Qx Qc')
    #hull = ConvexHull(Atilde, qhull_options='Q12 Qx Qc Qs')  
    u = np.unique(hull.vertices)    
    nr = goodrows[u]    
    h = goodrows[hull.simplices]
    
    # if nr[0] == 0:
    #     nr = nr[1:]
        
    Anr = A[nr, :]
    bnr = b[nr]
        
    return nr, Anr, bnr, h, x0


def compute_terminal_set(Ad, Bd, 
                         lb_x_tilde, ub_x_tilde, 
                         lb_u_tilde, ub_u_tilde, 
                         Q, R,
                         max_iter=50 ):
                         
    """
    Compute LQR terminal invariant set.
    
    Parameters:
        Ad, Bd       : Discrete-time linear system matrices
        lb_x_tilde   : Lower bounds on state deviation
        ub_x_tilde   : Upper bounds on state deviation
        lb_u_tilde   : Lower bounds on input deviation
        ub_u_tilde   : Upper bounds on input deviation
        Q, R         : Cost matrices for LQR
        max_iter     : Max iterations for invariant set computation

    Returns:
        A_inf, b_inf : Terminal constraint set A_inf x <= b_inf
    """
    # Solve DARE
    P_inf, _, K_inf = dare(Ad, Bd, Q, R)
    K = -K_inf  # LQR gain

    # Find invariant set
    A_inf_hist, b_inf_hist = find_lqr_invariant_set(
        Ad, Bd, K, lb_x_tilde, ub_x_tilde, lb_u_tilde, ub_u_tilde, max_iter
    )

    # Remove redundant constraints
    _, A_inf, b_inf, _, _ = remove_redundant_constraints(
        A_inf_hist[-1], b_inf_hist[-1]
    )

    return A_inf, b_inf, P_inf, K



