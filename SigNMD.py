import numpy as np
from utils import compute_step_length, fct_opti, sigmoid
from WLRA import WLRA

def sigma_NMD(X,r=1,W0=None,H0=None,X_filled=None,init="random",maxiter=1e4,tol=1e-4, alpha=0.9999, scaling=True, verbose=False):
    """
    This function solves the Non-linear Matrix Decomposition (NMD) problem with sigmoid function.
    Given an input matrix `X` of shape (m, n), a rank `r`, and optional initial matrices `W0` and `H0`,
    it iteratively solves the optimization problem:
    `min_{W,H} ||X - sigmoid(W @ H)||^2`,
    using a block coordinate descent method (columns of W and H are the blocks, like in HALS for NMF).
    It supports missing values in `X`.

    Args:
        X (numpy.ndarray): Input matrix of shape (m, n).
        r (int, optional): Rank of the decomposition. Default: 1.
        W0 (numpy.ndarray, optional): Initial matrix W of shape (m, r). Default: None.
        H0 (numpy.ndarray, optional): Initial matrix H of shape (r, n). Default: None.
        X_filled (numpy.ndarray, optional): Ground truth matrix with filled missing values (for validation). Default: None.
        init (int, optional): Initialization mode (random, tsvd). Default: random.
        maxiter (int, optional): Maximum number of iterations. Default: 1e4.
        tol (float, optional): Tolerance for convergence. Default: 1e-4.
        alpha (float, optional): Factor for early stopping. Default: 0.9999.
        scaling (bool, optional): Apply pre/post-scaling for stability. Default: True.
        verbose (bool, optional): Print progress every 10 iterations. Default: False.

    Returns:
        W_opt (numpy.ndarray): Optimal matrix W of shape (m, r).
        H_opt (numpy.ndarray): Optimal matrix H of shape (r, n).
        relative_errors (list): List of relative errors at each iteration. ||X-sigma(WH)||_F / || X ||_F
        error_rates (list): List of error rates at each iteration.
        rmse (list): List of RMSE values at each iteration.

    """
    m,n = X.shape

    # Binary mask
    M = np.isnan(X).astype(float)  # 1 for missing, 0 for observed
    P = 1-M # 1 for observed, 0 for missing
    X = np.nan_to_num(X)  # Replace NaN with 0 for computation

    # Initialize W and H matrices
    if W0 is None or H0 is None:

        # Random initialization
        if init == "random":
            W = np.random.rand(m, r) if W0 is None else W0.copy()
            H = np.random.rand(r, n) if H0 is None else H0.copy()

        # TSVD-based initialization
        elif init == "tsvd":
            # If missing values, use the truncated singular value decomposition of numpy
            if np.all(M==0):
                u, s, vh = np.linalg.svd(2*X-1, full_matrices=False)
                W = u[:, :r] @ np.diag(np.sqrt(s[:r])) if W0 is None else W0.copy()
                H = np.diag(np.sqrt(s[:r])) @ vh[:r, :] if H0 is None else H0.copy()
            
            # else, use the Weighted Low-Rank Approximation
            else:
                if W0 is None or H0 is None:
                    W_svd, H_svd, e1,e2,e3 = WLRA(2*X-1, P, r, X_filled=X_filled, nonneg=False)
                else:
                    W_svd, H_svd, e1,e2,e3 = WLRA(2*X-1, P, r, W0, H0.T, X_filled=X_filled, nonneg=False)
                W = W_svd if W0 is None else W0.copy()
                H = H_svd if H0 is None else H0.copy()
                H = H.T
    else:
        W = W0.copy()
        H = H0.copy()


    # Track optimal W and H
    W_opt, H_opt = W.copy(), H.copy()  
    
    # Compute denominator of relative error, error rate and RMSE
    rel_err0 = np.sum((X)**2)
    # if missing values are present and ground truth is available, 
    # we compute error_rate and rmse based on missing values
    if X_filled is not None and np.sum(M) > 0:
        err_rate0 = np.sum(M)  
        rmse0 = np.sum(M)
    else:
        err_rate0 = np.sum(P)
        rmse0 = np.sum(P)
        
    
    relative_errors, error_rates, rmse = [], [], []
    learning_rates_W = np.ones(W.shape[0]) * 0.9
    learning_rates_H = np.ones(H.shape[1]) * 0.9
    iteration = 0
    
    # Compute initial error
    relative_errors.append(np.sqrt(np.sum(((X - (sigmoid(W @ H))) * P)**2) / rel_err0))
    if X_filled is not None and np.sum(M) > 0:
        error_rates.append(np.sum(((X_filled - np.round(sigmoid(W @ H)))*M)**2) / err_rate0)
        rmse.append(np.sqrt(np.sum(((X_filled - (sigmoid(W @ H))) * M)**2) / rmse0))
    else:
        error_rates.append(np.sum(((X - np.round(sigmoid(W @ H)))*P)**2) / err_rate0)
        rmse.append(np.sqrt(np.sum(((X - (sigmoid(W @ H))) * P)**2) / rmse0))
    
    # Main loop
    while True:
        
        # Update H and W alternately
        H, learning_rates_H = updateH(X, W, H, learning_rates_H, P, scaling)
        WT, learning_rates_W = updateH(X.T, H.T, W.T, learning_rates_W, P.T, scaling)
        W = WT.T

        # Compute relative error
        rel_err = np.sqrt(np.sum(((X - (sigmoid(W @ H))) * P)**2) / rel_err0)

        # Update optimal matrices if relative error improves
        if rel_err < min(relative_errors):  
            W_opt, H_opt = W.copy(), H.copy()
        
        # Store errors
        relative_errors.append(rel_err)
        if X_filled is not None and np.sum(M) > 0:
            error_rates.append(np.sum(((X_filled - np.round(sigmoid(W @ H)))*M)**2) / err_rate0)
            rmse.append(np.sqrt(np.sum(((X_filled - (sigmoid(W @ H))) * M)**2) / rmse0))
        else:
            error_rates.append(np.sum(((X - np.round(sigmoid(W @ H)))*P)**2) / err_rate0)
            rmse.append(np.sqrt(np.sum(((X - (sigmoid(W @ H))) * P)**2) / rmse0))
        
        # Check for convergence
        if rel_err <= tol:
            break

        # Check for maximum iterations
        iteration += 1
        if maxiter is not None and iteration >= maxiter:
            break

        # Check if relative error has not improved by alpha factor in last 10 iterations
        if iteration % 10 == 0:
            if verbose:
                print(f"iteration {iteration}: relative error={rel_err}")
            if min(relative_errors[-10:]) > min(relative_errors[:-10]) * alpha:
                break
           
    return W_opt, H_opt, relative_errors, error_rates, rmse


def sigmoid_least_squares(A, b, x, alpha, P):
    """
    This function performs a least squares optimization with sigmoid function.
    Given a matrix `A` of shape (m, r), a vector `b` of shape (m,), a vector `x` of shape (r,),
    it iteratively solves the optimization problem:
    `min_x ||sigmoid(A @ x) - b||^2`.

    Args:
        A (numpy.ndarray): Input matrix of shape (m, r).
        b (numpy.ndarray): Target vector of shape (m,).
        x (numpy.ndarray): Initial vector of shape (r,).
        alpha (float): Initial step size.
        P (numpy.ndarray): Binary mask (1 for observed, 0 for missing), shape (m,).

    Returns:
        x (numpy.ndarray): Optimized vector of shape (r,).
        alpha (float): Optimized step size.

    """
    m, r = A.shape
    Obs_b = P > 0  # Filter elements based on P
    A = A[Obs_b, :]
    b = b[Obs_b]
    f, g = fct_opti(A, b, x)  # Compute function value and gradient
    d = -g  # Set direction to negative gradient
    if np.sum(np.abs(g)) > 10**(-6):  # Check convergence
        alpha = compute_step_length(fct_opti, A, b, x, d, alpha)  # Update step size
        x = x + alpha * d  # Update x
    return x, alpha


def updateH(X,W,H,learning_rates, P,scaling):
    """
    This function updates the matrix H using a block coordinate descent method.
    Given an input matrix `X` of shape (m, n), a matrix `W` of shape (m, r),
    a matrix `H` of shape (r, n), it updates each column of H using least squares optimization.

    Args:
        X (numpy.ndarray): Input matrix of shape (m, n).
        W (numpy.ndarray): Matrix W of shape (m, r).
        H (numpy.ndarray): Matrix H of shape (r, n).
        learning_rates (numpy.ndarray): Learning rates for each column of H.
        P (numpy.ndarray): Binary mask (1 for observed, 0 for missing), shape (m, n).
        scaling (bool): Apply pre/post-scaling for stability.
    
    Returns:
        H (numpy.ndarray): Updated matrix H of shape (r, n).
        learning_rates (numpy.ndarray): Updated learning rates for each column of H.

    """
    r, n = H.shape
    sc = np.zeros((r,))
    Wn = np.zeros(W.shape)

    # Pre-scaling step for numerical stability
    for i in range(r):
        sc[i] = np.linalg.norm(W[:, i], 2) if scaling else 1  # Compute norm of each column in W
        Wn[:, i] = W[:, i] / sc[i]  # Normalize W
        for j in range(n):
            H[i, j] *= sc[i]  # Scale H accordingly

    # Update each column of H
    for j in range(n):        
        
        # Update H using least squares optimization
        H[:, j], learning_rates[j] = sigmoid_least_squares(A=Wn, b=X[:, j], x=H[:, j], alpha=learning_rates[j], P=P[:, j])

        # Post-scaling step to restore original scale
        for i in range(r):
            H[i, j] /= sc[i]
    
    return H, learning_rates