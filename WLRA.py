import numpy as np
import matplotlib.pyplot as plt

def WLRA(X, P=None, r=1, W0=None, H0=None, nonneg=False, lambda_=1e-6, tol=1e-4, maxiter=1e4, alpha=0.9999, X_filled=None, verbose=False):
    """
    This codes solves the weighted low-rank matrix approximation problem.
    Given X (m x n), a nonnegative weight matrix P (m x n), and r or an
    initial matrix W (mxr), it iteratively solves
    min_{W(mxr), H(nxr)}  ||X-WH^T||_P^2
                               + lambda (||W||_F^2+||H||_F^2),
    where ||X-WH^T||_P^2 = sum_{i,j} P(i,j) (X-WH^T)_{i,j}^2,
    using a block coordinate descent method (columns of W and H are the
    blocks, like in HALS for NMF).
    It is possible to requires (W,H) >= 0 using nonneg = 1.
    This codes is an adaptation of a Matlab file available here : https://gitlab.com/ngillis/nmfbook.git

    Args:
        X (numpy.ndarray): Input matrix of shape (m, n).
        P (numpy.ndarray, optional): nonnegative weight matrix (m, n).
        r (int, optional): Factorization rank r. Default: 1.
        W0 (numpy.ndarray, optional): Initial matrix W of shape (m, r). Default: None.
        H0 (numpy.ndarray, optional): Initial matrix H of shape (r, n). Default: None.
        nonneg (bool, optional): if nonneg = True: W>=0 and H>=0 (default=False)
        lambda (float, optional): penalization parameter (default=1e-6)
        tol (float, optional): Tolerance for convergence. Default: 1e-4.
        maxiter (int, optional): Maximum number of iterations. Default: 1e4.
        alpha (float, optional): Factor for early stopping. Default: 0.9999.
        X_filled (numpy.ndarray, optional): Ground truth matrix with filled missing values (for validation). Default: None.
        verbose (bool, optional): Print progress every 10 iterations. Default: False.

    Returns:
        W_opt (numpy.ndarray): Optimal matrix W of shape (m, r).
        H_opt (numpy.ndarray): Optimal matrix H of shape (r, n).
        relative_errors (list): List of relative errors at each iteration.
        error_rates (list): List of error rates at each iteration.
        rmse (list): List of RMSE values at each iteration.
    """
    if P is None:
        P = np.float64(~np.isnan(X)) 

    m, n = X.shape
    W = np.random.rand(m, r) if W0 is None else W0.copy()
    H = np.random.rand(n, r) if H0 is None else H0.copy()

    W, H = scaling_WH(W, H)
    W_opt, H_opt = W.copy(),H.copy()
    
    # M = Binary mask (1 for missing, 0 for observed)
    M = 1 - P
    X = np.nan_to_num(X)  # Replace NaN with 0

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
    
    relative_errors = []
    error_rates = []
    rmse = []

    # Compute initial error
    WH = W @ H.T
    WH_bound = np.round(np.maximum(np.minimum(WH, 1), 0))
    relative_errors.append(np.sqrt(np.sum(((X - WH) * P)**2) / rel_err0))
    if X_filled is not None and np.sum(M) > 0:
        error_rates.append(np.sum(((X_filled - WH_bound)*M)**2) / err_rate0)
        rmse.append(np.sqrt(np.sum(((X_filled - WH) * M)**2) / rmse0))
    else:
        error_rates.append(np.sum(((X - WH_bound)*P)**2) / err_rate0)
        rmse.append(np.sqrt(np.sum(((X - WH) * P)**2) / rmse0))

    R = X - W @ H.T
    
    iteration = 0
    # Main loop
    while True:
        R = X - W @ H.T
        for k in range(r):
            R += np.outer(W[:, k], H[:, k])
            Rp = R * P

            W[:, k] = (Rp @ H[:, k]) / (P @ (H[:, k] ** 2) + lambda_)
            if nonneg:
                W[:, k] = np.maximum(np.finfo(float).eps, W[:, k])

            H[:, k] = (Rp.T @ W[:, k]) / (P.T @ (W[:, k] ** 2) + lambda_)
            if nonneg:
                H[:, k] = np.maximum(np.finfo(float).eps, H[:, k])

            R -= np.outer(W[:, k], H[:, k])

        
        # Compute relative error
        WH = W @ H.T
        rel_err = np.sqrt(np.sum(((X - ((WH))) * P)**2) / rel_err0)
        
        # Update optimal matrices if relative error improves
        if rel_err < min(relative_errors):  
            W_opt, H_opt = W.copy(), H.copy()
        
        # Store errors
        relative_errors.append(rel_err)
        WH_bound = np.round(np.maximum(np.minimum(WH, 1), 0))
        if X_filled is not None and np.sum(M) > 0:
            error_rates.append(np.sum(((X_filled - WH_bound)*M)**2) / err_rate0)
            rmse.append(np.sqrt(np.sum(((X_filled - WH) * M)**2) / rmse0))
        else:
            error_rates.append(np.sum(((X - WH_bound)*P)**2) / err_rate0)
            rmse.append(np.sqrt(np.sum(((X - WH) * P)**2) / rmse0))
        

        W, H = scaling_WH(W, H)

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

def scaling_WH(W, H):
    m,r = W.shape
    norm_W = np.sqrt(np.sum(W ** 2, axis=0)) + 1e-16
    norm_H = np.sqrt(np.sum(H ** 2, axis=0)) + 1e-16

    for k in range(r):
        W[:, k] = W[:, k]/np.sqrt(norm_W[k])*np.sqrt(norm_H[k])
        H[:, k] = H[:, k]/np.sqrt(norm_H[k])*np.sqrt(norm_W[k])

    return W, H


