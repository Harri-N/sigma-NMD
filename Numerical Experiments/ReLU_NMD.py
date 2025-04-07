import numpy as np
from WLRA import WLRA

def relu(x):
    return np.maximum(0, x)

def ReLU_NMD(X, r=1, W0=None, H0=None, X_filled=None, init="random", maxiter=1e4, tol=1e-4, alpha=0.9999,verbose=False):
    """
    Solves the Non-linear Matrix Decomposition (NMD) problem with the ReLU function.
    This function decomposes an input matrix `X` of shape (m, n) into two factor matrices `W` and `H` using
    a coordinate descent method, minimizing the reconstruction error:
    `min_{W,H} ||X - ReLU(W @ H)||^2`,
    The algorithm supports missing values in `X` and updates `W` and `H` iteratively.
    The method is inspired by the paper "Coordinate Descent Algorithm for Nonlinear Matrix Decomposition with the ReLU function"
    (Code avalaible here : https://gitlab.com/Atharva05/coordinate-descent-for-relu-nmd.git).
    
    Args:
        X (numpy.ndarray): Input matrix of shape (m, n) with possible missing values (NaNs).
        r (int, optional): Rank of the decomposition. Default: 1.
        W0 (numpy.ndarray, optional): Initial matrix W of shape (m, r). Default: None.
        H0 (numpy.ndarray, optional): Initial matrix H of shape (r, n). Default: None.
        X_filled (numpy.ndarray, optional): Ground truth matrix with filled missing values (for validation). Default: None.
        init (int, optional): Initialization mode (random, tsvd). Default: random.
        maxiter (int, optional): Maximum number of iterations. Default: 1e4.
        tol (float, optional): Tolerance for convergence. Default: 1e-4.
        alpha (float, optional): Factor for early stopping. Default: 0.9999.
        verbose (bool, optional): Print progress every 10 iterations. Default: False.
    
    Returns:
        W_opt (numpy.ndarray): Optimal matrix W of shape (m, r).
        H_opt (numpy.ndarray): Optimal matrix H of shape (r, n).
        relative_errors (list): List of relative errors at each iteration.
        error_rates (list): List of error rates at each iteration.
        rmse (list): List of RMSE values at each iteration.
    """

    m, n = X.shape
    
    # Binary mask for missing values
    M = np.isnan(X).astype(float)  # 1 for missing, 0 for observed
    Obs = 1 - M  # 1 for observed, 0 for missing
    X = np.nan_to_num(X)  # Replace NaN with 0
    
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
                u, s, vh = np.linalg.svd(X, full_matrices=False)
                W = u[:, :r] @ np.diag(np.sqrt(s[:r])) if W0 is None else W0.copy()
                H = np.diag(np.sqrt(s[:r])) @ vh[:r, :] if H0 is None else H0.copy()
            
            # else, use the Weighted Low-Rank Approximation
            else:
                if W0 is None or H0 is None:
                    W_svd, H_svd, e1,e2,e3 = WLRA(X, Obs, r, X_filled=X_filled, nonneg=False)
                else:
                    W_svd, H_svd, e1,e2,e3 = WLRA(X, Obs, r, W0, H0.T, X_filled=X_filled, nonneg=False)
                W = W_svd if W0 is None else W0.copy()
                H = H_svd if H0 is None else H0.copy()
                H = H.T
    else:
        W = W0.copy()
        H = H0.copy()

    # Track optimal W and H
    W_opt, H_opt = W.copy(), H.copy()
    
    # Scale W, H
    Z = relu(W @ H)
    optscal = np.sum(Z * X * Obs) / np.sum((Z*Obs) ** 2)
    W *= optscal

    # Compute denominator of relative error, error rate and RMSE
    rel_err0 = np.sum((X)**2)
    # if missing values are present and ground truth is available, 
    # we compute error_rate and rmse based on missing values
    if X_filled is not None and np.sum(M) > 0:
        err_rate0 = np.sum(M)  
        rmse0 = np.sum(M)
    else:
        err_rate0 = np.sum(Obs)
        rmse0 = np.sum(Obs)
    
    relative_errors = []
    error_rates = []
    rmse = []
    
    # Compute initial errors
    WH = relu(W @ H)
    WH_bound = np.round(np.maximum(np.minimum(WH, 1), 0))
    relative_errors.append(np.sqrt(np.sum(((X - WH) * Obs)**2) / rel_err0))
    if X_filled is not None and np.sum(M) > 0:
        error_rates.append(np.sum(((X_filled - WH_bound)*M)**2) / err_rate0)
        rmse.append(np.sqrt(np.sum(((X_filled - WH) * M)**2) / rmse0))
    else:
        error_rates.append(np.sum(((X - WH_bound)*Obs)**2) / err_rate0)
        rmse.append(np.sqrt(np.sum(((X - WH) * Obs)**2) / rmse0))
    
    iteration = 0
    
    # Main loop
    while True:

        # Update H and W alternately
        H = updateH(X, W, H, Obs)
        W = updateH(X.T, H.T, W.T, Obs.T).T
        
        # Compute relative error
        WH = relu(W @ H)
        rel_err = np.sqrt(np.sum(((X - WH) * Obs) ** 2) / rel_err0)

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
            error_rates.append(np.sum(((X - WH_bound)*Obs)**2) / err_rate0)
            rmse.append(np.sqrt(np.sum(((X - WH) * Obs)**2) / rmse0))
        

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

def updateH(X, W, H, Obs):
    r, n = H.shape
    WH = W @ H
    
    for j in range(n):
        c = X[:, j]
        for i in range(r):
            a = W[:, i]
            b = WH[:, j] - W[:, i] * H[i, j]
            Hij_old = H[i, j]
            H[i, j] = findmin(a, b, c, Obs[:, j])
            WH[:, j] += W[:, i] * (H[i, j] - Hij_old)
    
    return H

def findmin(a, b, c, Obs):

    # Keep values for observed entries
    obs_mask = Obs.astype(bool)
    a,b,c = a[obs_mask],b[obs_mask],c[obs_mask]

    pos = np.abs(a) > 1e-16
    a, b, c= a[pos], b[pos], c[pos]

    if len(a)==0:
        return 0
    
    breakpts = -b / a
    breakpts, ind = np.sort(breakpts), np.argsort(breakpts)
    a, b, c = a[ind], b[ind], c[ind]
    
    cc = c ** 2
    bb = b ** 2
    aa = a ** 2
    bc = b * c
    ac = a * c
    ab = a * b
    
    neg = a < -1e-16
    ti = np.sum(cc) - 2 * np.sum(bc[neg]) + np.sum(bb[neg])
    tl = 2 * (np.sum(ab[neg]) - np.sum(ac[neg]))
    tq = np.sum(aa[neg])
    
    xmin = -tl / (2 * tq)
    xopt = xmin if xmin < breakpts[0] else breakpts[0]
    yopt = ti + tl * xopt + tq * xopt ** 2
    
    for i in range(len(breakpts)):
        ti += np.sign(a[i]) * (bb[i] - 2 * bc[i])
        tl += np.sign(a[i]) * (2 * ab[i] - 2 * ac[i])
        tq += np.sign(a[i]) * aa[i]
        xmin = -tl / (2 * tq)
        xt = xmin if (i + 1 < len(breakpts) and breakpts[i] < xmin < breakpts[i + 1]) else breakpts[i]
        if ti + tl * xt + tq * xt ** 2 < yopt:
            xopt = xt
            yopt = ti + tl * xopt + tq * xopt ** 2
    
    return xopt