import sys
sys.path.insert(0, '..')
import numpy as np
from numpy.linalg import norm, qr
from WLRA import WLRA

def eBCD_NMD(X, r, W0=None, H0=None, X_filled=None, init="random", maxit=1e4, tol=1e-4, tolerr=1e-4, display=0, param=None):
    """
    Extrapolated Block Coordinate Descent to solve the Non-linear Matrix Decomposition (NMD) problem with the ReLU function.
    The method is described in the paper "An extrapolated and provably convergent algorithm for nonlinear matrix decomposition with the ReLU function", 
    (Code avalaible here : https://github.com/giovanniseraghiti/ReLU-NMD.git).

    Args:
        X (np.ndarray): Matrice d'entrée creuse et non négative avec NaN pour les entrées manquantes.
        r (int): Rang cible de la décomposition.
        W0 (numpy.ndarray, optional): Initial matrix W of shape (m, r). Default: None.
        H0 (numpy.ndarray, optional): Initial matrix H of shape (r, n). Default: None.
        X_filled (numpy.ndarray, optional): Ground truth matrix with filled missing values (for validation). Default: None.
        init (int, optional): Initialization mode (random, tsvd). Default: random.
        tol (float, optional): Tolerance for convergence. Default: 1e-4.
        tolerr (float, optional): tolerance on 10 successive errors (err(i+1)-err(i-10)) Default: 1e-4.
        display   = if set to 1, it diplayes error along iterations Default: 0.
        param (dict, optional): structure, containing the parameter of the model

    Returns:
        Theta (np.ndarray): m-by-n matrix, approximate solution of min_{Theta}||X-max(0,Theta)||_F^2  s.t. rank(Theta)<=r.
        relative_errors (list): List of relative errors at each iteration. ||X-max(0,WH)||_F / || X ||_F
        error_rates (list): List of error rates at each iteration.
        rmse (list): List of RMSE values at each iteration.
    """
    # Initialization
    m, n = X.shape
    X = np.maximum(X, 0) 
    M = np.isnan(X).astype(float) #mask to compute error rate on missing values

    if np.all(M==0):
        idx = X==0
    else:
        idx = np.isnan(X)
    X = np.nan_to_num(X)
    
        
    normX = np.linalg.norm(np.nan_to_num(X),'fro')

    # Configuration of hyperparameters
    param = param or {}
    alpha = param.get("alpha", 1)
    alpha_max = param.get("alpha_max", 4)
    mu = param.get("mu", 0.3)
    delta_bar = param.get("delta_bar", 0.8)
    check = param.get("check", 0)

    # Define the initial variables
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
                    W_svd, H_svd, e1,e2,e3 = WLRA(X, ~idx, r, X_filled=X_filled, nonneg=False)
                else:
                    W_svd, H_svd, e1,e2,e3 = WLRA(X, ~idx, r, W0, H0.T, X_filled=X_filled, nonneg=False)
                W = W_svd if W0 is None else W0.copy()
                H = H_svd if H0 is None else H0.copy()
                H = H.T
    else:
        W = W0.copy()
        H = H0.copy()

    Z = np.nan_to_num(param.get("Z0", X.copy()), nan=0.0)  
    Theta = W @ H
    S = norm(Z - Theta, 'fro')

    # Initialize the error
    if np.all(M==0):
        err = [norm((Theta - Z)) / normX] if normX > 0 else [0]
    else: 
        err = [norm((Theta - Z)[~idx]) / normX] if normX > 0 else [0]
    rel_err0 = np.sum((X)**2)
    # if missing values are present and ground truth is available, 
    # we compute error_rate and rmse based on missing values
    if X_filled is not None and np.sum(M) > 0:
        err_rate0 = np.sum(M)  
        rmse0 = np.sum(M)
    else:
        err_rate0 = np.sum(1-M)
        rmse0 = np.sum(1-M)

    relative_errors,error_rates, rmse = [],[],[]
    WH = np.maximum(0,Theta)
    WH_bound = np.round(np.maximum(np.minimum(WH, 1), 0))
    relative_errors.append(np.sqrt(np.sum(((X - WH) * (1-M))**2) / rel_err0))
    if X_filled is not None and np.sum(M) > 0:
        error_rates.append(np.sum(((X_filled - WH_bound)*M)**2) / err_rate0)
        rmse.append(np.sqrt(np.sum(((X_filled - WH) * M)**2) / rmse0))
    else:
        error_rates.append(np.sum(((X - WH_bound)*(1-M))**2) / err_rate0)
        rmse.append(np.sqrt(np.sum(((X - WH) * (1-M))**2) / rmse0))

    if display:
        print("Running eBCD-NMD, evolution of [iteration number : relative error in %]")

    for i in range(int(maxit)):
        # Indirect extrapolation step
        Z_alpha = alpha * Z + (1 - alpha) * Theta

        # Update of W computing an orthogonal basis of Z_alpha*H^T
        Q, _ = qr(Z_alpha @ H.T, mode='reduced')
        if check and np.linalg.matrix_rank(Z_alpha @ H.T) < r:
            Q = Q[:, :np.linalg.matrix_rank(Z_alpha @ H.T)]
        W_new = Q

        # Update of H
        H_new = W_new.T @ Z_alpha

        # Approximation matrix
        Theta_new = W_new @ H_new

        # check on the boundness
        if check and np.max(np.abs(Theta_new)) > 1e10:
            print("Warning : The sequence might be unbounded.")

        # Update of Z
        Z_new = np.minimum(0, Theta_new * idx)
        Z_new = Z_new + np.nan_to_num(X)

        # Evaluating the residual
        S_new = norm(Z_new - Theta_new, 'fro')
        res_ratio = S_new / S
        WH = np.maximum(0,Theta_new)
        WH_bound = np.round(np.maximum(np.minimum(WH, 1), 0))
        relative_errors.append(np.sqrt(np.sum(((X - WH) * (1-M))**2) / rel_err0))
        if X_filled is not None and np.sum(M) > 0:
            error_rates.append(np.sum(((X_filled - WH_bound)*M)**2) / err_rate0)
            rmse.append(np.sqrt(np.sum(((X_filled - WH) * M)**2) / rmse0))
        else:
            error_rates.append(np.sum(((X - WH_bound)*(1-M))**2) / err_rate0)
            rmse.append(np.sqrt(np.sum(((X - WH) * (1-M))**2) / rmse0))

        # Adaptive strategy to select extrapolation parameter
        if res_ratio < 1:
            W, H, Z, Theta, S = W_new, H_new, Z_new, Theta_new, S_new
            if res_ratio > delta_bar:
                mu = max(mu, 0.25 * (alpha - 1))
                alpha = min(alpha + mu, alpha_max)
                alpha = 1 if alpha == alpha_max else alpha
        else:
            alpha = 1

        # Compute the relative residual 
        if np.all(M==0):
            obs_err = S/normX
        else:
            obs_err = norm((Theta - Z)[~idx]) / normX if normX > 0 else 0
        err.append(obs_err)

        # Stopping criteria on relative residual
        if obs_err < tol:
            if display: print(f"The algorithm has converged: ||Z-WH||/||X|| < {tol}")
            break

        # Stopping criteria if the residual is not reduced sufficiently in 10 iterations
        if i >= 10 and abs(err[-1] - err[-11]) < tolerr:
            if display: print(f"The algorithm has converged: rel. err.(i+1) - rel. err.(i+10) < {tolerr}")
            break

        # Display
        if display and i % 10 == 0:
            print(f"[{i:03d} : {100 * obs_err:.4f}%]")

    if display:
        print(f"Final relative error : {100 * err[-1]:.4f}% after {i+1} iterations")

    return Theta, relative_errors, error_rates, rmse