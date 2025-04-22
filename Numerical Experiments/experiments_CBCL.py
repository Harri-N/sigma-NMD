
import sys
sys.path.insert(0, '..')

from scipy.io import loadmat
import numpy as np
import os
from SigNMD import sigma_NMD
from WLRA import WLRA
from eBCD_NMD import eBCD_NMD
from tqdm import tqdm
from utils import replace_with_nan

ranks = [2,5,10]
num_exp = 10
missing_percents = np.arange(0, 0.51, 0.1) 


print(f"Running experiments for CBCL dataset")
X0 = loadmat("../Data/CBCL.mat")['X']
X_filled = np.round(X0/np.max(X0))[:, 0::10]
m, n = X_filled.shape

for missing_percent in missing_percents:
    folder_path = f"CBCL/{int(missing_percent * 100)}%"
    os.makedirs(folder_path, exist_ok=True)
    X = replace_with_nan(X_filled, missing_percent)
    M = np.isnan(X).astype(float)
    missings = np.sum(M)/(m*n)
    ones = np.sum(X_filled*M)/np.sum(M) if np.sum(M)>0 else np.sum(X_filled)/(m*n)
    results = []
    
    for r in ranks:
        print(f"Running experiments for rank {r} with {missing_percent * 100:.0f}% missing values")

        method_results = {"sigma-NMD (TSVD init)": [], "sigma-NMD (Random init)": [], "eBCD-NMD (TSVD init)": [], "eBCD-NMD (Random init)": [], "TSVD": []}
        
        
        # TSVD
        if missing_percent == 0:
            u, s, vh = np.linalg.svd(X_filled, full_matrices=False)
            W_svd = u[:, :r] @ np.diag(np.sqrt(s[:r]))
            H_svd = np.diag(np.sqrt(s[:r])) @ vh[:r, :]
            WH = W_svd @ H_svd
            WH_bound = np.round(np.maximum(np.minimum(WH, 1), 0))
            rel_err =  np.sqrt(np.sum((X - WH)**2) /np.sum(X**2) )
            err_rate = np.nansum((X - WH_bound) ** 2) / (m*n)
            rmse = np.sqrt(np.sum((X - WH_bound)**2) /(m*n) )
            method_results["TSVD"].append((rel_err, err_rate, rmse,1))
        

            # sigma-NMD (TSVD init)
            W, H, rel_err_list, err_rate_list, rmse = sigma_NMD(X, r=r, init="tsvd", X_filled=X_filled)
            index_opt = np.argmin(np.nan_to_num(rel_err_list, nan=1e16))
            method_results["sigma-NMD (TSVD init)"].append((rel_err_list[index_opt], err_rate_list[index_opt], rmse[index_opt], len(rel_err_list)))

            # eBCD-NMD (TSVD init)
            Theta, rel_err_list, err_rate_list, rmse = eBCD_NMD(X, r, init="tsvd", X_filled=X_filled)
            index_opt = np.argmin(np.nan_to_num(rel_err_list, nan=1e16))
            method_results["eBCD-NMD (TSVD init)"].append((rel_err_list[index_opt], err_rate_list[index_opt], rmse[index_opt], len(rel_err_list)))
        
        for k in tqdm(range(num_exp)):
            W0 = np.random.rand(m, r)
            H0 = np.random.rand(r, n)
            
            # sigma-NMD (random init)
            W, H, rel_err_list, err_rate_list, rmse = sigma_NMD(X, r=r, W0=W0, H0=H0, X_filled=X_filled)
            index_opt = np.argmin(np.nan_to_num(rel_err_list, nan=1e16))
            method_results["sigma-NMD (Random init)"].append((rel_err_list[index_opt], err_rate_list[index_opt], rmse[index_opt], len(rel_err_list)))
            
            # eBCD-NMD (random init)
            Theta, rel_err_list, err_rate_list, rmse = eBCD_NMD(X, r, W0=W0, H0=H0, X_filled=X_filled)
            index_opt = np.argmin(np.nan_to_num(rel_err_list, nan=1e16))
            method_results["eBCD-NMD (Random init)"].append((rel_err_list[index_opt], err_rate_list[index_opt], rmse[index_opt], len(rel_err_list)))
            
            # TSVD
            if missing_percent != 0:
                W_svd,H_svd, rel_err_list, err_rate_list, rmse = WLRA(X, r=r, W0=W0, H0=H0.T, nonneg=False, X_filled=X_filled)
                index_opt = np.argmin(np.nan_to_num(rel_err_list, nan=1e16))
                method_results["TSVD"].append((rel_err_list[index_opt], err_rate_list[index_opt], rmse[index_opt], len(rel_err_list)))

                # sigma-NMD (TSVD init)
                W, H, rel_err_list, err_rate_list, rmse = sigma_NMD(X, r=r, W0=W_svd, H0=H_svd.T,X_filled=X_filled)
                index_opt = np.argmin(np.nan_to_num(rel_err_list, nan=1e16))
                method_results["sigma-NMD (TSVD init)"].append((rel_err_list[index_opt], err_rate_list[index_opt], rmse[index_opt], len(rel_err_list)))
                
                # eBCD-NMD (TSVD init)
                Theta, rel_err_list, err_rate_list, rmse = eBCD_NMD(X, r, W0=W_svd, H0=H_svd.T, X_filled=X_filled)
                index_opt = np.argmin(np.nan_to_num(rel_err_list, nan=1e16))
                method_results["eBCD-NMD (TSVD init)"].append((rel_err_list[index_opt], err_rate_list[index_opt], rmse[index_opt], len(rel_err_list)))
        
        # Compute mean results for each method
        for method in method_results:
            rel_errors = [res[0] for res in method_results[method]]
            err_rates = [res[1] for res in method_results[method]]
            rmses = [res[2] for res in method_results[method]]
            num_iterations = [res[3] for res in method_results[method]]
            results.append((missings, ones, r, method, np.mean(rel_errors), np.mean(err_rates), np.mean(rmses), np.mean(num_iterations), np.std(rel_errors, ddof=1) / np.sqrt(len(rel_errors)), np.std(err_rates, ddof=1) / np.sqrt(len(err_rates)), np.std(rmses, ddof=1) / np.sqrt(len(rmses)), np.std(num_iterations, ddof=1) / np.sqrt(len(num_iterations))))
    
        # Save results
        results_file = os.path.join(folder_path, "results.txt")
        with open(results_file, "a") as f:
            f.write("Missing %, 1's %, Rank, Method, Mean Relative Error, Mean Error Rate, Mean RMSE, Mean Iterations\n")
            for missings, ones, r, method, rel_err, err_rate, rmse, num_iter, sem_rel_err, sem_err_rate, sem_rmse, sem_num_iter in results:
                f.write(f"{missings:.2f}, {ones:.2f}, {r}, {method}, {rel_err:.6f} ± {sem_rel_err:.6f}, {err_rate:.6f} ± {sem_err_rate:.6f}, {rmse:.6f} ± {sem_rmse:.6f}, {num_iter:.2f} ± {sem_num_iter:.6f}\n")
