import sys
sys.path.insert(0, '..')

import numpy as np
import os
from SigNMD import sigma_NMD
from eBCD_NMD import eBCD_NMD
from tqdm import tqdm

datasets = ["zoo", "heart", "lymp", "apb"]
ranks = [2, 5, 10]
num_exp = 10

for data in datasets:
    print(f"Running experiments for {data} dataset")
    X = np.loadtxt(f"../Data/{data}.txt")
    m, n = X.shape


    folder_path = f"{data}"
    os.makedirs(folder_path, exist_ok=True)
    
    ones = np.sum(X)/(m*n)
    results = []
    
    for r in ranks:
        print(f"Running experiments for rank {r}")

        method_results = {"TSVD": [], "sigma-NMD (TSVD init)": [], "eBCD-NMD (TSVD init)": [], "sigma-NMD (Random init)": [], "eBCD-NMD (Random init)": []}
        
        
        # TSVD
        u, s, vh = np.linalg.svd(X, full_matrices=False)
        W_svd = u[:, :r] @ np.diag(np.sqrt(s[:r]))
        H_svd = np.diag(np.sqrt(s[:r])) @ vh[:r, :]
        WH = W_svd @ H_svd
        WH_bound = np.round(np.maximum(np.minimum(WH, 1), 0))
        rel_err =  np.sqrt(np.sum((X - WH)**2) /np.sum(X**2) )
        err_rate = np.nansum((X - WH_bound) ** 2) / (m*n)
        rmse = np.sqrt(np.sum((X - WH_bound)**2) /(m*n) )
        method_results["TSVD"].append((rel_err, err_rate, rmse,1))
    

        # sigma-NMD (TSVD init)
        W, H, rel_err_list, err_rate_list, rmse = sigma_NMD(X, r=r, init="tsvd")
        index_opt = np.argmin(np.nan_to_num(rel_err_list, nan=1e16))
        method_results["sigma-NMD (TSVD init)"].append((rel_err_list[index_opt], err_rate_list[index_opt], rmse[index_opt], len(rel_err_list)))
        
        # eBCD-NMD (TSVD init)
        Theta, rel_err_list, err_rate_list, rmse = eBCD_NMD(X, r=r, init="tsvd")
        index_opt = np.argmin(np.nan_to_num(rel_err_list, nan=1e16))
        method_results["eBCD-NMD (TSVD init)"].append((rel_err_list[index_opt], err_rate_list[index_opt], rmse[index_opt], len(rel_err_list)))
        
        for k in tqdm(range(num_exp)):
            W0 = np.random.rand(m, r)
            H0 = np.random.rand(r, n)
            
            # sigma-NMD (random init)
            W, H, rel_err_list, err_rate_list, rmse = sigma_NMD(X, r=r, W0=W0, H0=H0)
            index_opt = np.argmin(np.nan_to_num(rel_err_list, nan=1e16))
            method_results["sigma-NMD (Random init)"].append((rel_err_list[index_opt], err_rate_list[index_opt], rmse[index_opt], len(rel_err_list)))
            
            # eBCD-NMD (random init)
            Theta, rel_err_list, err_rate_list, rmse = eBCD_NMD(X, r=r, W0=W0, H0=H0)
            index_opt = np.argmin(np.nan_to_num(rel_err_list, nan=1e16))
            method_results["eBCD-NMD (Random init)"].append((rel_err_list[index_opt], err_rate_list[index_opt], rmse[index_opt], len(rel_err_list)))
            

        # Compute mean results for each method
        for method in method_results:
            rel_errors = [res[0] for res in method_results[method]]
            err_rates = [res[1] for res in method_results[method]]
            rmses = [res[2] for res in method_results[method]]
            num_iterations = [res[3] for res in method_results[method]]
            results.append((ones, r, method, np.mean(rel_errors), np.mean(err_rates), np.mean(rmses), np.mean(num_iterations)))
    
    # Save results
    results_file = os.path.join(folder_path, "results.txt")
    with open(results_file, "w") as f:
        f.write("1's %, Rank, Method, Mean Relative Error, Mean Error Rate, Mean RMSE, Mean Iterations\n")
        for ones, r, method, rel_err, err_rate, rmse, num_iter in results:
            f.write(f"{ones:.2f}, {r}, {method}, {rel_err:.6f}, {err_rate:.6f}, {rmse:.6f}, {num_iter:.2f}\n")
