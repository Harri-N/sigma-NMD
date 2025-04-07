import sys
sys.path.insert(0, '..')

import numpy as np
import os
from SigNMD import sigma_NMD
from ReLU_NMD import ReLU_NMD
from WLRA import WLRA
from tqdm import tqdm

matrix_sizes = range(5, 30, 5)
ranks = [3, 4, 5]
num_exp = 100

results_folder = "identity"
os.makedirs(results_folder, exist_ok=True)

for matrix_size in matrix_sizes:
    print(f"Running experiments for identity matrix of size {matrix_size}")
    identity_matrix = np.identity(matrix_size)
    results = []
    
    for r in ranks:
        print(f"Running experiments for rank {r}")
        method_results = {"TSVD": [], "sigma-NMD (TSVD init)": [], "ReLU-NMD (TSVD init)": [], "sigma-NMD (Random init)": [], "ReLU-NMD (Random init)": [], "NMF": []}
        
        # TSVD
        u, s, vh = np.linalg.svd(identity_matrix, full_matrices=False)
        W_svd = u[:, :r] @ np.diag(np.sqrt(s[:r]))
        H_svd = np.diag(np.sqrt(s[:r])) @ vh[:r, :]
        WH = W_svd @ H_svd
        WH_bound = np.round(np.maximum(np.minimum(WH, 1), 0))
        err_rate = np.sum((identity_matrix - WH_bound) ** 2) / (matrix_size ** 2)
        if err_rate == 0:
            method_results["TSVD"].append(100)
        else:
            method_results["TSVD"].append(0)

        # sigma-NMD (TSVD init)
        W, H, rel_err_list, err_rate_list, _ = sigma_NMD(identity_matrix, r=r, init="tsvd")
        index_opt = np.argmin(np.nan_to_num(rel_err_list, nan=1e16))
        if err_rate_list[index_opt] == 0:
            method_results["sigma-NMD (TSVD init)"].append(100)
        else:
            method_results["sigma-NMD (TSVD init)"].append(0)
        
        # ReLU-NMD (TSVD init)
        W, H, rel_err_list, err_rate_list, _ = ReLU_NMD(identity_matrix, r=r, init="tsvd")
        index_opt = np.argmin(np.nan_to_num(rel_err_list, nan=1e16))
        if err_rate_list[index_opt] == 0:
            method_results["ReLU-NMD (TSVD init)"].append(100)
        else:
            method_results["ReLU-NMD (TSVD init)"].append(0)
        
        for k in tqdm(range(num_exp)):
            W0 = np.random.rand(matrix_size, r)
            H0 = np.random.rand(r, matrix_size)
            
            # sigma-NMD (random init)
            W, H, rel_err_list, err_rate_list, _ = sigma_NMD(identity_matrix, r=r, W0=W0, H0=H0)
            index_opt = np.argmin(np.nan_to_num(rel_err_list, nan=1e16))
            method_results["sigma-NMD (Random init)"].append(err_rate_list[index_opt] == 0)
            
            # ReLU-NMD (random init)
            W, H, rel_err_list, err_rate_list, _ = ReLU_NMD(identity_matrix, r=r, W0=W0, H0=H0)
            index_opt = np.argmin(np.nan_to_num(rel_err_list, nan=1e16))
            method_results["ReLU-NMD (Random init)"].append(err_rate_list[index_opt] == 0)
            
            # NMF
            W, H, rel_err_list, err_rate_list, _ = WLRA(identity_matrix, r=r, W0=W0, H0=H0.T, nonneg=True)
            index_opt = np.argmin(np.nan_to_num(rel_err_list, nan=1e16))
            method_results["NMF"].append(err_rate_list[index_opt] == 0)
        
        for method in method_results:
            zero_error_count = sum(method_results[method])
            results.append((matrix_size, r, method, zero_error_count))
    
    results_file = os.path.join(results_folder, "results.txt")
    with open(results_file, "a") as f:
        f.write("Matrix Size, Rank, Method, Zero Error Count\n")
        for matrix_size, r, method, zero_error_count in results:
            f.write(f"{matrix_size}, {r}, {method}, {zero_error_count}\n")
