import numpy as np
from utils import sigmoid
import matplotlib.pyplot as plt
from SigNMD import sigma_NMD

# Define matrix dimensions and rank
m = 10  
n = 10  
r = 2   

# Generate synthetic data for testing
np.random.seed(42)
Wt = np.random.rand(m, r)  
Ht = np.random.rand(r, n)  
X = sigmoid(Wt @ Ht)

# Perform sigma-NMD
W, H, relative_errors, error_rates, rmse = sigma_NMD(X,r=r,verbose=True)

# Plot the relative error over iterations
plt.figure()
plt.semilogy(relative_errors)  
plt.xlabel('Iterations')
plt.ylabel('Relative Error') 
plt.show()
index_opt = np.argmin(np.nan_to_num(relative_errors, nan=1e16))
print(f"Final relative error: {relative_errors[index_opt]}")  
print(f"Final error rate: {error_rates[index_opt]}")  
print(f"Final RMSE: {rmse[index_opt]}")