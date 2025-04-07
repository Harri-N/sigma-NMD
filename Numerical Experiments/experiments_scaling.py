import sys
sys.path.insert(0, '..')

import numpy as np
from utils import sigmoid, set_size
import matplotlib.pyplot as plt
from SigNMD import sigma_NMD
import matplotlib
from tqdm import tqdm

m = 200
n = 200
r = 5
errors1 = np.zeros((201,))
errors2 = np.zeros((201,))

for i in tqdm(range(10)):
    
    Wt = np.random.rand(m,r)
    Ht = np.random.rand(r,n)
    X = sigmoid(Wt@Ht)
    
    # Initialize W and H matrices randomly
    W0 = np.random.rand(m,r)
    H0 = np.random.rand(r,n)

    W, H, relative_errors1, error_rates, rmse = sigma_NMD(X,r=r,W0=W0,H0=H0, maxiter=200, alpha=1, tol=1e-16)

    W, H, relative_errors2, error_rates, rmse = sigma_NMD(X,r=r,W0=W0,H0=H0, maxiter=200, alpha=1, scaling=False, tol=1e-16)
    
    errors1 += np.array(relative_errors1)
    errors2 += np.array(relative_errors2)

errors1 /= 10
errors2 /= 10


matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

fig, ax = plt.subplots(1, 1, figsize=set_size(250))
ax.semilogy(errors2)
ax.semilogy(errors1, linestyle='dashed')

plt.legend([r'$\sigma$-NMD w/o scaling', r'$\sigma$-NMD with scaling'])
ax.set_xlabel('Iterations')
ax.set_ylabel('Relative Error')
fig.tight_layout()
plt.show()
plt.savefig(f'Numerical Experiments/scaling_m{m}_n{n}_r{r}.pgf')