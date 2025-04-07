# Non-linear Matrix Decomposition with the sigmoid function

This codes provides algorithm to solve the following Nonlinear Matrix Decomposition (NMD) problems with the sigmoid function:  
Given a matrix $X \in \mathbb{R}^{m \times n}$ and an integer $r$, solve  
$$
    \min_{W, H} \| X - \sigma(WH) \|_F^2, 
$$

where $W$ has $r$ columns, and $H$ has $r$ rows. The sigmoid function $\sigma(\cdot)$ is applied component-wise on $WH$ and is defined by :
        
$$f(z) = \sigma(z) = \frac{1}{1 + e^{-z}}.$$

This algorithm is described in the paper "Non-linear Matrix Decomposition with the sigmoid function" by Harrison Nguyen, Arnaud Vandaele, Atharva Awari, Nicolas Gillis, 2025.

You can run main.py for a simple example on synthetic data.

All the experiments comparing ReLU-NMD and TSVD with our method can be found in the folder "Numerical Experiments".
