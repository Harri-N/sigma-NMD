import numpy as np

def sigmoid(x):
    """
    Computes the sigmoid function element-wise.

    Args:
        x (numpy.ndarray): Input array.
    
    Returns:
        numpy.ndarray: Sigmoid-transformed array.
    """
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
  """
  Computes the derivative of the sigmoid function element-wise.

  Args:
      x (numpy.ndarray): Input array.

  Returns:
      numpy.ndarray: Derivative of the sigmoid function.
  """
  return sigmoid(x)*(np.ones(x.shape)-sigmoid(x))

def fct_opti(A,b,x):
  """
  Computes the function value and gradient for the least squares optimization with sigmoid function.

  Args:
      A (numpy.ndarray): Coefficient matrix.
      b (numpy.ndarray): Target vector.
      x (numpy.ndarray): Current parameter estimate.

  Returns:
      tuple: Function value and gradient.
  """
  f = np.linalg.norm(sigmoid(A@x)-b,2)
  g = 2*A.T@((sigmoid(A@x)-b)*sigmoid_prime(A@x))
  return f,g

def w1(fct,A,b,x,d,alpha,beta1):
  """
  Checks the first Amijo-Wolfe condition.

  Args:
      fct (function): Objective function.
      A (numpy.ndarray): Coefficient matrix.
      b (numpy.ndarray): Target vector.
      x (numpy.ndarray): Current parameter estimate.
      d (numpy.ndarray): Search direction.
      alpha (float): Step size.
      beta1 (float): Wolfe condition parameter.
  
  Returns:
      bool: True if condition is satisfied, False otherwise.
  """
  f0, g0 = fct(A,b,x)
  f1, g1 = fct(A,b,x+alpha*d)
  return f1 <= f0 + alpha*beta1*d@g0

def w2(fct,A,b,x,d,alpha,beta2):
  """
  Checks the second Wolfe condition.

  Args:
      fct (function): Objective function.
      A (numpy.ndarray): Coefficient matrix.
      b (numpy.ndarray): Target vector.
      x (numpy.ndarray): Current parameter estimate.
      d (numpy.ndarray): Search direction.
      alpha (float): Step size.
      beta2 (float): Wolfe condition parameter.
  
  Returns:
      bool: True if condition is satisfied, False otherwise.
  """
  f0, g0 = fct(A,b,x)
  f1, g1 = fct(A,b,x+alpha*d)
  return d@g1 >= beta2*d@g0

def compute_step_length(fct,A,b,x,d,alpha,beta1=0.0001,beta2=0.9):
  """
  Performs a bisection search to find a step size that satisfies the Amijo-Wolfe conditions.

  Args:
      fct (function): Objective function.
      A (numpy.ndarray): Coefficient matrix.
      b (numpy.ndarray): Target vector.
      x (numpy.ndarray): Current parameter estimate.
      d (numpy.ndarray): Search direction.
      alpha (float): Initial step size.
      beta1 (float, optional): Wolfe condition parameter. Default is 0.0001.
      beta2 (float, optional): Wolfe condition parameter. Default is 0.9.
  
  Returns:
      float: Step size satisfying Wolfe conditions.
  """
  aleft = 0
  aright = np.inf

  while True:

    if w1(fct,A,b,x,d,alpha,beta1) and w2(fct,A,b,x,d,alpha,beta2):
      break

    if not w1(fct,A,b,x,d,alpha,beta1):
      aright = alpha
      alpha = (aleft+aright)/2
    elif not w2(fct,A,b,x,d,alpha,beta2):
      aleft = alpha 
      if aright<np.inf:
        alpha = (aleft+aright)/2
      else : 
        alpha = 2*alpha 

  return alpha


def replace_with_nan(matrix, percentage):
    """
    Replaces a given percentage of elements in a matrix with NaN.

    Args:
        matrix (numpy.ndarray): Input matrix.
        percentage (float): Fraction of elements to replace (0 to 1).
    
    Returns:
        numpy.ndarray: Modified matrix with NaN values.
    """
    if not (0 <= percentage <= 1):
        raise ValueError("Percent between 0 et 1.")
    modified_matrix = matrix.copy()
    modified_matrix = modified_matrix.astype(float)
    total_elements = matrix.size
    num_elements_to_replace = int(total_elements * percentage)
    indices = np.random.choice(total_elements, num_elements_to_replace, replace=False)
    multi_dim_indices = np.unravel_index(indices, matrix.shape)
    modified_matrix[multi_dim_indices] = np.nan

    return modified_matrix



def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """
    Sets figure dimensions for document compatibility.

    Args:
        width_pt (float): Document width in points.
        fraction (float, optional): Fraction of width to occupy. Default is 1.
        subplots (tuple, optional): (rows, cols) of subplots. Default is (1,1).
    
    Returns:
        tuple: Figure dimensions in inches.
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.55 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


