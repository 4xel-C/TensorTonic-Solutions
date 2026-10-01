import numpy as np

def leaky_relu(x: list | float, alpha: float = 0.01) -> np.ndarray:
    """
    Returns elementwise Leaky ReLU values as a NumPy array matching the input shape.
    """

    x_array = np.asarray(x, dtype = float)
    
    result = x_array
    result[x_array <= 0] = x_array[x_array <= 0] * alpha
    
    # Write code here
    return result