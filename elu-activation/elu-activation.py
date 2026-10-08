import math

def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """
    # Write code here
    for i, z in enumerate(x):
        if z < 0:
            x[i] = alpha * (math.exp(z) - 1)
    
    
    return x