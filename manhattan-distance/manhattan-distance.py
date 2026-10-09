import numpy as np

def manhattan_distance(x: list, y: list) -> float:
    """
    Returns the Manhattan distance as a Python float.
    """
    # Write code here
    x_mat = np.asarray(x, dtype = float)
    y_mat = np.asarray(y, dtype = float)

    return np.sum(np.abs(x_mat - y_mat))