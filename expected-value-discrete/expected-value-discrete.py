import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    x_vec = np.asarray(x)
    p_vec = np.asarray(p)

    return float(x_vec @ p_vec)