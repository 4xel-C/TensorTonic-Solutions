import numpy as np

def percentiles(x: list, q: list) -> np.ndarray:
    """
    Returns a NumPy array of percentiles.
    """
    # Write code here
    x_arr = np.sort(np.asarray(x))
    q_arr = np.asarray(q)

    n = len(x_arr)

    # Compute the ranks
    r = (q_arr/100) * (n - 1)

    # compute the ceiled and floored ranks
    l = np.floor(r).astype(int)
    u = np.ceil(r).astype(int)

    # compute the weights between the ranks
    w = r - l

    # Interpolate between values
    p = (1 - w) * x_arr[l] + w * x_arr[u]

    return p