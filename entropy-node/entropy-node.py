import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    y_vec = np.asarray(y)
    values, counts = np.unique(y_vec, return_counts = True)

    proportions = counts / sum(counts)

    return -np.sum(proportions*np.log2(proportions))
    