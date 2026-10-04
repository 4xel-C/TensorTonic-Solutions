import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """
    x = np.asarray(X)

    mu = np.mean(x, axis=axis, keepdims=True)
    sigma = np.std(x, axis=axis, keepdims=True) 

    return np.where(sigma <= eps, 0, (x - mu) / sigma)