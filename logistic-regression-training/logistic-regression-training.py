import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(
    X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000
) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Initialize weights and biases
    b = 0
    w = np.full(X.shape[1], 0)
    n = X.shape[0]

    # Initialize the BCE as infinity
    bce = np.inf

    # Compute gradient descent
    for i in range(steps):
        # Compute the first predictions
        z = X @ w + b
        p = _sigmoid(z)

        # compute the first value of the loss function (BCE) (Could be used to check loss stability for early stopping)
        bce = -(1 / n) * np.sum((y * np.log(p) + (1 - y) * np.log(1 - p)))

        # Compute the gradient the mean gradient using chain rule with logits
        dldz = p - y
        dldw = np.mean(X[:, :] * dldz[:, np.newaxis], axis=0)

        # Compute the gradient for the bias (chain rule, dz/db = 1)
        dldb = np.mean(dldz)

        # update the weights
        w = w - lr * dldw
        b = b - lr * dldb

    return (w, float(b))
