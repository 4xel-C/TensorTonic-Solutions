import numpy as np


def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here

    w_vec = np.asarray(w)
    g_vec = np.asarray(g)
    s_vec = np.asarray(s)

    # compute the updated s
    new_s = beta * s_vec + (1 - beta) * g_vec**2

    # compute the new weights
    new_w = w_vec - (lr * g_vec / (np.sqrt(new_s) + eps))

    return list(new_w), list(new_s)