def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """

    param_vec = np.asarray(param)
    grad_vec = np.asarray(grad)
    m_vec = np.asarray(m)
    v_vec = np.asarray(v)

    # compute the first moment
    m_vec = beta1 * m_vec + (1 - beta1) * grad_vec

    # compute the second moment
    v_vec = beta2 * v_vec + (1 - beta2) * grad_vec**2

    # bias correction
    m_corrected = m_vec / (1 - beta1**t)
    v_corrected = v_vec / (1 - beta2**t)

    # update the parameters
    param_vec = param_vec - lr * (m_corrected / (np.sqrt(v_corrected) + eps))

    return (param_vec, m_vec, v_vec)