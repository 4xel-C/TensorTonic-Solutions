def gradient_descent_quadratic(
    a: float, b: float, c: float, x0: float, lr: float, steps: int
) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    # Write code here
    # initialize x
    x = x0

    for i in range(steps):
        # compute the gradient
        dx = (2 * a * x) + b

        x = x - lr * dx

    return x