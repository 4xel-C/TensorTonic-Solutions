import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    x_array = np.asarray(x)

    result = dict()

    result["variance"] = float(np.sum((x_array - np.mean(x_array))**2) / (len(x_array) - 1))
    result["standard_deviation"] = float(np.sqrt(result["variance"]))
    return result