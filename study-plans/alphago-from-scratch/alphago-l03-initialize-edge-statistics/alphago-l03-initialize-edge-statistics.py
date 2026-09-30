import numpy as np

def initialize_mcts_edges(policy_logits: np.ndarray, legal_mask: np.ndarray) -> dict:
    """
    Returns: N as an int64 array; W, Q, and P as float64 arrays in a dictionary.
    """
    # Initialize the parameters
    N = np.zeros_like(policy_logits, dtype = "int64")
    W = np.zeros_like(policy_logits, dtype = float)
    Q = np.zeros_like(policy_logits, dtype = float)
    P = np.zeros_like(policy_logits, dtype = float)

    # Use the mask to compute the logits
    true_logits = policy_logits[legal_mask]
    corrected_logits = true_logits - np.max(true_logits)

    # Compute the logits
    probabilities_masked = np.exp(corrected_logits) / np.sum(np.exp(corrected_logits)) 

    P[legal_mask] = probabilities_masked

    result = {
        "N": N,
        "W": W,
        "Q": Q,
        "P": P
    }
    
    return result
