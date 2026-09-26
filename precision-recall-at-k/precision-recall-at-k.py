def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    precision = sum([item in recommended[:k] for item in relevant]) / k
    recall = sum([item in recommended[:k] for item in relevant]) / len(relevant)

    return [precision, recall]