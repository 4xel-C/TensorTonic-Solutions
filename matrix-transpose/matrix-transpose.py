import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """

    # Initializa the matrix to 0
    transposed_matrix = [[0] * len(A) for _ in range(len(A[0]))]

    # Write code here
    for row in range(len(A)):
        for col in range(len(A[0])):
            transposed_matrix[col][row] = A[row][col]

    return np.asarray(transposed_matrix)
