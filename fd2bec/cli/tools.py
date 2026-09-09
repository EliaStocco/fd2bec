import numpy as np


def matrix_norm(matrix: np.ndarray):
    return np.linalg.norm(matrix, "fro") / np.sqrt(matrix.shape[0])
