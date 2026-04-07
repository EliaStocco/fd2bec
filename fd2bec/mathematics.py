import numpy as np
from fd2bec import ATOL

def pseudo_inverse(A, tol=ATOL):
    U, S, Vh = np.linalg.svd(A, full_matrices=True)
    
    # Vectorized reciprocal with thresholding
    S_inv = np.where(np.abs(S) > tol, 1.0 / S, 0.0)
    
    A_inv = Vh.T @ np.diag(S_inv) @ U.T
    return A_inv