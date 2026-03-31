import numpy as np
from scipy.optimize import linear_sum_assignment
from fd2bec import SYMPREC
from fd2bec.tools import wrap

def equal_rows_hungarian(a: np.ndarray, b: np.ndarray, atol=SYMPREC) -> bool:
    if a.shape != b.shape:
        return False

    # Compute pairwise distance matrix (m x m)
    dist =a[:, None, :] - b[None, :, :]
    dist = wrap(dist)
    cost = np.linalg.norm(dist, axis=2)

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost)

    # Check if all matched distances are within tolerance
    out = cost[row_ind, col_ind]
    return np.all(out <= atol)