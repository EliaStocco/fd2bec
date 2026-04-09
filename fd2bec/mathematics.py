import numpy as np
from scipy.spatial import cKDTree
from fd2bec import SYMPREC
from typing import Tuple

def wrap(x:np.ndarray):
    return (x + 0.5) % 1.0 - 0.5

def find_mapping(a: np.ndarray, b: np.ndarray, atol: float = SYMPREC):
    """
    Map positions in b to nearest positions in a using minimum image convention.

    Returns:
        mapping: indices in a for each atom in b
        is_equal: all matches within tolerance
    """
    if a.shape != b.shape:
        return np.array([]), False

    tree = cKDTree(a)

    mapping = np.zeros(len(b), dtype=int)
    dists = np.zeros(len(b))

    for i, pos in enumerate(b):
        dist_vec = wrap(a - pos)
        dist = np.linalg.norm(dist_vec, axis=1)
        j = np.argmin(dist)

        mapping[i] = j
        dists[i] = dist[j]

    return mapping, np.all(dists <= atol)

def invert_indices(indices:np.ndarray,axis=None)->np.ndarray:
    """
    Given a list of indices that map atoms_A to atoms_B,
    returns the reverted indices that would restore atoms_A from atoms_B.
    """
    inverted_indices = np.argsort(indices,axis=axis)
    return inverted_indices