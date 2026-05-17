from typing import List, Tuple

import numpy as np

from fd2bec import ATOL


def wrap(x: np.ndarray):
    return (x + 0.5) % 1.0 - 0.5


def find_mapping(
    a: np.ndarray, b: np.ndarray, atol: float = ATOL, pbc: bool = False
) -> Tuple[np.ndarray, bool, np.ndarray]:
    """
    Map positions in b to nearest positions in a using minimum image convention.

    Returns:
        mapping: indices in a for each atom in b
        is_equal: all matches within tolerance
    """
    if a.shape != b.shape:
        return np.array([]), False

    # tree = cKDTree(a)

    mapping = np.zeros(len(b), dtype=int)
    dists = np.zeros(len(b))
    # n_min = np.zeros(len(b), dtype=int)
    for i, pos in enumerate(b):
        dist_vec = wrap(a - pos) if pbc else a - pos
        dist = np.linalg.norm(dist_vec, axis=1)
        j = np.argmin(dist)
        # n_min[i] = np.sum(dist >= dist[j] - 0.1)
        mapping[i] = j
        dists[i] = dist[j]
    ok = np.all(dists <= atol)
    if not np.all(np.sort(mapping) == np.arange(len(mapping))):
        raise ValueError(f"Invalid mapping: {mapping.tolist()}")
    # if not ok:
    #     d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    #     pass
    return mapping, ok, dists


def invert_indices(indices: np.ndarray, axis=None) -> np.ndarray:
    """
    Given a list of indices that map atoms_A to atoms_B,
    returns the reverted indices that would restore atoms_A from atoms_B.
    """
    inverted_indices = np.argsort(indices, axis=axis)
    return inverted_indices


def append_one(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Append a slice of ones (1.) along the given axis."""
    shape = list(x.shape)
    shape[axis] = 1
    ones = np.ones(shape, dtype=x.dtype)
    return np.concatenate([x, ones], axis=axis)


def remove_one(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Remove the last slice along the given axis (inverse of append_one)."""
    index = [slice(None)] * x.ndim
    index[axis] = slice(0, -1)
    return x[tuple(index)]


def affine2homogeneous(R_flat: np.ndarray, T_flat: np.ndarray) -> np.ndarray:
    """
    Convert a batch of affine transformations into homogeneous matrices.

    Parameters
    ----------
    R_flat : np.ndarray
        Array of shape (Nops, dim, dim)
    T_flat : np.ndarray
        Array of shape (Nops, dim)

    Returns
    -------
    H_ops : np.ndarray
        Array of shape (Nops, dim+1, dim+1) containing homogeneous matrices.
    """
    R_flat = np.asarray(R_flat)
    T_flat = np.asarray(T_flat)

    if R_flat.ndim != 3:
        raise ValueError("R_flat must have shape (Nops, dim, dim).")
    if T_flat.ndim != 2:
        raise ValueError("T_flat must have shape (Nops, dim).")

    Nops, dim, _ = R_flat.shape

    if T_flat.shape != (Nops, dim):
        raise ValueError("T_flat must have shape (Nops, dim).")

    H_ops = np.zeros((Nops, dim + 1, dim + 1))

    for i in range(Nops):
        H = np.eye(dim + 1)
        H[:dim, :dim] = R_flat[i]
        H[:dim, dim] = T_flat[i]
        H_ops[i] = H

    return H_ops


def homogeneous2affine(H: np.ndarray, tol=ATOL) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a homogeneous transformation matrix (or a batch of them),
    returns the rotation matrix R and translation vector t.

    Supports:
    - Single matrix of shape (n+1, n+1)
    - Batch of matrices of shape (Nops, n+1, n+1)

    Returns
    -------
    For single input:
        R : (n, n)
        t : (n,)

    For batch input:
        R : (Nops, n, n)
        t : (Nops, n)
    """
    H = np.asarray(H)

    # ------------------------
    # Batch case
    # ------------------------
    if H.ndim == 3:
        _, m, n = H.shape

        if m != n:
            raise ValueError("Each matrix must be square.")

        if n < 2:
            raise ValueError("Matrices must be at least 2x2.")

        # Check homogeneous structure for all matrices
        expected_last_row = np.zeros(n)
        expected_last_row[-1] = 1.0

        if not np.allclose(H[:, -1, :], expected_last_row, atol=tol):
            raise ValueError("Invalid homogeneous matrices: last row must be [0, ..., 0, 1].")

        dim = n - 1

        R = H[:, :dim, :dim]
        t = H[:, :dim, dim]

        return R, t

    # ------------------------
    # Single matrix case
    # ------------------------
    if H.ndim == 2:
        m, n = H.shape

        if m != n:
            raise ValueError("Input must be a square matrix.")

        if n < 2:
            raise ValueError("Matrix must be at least 2x2.")

        expected_last_row = np.zeros(n)
        expected_last_row[-1] = 1.0

        if not np.allclose(H[-1, :], expected_last_row, atol=tol):
            raise ValueError("Invalid homogeneous matrix: last row must be [0, ..., 0, 1].")

        dim = n - 1
        R = H[:dim, :dim]
        t = H[:dim, dim]

        return R, t

    raise ValueError("Input must be a 2D or 3D array.")


def block_diag(matrices: List[np.ndarray]):
    total_rows = sum(m.shape[0] for m in matrices)
    total_cols = sum(m.shape[1] for m in matrices)

    out = np.zeros((total_rows, total_cols))

    r = c = 0
    for m in matrices:
        rows, cols = m.shape
        out[r : r + rows, c : c + cols] = m
        r += rows
        c += cols

    return out
