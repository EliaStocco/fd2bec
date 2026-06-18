from typing import Union

import numpy as np
from ase.cell import Cell


# ---------------------------------------#
# Decorator to convert ase.Cell to np.array and transpose
def ase_cell_to_np_transpose(func):
    """Decorator to convert an ASE cell to NumPy array and transpose for use in the decorated function."""

    def wrapper(cell: Union[np.ndarray, Cell], *args, **kwargs):
        if isinstance(cell, Cell):
            cell = np.asarray(cell).T
        if cell.shape != (3, 3):
            raise ValueError("cell with wrong shape:", cell.shape)
        return func(cell, *args, **kwargs)

    return wrapper


# ---------------------------------------#
def return_transformed_components(func):
    """Decorator to automatically compute the matrix multiplication if a vector is provided."""

    def wrapper(cell: Union[np.ndarray, Cell], v: np.ndarray = None, *args, **kwargs) -> np.ndarray:
        matrix = func(cell=cell, *args, **kwargs)
        if not isinstance(matrix, np.ndarray):
            raise TypeError("'matrix' should be 'np.ndarray'")
        if v is None:
            return matrix
        else:
            shape = v.shape
            v = np.asarray(v).reshape((-1, 3))
            out = (matrix @ v.T).T
            return out.reshape(shape)

    return wrapper


# ---------------------------------------#
@return_transformed_components
@ase_cell_to_np_transpose
def lattice2cart(cell: Union[np.ndarray, Cell], v: np.ndarray = None) -> np.ndarray:
    """Lattice to Cartesian coordinates rotation matrix."""
    from copy import copy

    # normalize the lattice parameters
    length = np.linalg.norm(cell, axis=0)
    matrix = copy(cell)
    # normalize the columns
    for i in range(3):
        matrix[:, i] /= length[i]
    return matrix


# ---------------------------------------#
@return_transformed_components
@ase_cell_to_np_transpose
def cart2lattice(cell: Union[np.ndarray, Cell], v: np.ndarray = None) -> np.ndarray:
    """Cartesian to lattice coordinates rotation matrix."""
    matrix = lattice2cart(cell)
    matrix = np.linalg.inv(matrix)
    return matrix


# ---------------------------------------#
@return_transformed_components
@ase_cell_to_np_transpose
def frac2cart(cell: Union[np.ndarray, Cell], v: np.ndarray = None) -> np.ndarray:
    """Return a 3x3 rotation matrix from fractional to cartesian coordinates.\n
    The lattice vectors are not normalized, differently w.r.t. `lattice2cart`.\n
    The function is decorated such that you can provide vector `v` as `numpy.ndarray` to automatically rotate them.

    Input:
        `cell`: lattice parameters
            a `numpy.ndarray` with the i^th lattice vector stored in the i^th columns (it's the opposite of ASE, QE, FHI-aims)\n
            or an `ase.cell.Cell` object (the code will automatically convert it to `numpy.ndarray` and take the transpose).\n
            The format of `cell` should be the following:
                | a_1x a_2x a_3x |\n
                | a_1y a_2y a_3y |\n
                | a_1z a_2z a_3z |\n
        `v`: vectors (optional)
            `numpy.ndarray` with the vectors to rotate.

    Output:
        rotation matrix: `numpy.ndarray` of shape 3x3,\n
        or rotated vectors if `v` is provided: `numpy.ndarray` with the same shape of `v`
    """
    return cell


# ---------------------------------------#
@return_transformed_components
@ase_cell_to_np_transpose
def cart2frac(cell: Union[np.ndarray, Cell], v: np.ndarray = None) -> np.ndarray:
    """Cartesian to lattice coordinates rotation matrix."""
    matrix = frac2cart(cell)
    matrix = np.linalg.inv(matrix)
    return matrix
