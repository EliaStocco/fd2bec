import numpy as np
import spglib
from ase import Atoms
from ase.utils import atoms_to_spglib_cell


def ase2spglib_dataset(atoms: Atoms, **kwargs) -> spglib.SpglibDataset:
    cell = atoms_to_spglib_cell(atoms)
    return spglib.get_symmetry_dataset(cell, **kwargs)


def invert_mapping_to_list(mapping: list[int]) -> list[list[int]]:
    """
    Invert a mapping from supercell atoms to primitive atoms
    into a list of lists grouped by primitive atom index.

    Parameters
    ----------
    mapping : array-like of int
        mapping_to_primitive from spglib (length N_super),
        where each entry gives the primitive atom index.

    Returns
    -------
    list[list[int]]
        reverse mapping such that:
        reverse_map[p] = list of supercell indices belonging to primitive atom p
    """
    mapping = np.asarray(mapping)

    n_prim = int(mapping.max()) + 1
    reverse = [[] for _ in range(n_prim)]

    for super_idx, prim_idx in enumerate(mapping):
        reverse[prim_idx].append(super_idx)

    return reverse


def allclose_chunked(a: np.ndarray, b: np.ndarray, atol: float) -> bool:
    for i in range(a.shape[0]):
        if not np.all(np.abs(a[i] - b[i]) <= atol):
            return False
    return True
