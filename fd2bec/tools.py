import numpy as np
import spglib
from ase import Atoms

def ase2spglib_cell(atoms:Atoms):
    return atoms.get_cell()[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers()

def ase2spglib_dataset(atoms:Atoms,**kwargs) -> spglib.SpglibDataset:
    cell = ase2spglib_cell(atoms)
    return spglib.get_symmetry_dataset(cell, **kwargs)

def wrap(x:np.ndarray):
    return (x + 0.5) % 1.0 - 0.5

def invert_mapping_to_list(mapping:list[int]) -> list[list[int]]:
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