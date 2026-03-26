import numpy as np
import spglib
from ase import Atoms

def ase2spglib(atoms:Atoms,**kwargs) -> spglib.SpglibDataset:
    cell = (atoms.get_cell()[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    return spglib.get_symmetry_dataset(cell, **kwargs)

def wrap(x:np.ndarray):
    return (x + 0.5) % 1.0 - 0.5