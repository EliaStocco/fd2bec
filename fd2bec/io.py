from typing import Dict, Union
from warnings import warn

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write


def write(*argv, **kwargs):
    return ase_write(*argv, **kwargs)


def read(*argv, **kwargs):
    structures = ase_read(*argv, **kwargs)
    if isinstance(structures, Atoms):
        return format_atoms(structures)
    elif isinstance(structures, list):
        return [format_atoms(structure) for structure in structures]
    else:
        raise TypeError(f"type not supported: {type(structures)}.")


def format_atoms(atom: Atoms) -> Atoms:
    if atom.calc is not None:
        results: Dict[str, Union[float, np.ndarray]] = atom.calc.results
        for key, value in results.items():
            if key in ["energy", "free_energy", "dipole", "stress"]:
                atom.info[key] = value
                warn(f"Found keyword '{key}': we recommend using 'REF_{key}.'")
            elif key in ["forces"]:
                atom.arrays[key] = value
                warn(f"Found keyword '{key}': we recommend using 'REF_{key}.'")
            else:
                atom.info[key] = value
                warn(f"Found keyword '{key}': we recommend using 'REF_{key}.'")
    atom.calc = None
    return atom
