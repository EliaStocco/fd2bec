from typing import Dict, Union
from warnings import warn

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write


def write(*argv, **kwargs):
    return ase_write(*argv, **kwargs)


def read(*argv, rename: bool = False, **kwargs):
    structures = ase_read(*argv, **kwargs)
    if isinstance(structures, Atoms):
        return format_atoms(structures, rename)
    elif isinstance(structures, list):
        return [format_atoms(structure, rename) for structure in structures]
    else:
        raise TypeError(f"type not supported: {type(structures)}.")


def format_atoms(atom: Atoms, rename: bool) -> Atoms:
    if atom.calc is not None:
        results: Dict[str, Union[float, np.ndarray]] = atom.calc.results
        for key, value in results.items():
            if key in ["energy", "free_energy", "dipole", "stress"]:
                if rename:
                    atom.info[f"REF_{key}"] = value
                    warn(f"Renaming '{key}' to 'REF_{key}.'")
                else:
                    atom.info[key] = value
                    # warn(f"Found keyword '{key}': we recommend using 'REF_{key}.'")
            elif key in ["forces"]:
                if rename:
                    atom.arrays[f"REF_{key}"] = value
                    warn(f"Renaming '{key}' to 'REF_{key}.'")
                else:
                    atom.arrays[key] = value
                    # warn(f"Found keyword '{key}': we recommend using 'REF_{key}.'")
            else:
                if rename:
                    atom.info[f"REF_{key}"] = value
                    warn(f"Renaming '{key}' to 'REF_{key}.'")
                else:
                    atom.info[key] = value
                    # warn(f"Found keyword '{key}': we recommend using 'REF_{key}.'")
    atom.calc = None
    return atom
