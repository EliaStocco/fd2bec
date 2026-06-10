import numpy as np
import pytest
from pathlib import Path
from fd2bec.io import read

from fd2bec import ATOL
from fd2bec.atomic import AtomicStructure
from fd2bec.mathematics import append_one, remove_one
from typing import Dict, Tuple, Type
from fd2bec.tools import atoms2bec

# # from fd2bec.conftest import structure # noqa: F401
from fd2bec.tensor import BornCharges, Forces, Position, Dipole, Stress, Tensor

FILE = Path(__file__).parent / "rotations/rotated.extxyz"

instructions: Dict[str, Tuple[str, type]] = {
    "positions": ("array", Position),
    "MACE_BEC": ("array", BornCharges),
    "MACE_forces": ("array", Forces),
    "MACE_dipole": ("info", Dipole),
    "MACE_stress": ("info", Stress),
}

@pytest.mark.parametrize("basis",["fractional","cartesian"])
@pytest.mark.parametrize("n", range(10))
def test_symmetrizer(n,basis):

    atoms = read(FILE, index=n)
    Natoms = atoms.get_global_number_of_atoms()
    atomic_structure = AtomicStructure.from_ase(atoms)
    atomic_structure._test_symmetry(basis=basis)

    for keyword, (where, classname) in instructions.items():
        if keyword == "MACE_BEC":
            array = atoms2bec(atoms, keyword)
        elif where == "array":
            array = atoms.arrays[keyword]
        else:
            array = atoms.info[keyword]

        classname:Type[Tensor]
        template = classname.template(Natoms)
        tensor: Tensor = classname(data=array, cell=atoms.cell, basis="cartesian").to(basis=basis)
        atomic_structure.get_symmetrizer(tensor=tensor)
        pass


if __name__ == "__main__":
    pytest.main([__file__])
