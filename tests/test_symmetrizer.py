
import pytest
from fd2bec.io import read
from pathlib import Path

from fd2bec.atomic import AtomicStructure
from typing import Dict, Tuple, Type

# # from fd2bec.conftest import structure # noqa: F401
from fd2bec.tensor import BornCharges, Forces, Position, Dipole, Stress, Tensor


instructions: Dict[str, Tuple[str, type]] = {
    "positions": ("array", Position),
    "MACE_BEC": ("array", BornCharges),
    "MACE_forces": ("array", Forces),
    "MACE_dipole": ("info", Dipole),
    "MACE_stress": ("info", Stress),
}

DATA_DIR = Path(__file__).parent / "molecules/point_group_dataset"
xyz_files = sorted(DATA_DIR.glob("*.xyz"))
@pytest.mark.parametrize("filepath", xyz_files)
def test_symmetrizer_molecules(filepath):

    atoms = read(filepath)
    Natoms = atoms.get_global_number_of_atoms()
    atomic_structure = AtomicStructure.from_ase(atoms)
    atomic_structure._test_symmetry(basis="cartesian")

    for keyword, (_, classname) in instructions.items():
        classname:Type[Tensor]
        if keyword == "positions":
            tensor = classname(data=atoms.get_positions()) # I need the positions in this case
        else:
            tensor = classname.template(Natoms)
        S, theta, theta_real = atomic_structure.get_symmetrizer(tensor=tensor)

def test_theta_length_periodic(sg_case):
    dataset, filepath, n = sg_case
    atoms = read(filepath)
    Natoms = atoms.get_global_number_of_atoms()

    atomic_structure = AtomicStructure.from_ase(atoms)

    lengths = {}

    for basis in ["fractional", "cartesian"]:
        atomic_structure._test_symmetry(basis=basis)

        lengths[basis] = {}

        for keyword, (_, classname) in instructions.items():
            classname:Type[Tensor]
            if keyword == "positions":
                tensor = classname(data=atoms.get_positions()) # I need the positions in this case
            else:
                tensor = classname.template(Natoms)

            _, theta, _ = atomic_structure.get_symmetrizer(tensor=tensor)
            lengths[basis][keyword] = len(theta)

    for key in lengths["fractional"]:
        assert lengths["fractional"][key] == lengths["cartesian"][key], \
            f"len(theta) depends on basis for {key}"

@pytest.mark.parametrize("basis",["fractional","cartesian"])
def test_symmetrizer_periodic(sg_case,basis):

    dataset, filepath, n = sg_case
    atoms = read(filepath)
    Natoms = atoms.get_global_number_of_atoms()
    atomic_structure = AtomicStructure.from_ase(atoms)
    assert atomic_structure.space_group == n, \
        f"Wrong space group: {n} != {atomic_structure.space_group} on file {filepath}"
    atomic_structure._test_symmetry(basis=basis)

    for keyword, (_, classname) in instructions.items():
        classname:Type[Tensor]
        if keyword == "positions":
            tensor = classname(data=atoms.get_positions()) # I need the positions in this case
        else:
            tensor = classname.template(Natoms)
        S, theta, theta_real = atomic_structure.get_symmetrizer(tensor=tensor)



          


if __name__ == "__main__":
    pytest.main([__file__])
