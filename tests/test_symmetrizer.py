from pathlib import Path
from typing import Dict, Tuple, Type

import numpy as np
import pytest

from fd2bec.atomic import AtomicStructure
from fd2bec.io import read
from fd2bec.tensor import BornCharges, Dipole, Forces, Position, Stress, Tensor

instructions: Dict[str, Tuple[str, type]] = {
    "positions": ("array", Position),
    "MACE_BEC": ("array", BornCharges),
    "MACE_forces": ("array", Forces),
    "REF_dipole": ("info", Dipole),
    "MACE_stress": ("info", Stress),
}

DATA_DIR = Path(__file__).parent / "molecules/point_group_dataset"
xyz_files = sorted(DATA_DIR.glob("*.xyz"))
@pytest.mark.parametrize("filepath", xyz_files)
def test_symmetry_modes_for_molecules(filepath):

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
        atomic_structure.get_symmetry_modes(tensor=tensor)

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

            _, mode_coefficients, _ = atomic_structure.get_symmetry_modes(tensor=tensor)
            lengths[basis][keyword] = len(mode_coefficients)

    for key in lengths["fractional"]:
        assert lengths["fractional"][key] == lengths["cartesian"][key], \
            f"len(theta) depends on basis for {key}"


def test_nonorthogonal_projection_uses_a_real_mode_basis(monkeypatch):
    """A degenerate real eigenspace may be returned with complex eigenvectors."""
    projection = np.array(
        [
            [1.0, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    tensor = Forces(data=np.zeros((1, 3)), basis="fractional")
    structure = object.__new__(AtomicStructure)
    monkeypatch.setattr(
        AtomicStructure, "get_symmetry_projection", lambda _self, tensor: projection
    )

    original_eig = np.linalg.eig

    def eig_with_complex_invariant_vector(matrix):
        eigenvalues, eigenvectors = original_eig(matrix)
        eigenvectors = eigenvectors.astype(complex)
        eigenvectors[:, eigenvalues > 0.5] *= 1j
        return eigenvalues, eigenvectors

    monkeypatch.setattr(np.linalg, "eig", eig_with_complex_invariant_vector)

    _, _, modes = structure.get_symmetry_modes(tensor)

    assert np.isrealobj(modes)
    np.testing.assert_allclose(projection @ modes.T, modes.T)

@pytest.mark.parametrize("basis",["fractional","cartesian"])
def test_symmetry_modes_for_periodic_structure(sg_case,basis):

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
        atomic_structure.get_symmetry_modes(tensor=tensor)






if __name__ == "__main__":
    pytest.main([__file__])
