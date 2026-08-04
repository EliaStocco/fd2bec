import numpy as np
import pytest
from fd2bec.io import read

from fd2bec import ATOL, SYMPREC
from fd2bec.atomic import AtomicStructure
from fd2bec.mathematics import wrap
from fd2bec.tools import ase2spglib_dataset

# structures_dir = Path(__file__).resolve().parents[1] / "fd2bec" / "structures"


def test_spacegroup(structure):
    """
    Test that the structures have the correct number of atoms
    and all have space group 99.
    """
    n, file_path = structure

    atoms = read(file_path, index=0)

    factor = n**3
    expected_atoms = factor * 5
    actual_atoms = atoms.get_global_number_of_atoms()

    assert (
        actual_atoms == expected_atoms
    ), f"File: {file_path}\nExpected atoms: {expected_atoms}\nActual atoms: {actual_atoms}"

    dataset = ase2spglib_dataset(atoms, symprec=SYMPREC)

    assert dataset.number == 99, (
        f"File: {file_path}\n"
        f"Expected: 99\n"
        f"Got: {dataset.number}\n"
        f"International symbol: {dataset.international}\n"
        f"Symprec: {SYMPREC}"
    )


def test_number_operations(structures_dir):
    """
    Test that the structures have the correct number of symmetry operations
    across supercells.
    """
    first = True
    first_No = 0

    for n in [1, 2, 3]:
        file_path = structures_dir / f"BaTiO3.{n}x{n}x{n}.extxyz"
        atoms = read(file_path, index=0)

        factor = n**3
        dataset = ase2spglib_dataset(atoms, symprec=SYMPREC)

        n_ops = dataset.rotations.shape[0]

        if first:
            first_No = n_ops
            first = False
        else:
            expected = factor * first_No

            assert n_ops == expected, (
                f"{file_path}: Unexpected number of symmetry operations\n"
                f"Expected: {expected}\n"
                f"Got: {n_ops}\n"
                f"Unit cell operations: {first_No}\n"
                f"Supercell factor (n^3): {factor}\n"
                f"Symprec: {SYMPREC}"
            )


def test_structures_spacegroup_positions(structure):
    """
    Test that the structures have the correct number of atoms,
    correct space group, and that atomic positions are symmetric.
    """
    n, file_path = structure
    atoms = read(file_path, index=0)
    atomic_structure = AtomicStructure.from_ase(atoms)
    atomic_structure._test_symmetry(
        atol=ATOL
    )




if __name__ == "__main__":
    pytest.main([__file__])
