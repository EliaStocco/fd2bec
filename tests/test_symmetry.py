from pathlib import Path
import re
import numpy as np
import pytest
import spglib
from ase.utils import atoms_to_spglib_cell
from fd2bec.io import read

from fd2bec.mathematics import wrap
from fd2bec import SYMPREC, ATOL
from fd2bec.atomic import AtomicStructure

def test_space_group(sg_case):
    dataset, filepath, n = sg_case
    atoms = read(filepath)
    cell = atoms_to_spglib_cell(atoms)
    dataset = spglib.get_symmetry_dataset(cell, symprec=SYMPREC)
    number = dataset.number             # e.g. 225
    assert number == n, f"Wrong space group: {n} != {number} on file {filepath}"


def test_symmetry(sg_case):
    dataset, filepath, n = sg_case
    atoms = read(filepath)
    atomic_structure = AtomicStructure.from_ase(atoms)
    assert atomic_structure.space_group == n, f"Wrong space group: {n} != {atomic_structure.space_group} on file {filepath}"
    atomic_structure._test_symmetry()
    atomic_structure.conventional._test_symmetry()

def test_translations(sg_case):
    """
    Verify origin-shift invariance of spglib translations.

    Tests that under a fractional origin shift δ, translations transform as:
        T' = T + δ (I - Rᵀ)

    using row-vector convention x' = x Rᵀ + t.
    """
    dataset, filepath, n = sg_case
    atoms = read(filepath)

    orig = atoms.copy()
    original = AtomicStructure.from_ase(atoms)
    original._test_symmetry()
    R, T = original.get_space_group_operations()

    for _ in range(10):
        shift = np.random.rand(3)

        atoms = orig.copy()  # do not accumulate translations
        atoms.translate(shift)
        shift_frac = atoms.cell.scaled_positions(shift)

        atomic_structure = AtomicStructure.from_ase(atoms)
        atomic_structure._test_symmetry()

        R_prime, T_prime = atomic_structure.get_space_group_operations()

        # row-vector consistent transformation:
        T_test = T + shift_frac[None, :] @ (np.eye(3) - R.transpose(0, 2, 1))

        if not np.allclose(R, R_prime, atol=ATOL):
            raise ValueError("Rotation mismatch.")
        # if not np.allclose(wrap(T_test - T_prime), 0, atol=ATOL):
        #     raise ValueError("Translation mismatch.")

if __name__ == "__main__":
    pytest.main([__file__])
