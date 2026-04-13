import pytest
import numpy as np
from ase.io import read
from fd2bec import SYMPREC
from fd2bec.conftest import structure, structures_dir
from fd2bec.tools import ase2spglib_dataset
from fd2bec.mathematics import wrap
from fd2bec.atomic import AtomicStructure

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

    assert actual_atoms == expected_atoms, (
        f"File: {file_path}\n"
        f"Expected atoms: {expected_atoms}\n"
        f"Actual atoms: {actual_atoms}"
    )

    dataset = ase2spglib_dataset(atoms, symprec=SYMPREC)

    assert dataset.number == 99, (
        f"File: {file_path}\n"
        f"Expected: 99\n"
        f"Got: {dataset.number}\n"
        f"International symbol: {dataset.international}\n"
        f"Symprec: {SYMPREC}"
    )
            
def test_number_operations():
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
    N = atoms.get_global_number_of_atoms()

    dataset = ase2spglib_dataset(atoms, symprec=SYMPREC)
    new_atoms = atoms.copy()

    frac_pos = atoms.get_scaled_positions()
    new_frac = frac_pos.copy()
    
    tmp = AtomicStructure.from_ase(atoms)
    assert tmp._test_symmetry(), "Error in AtomicStructure._test_symmetry() method"
    
    tmp.get_affine_symmetry_operations()

    for op_idx, (R, t) in enumerate(zip(dataset.rotations, dataset.translations)):
        new_frac = (new_frac @ R + t[None, :])
        new_atoms.set_scaled_positions(new_frac)

        if AtomicStructure.from_ase(atoms) != AtomicStructure.from_ase(new_atoms):
            # compute distances for debugging
            diff = new_frac - frac_pos
            diff = wrap(diff)
            max_dev = np.max(np.abs(diff))

            raise AssertionError(
                f"{file_path}: Symmetry operation #{op_idx} failed\n"
                f"Rotation:\n{R}\n"
                f"Translation: {t}\n"
                f"Max deviation after wrapping: {max_dev:.3e}\n"
                f"Tolerance: {SYMPREC}\n"
                f"Number of atoms: {N}"
            )

    diff = new_frac - frac_pos
    diff = wrap(diff)
    max_dev = np.max(np.abs(diff))

    assert np.allclose(diff, 0, atol=1e-5 * N), (
        f"{file_path}: Final symmetry mismatch\n"
        f"Max deviation: {max_dev:.3e}\n"
        f"Tolerance: {1e-5 * N:.3e}\n"
        f"Mean deviation: {np.mean(np.abs(diff)):.3e}"
    )
        

if __name__ == "__main__":
    pytest.main([__file__])