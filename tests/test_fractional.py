import pytest
import numpy as np
from ase.io import read
from fd2bec import ATOL
# # from fd2bec.conftest import structure # noqa: F401
from fd2bec.atomic import AtomicStructure


def test_fractional_rank_1(structure):
    """
    Test that structures have correct consistency between Cartesian and fractional transforms.
    """
    n, file_path = structure

    atoms = read(file_path, index=0)

    atomic_structure = AtomicStructure.from_ase(atoms)

    arrays_to_test = ['positions', 'REF_forces', 'REF_atomic_dipoles']

    for name in arrays_to_test:
        if name not in atoms.arrays:
            pytest.skip(f"{file_path}: missing array '{name}'")

        pos = atoms.arrays[name]

        frac_pos_ref = atoms.cell.scaled_positions(pos)
        frac_pos_test = atomic_structure.to_fractional(pos)

        diff_frac = np.abs(frac_pos_ref - frac_pos_test)
        max_frac = np.max(diff_frac)

        assert np.allclose(frac_pos_ref, frac_pos_test, atol=ATOL), (
            f"\n[FRACTIONAL TRANSFORM MISMATCH]"
            f"\nFile: {file_path}"
            f"\nStructure id: {n}"
            f"\nArray: {name}"
            f"\nShape: {pos.shape}"
            f"\nMax |Δ|: {max_frac:.3e}"
        )

        cart_test = atomic_structure.to_cartesian(frac_pos_ref)
        diff_cart = np.abs(pos - cart_test)
        max_cart = np.max(diff_cart)

        assert np.allclose(pos, cart_test, atol=ATOL), (
            f"\n[CARTESIAN ROUND-TRIP MISMATCH]"
            f"\nFile: {file_path}"
            f"\nStructure id: {n}"
            f"\nArray: {name}"
            f"\nShape: {pos.shape}"
            f"\nMax |Δ|: {max_cart:.3e}"
        )
        
def test_fractional_rank_2(structure):
    """
    Test that structures have correct consistency between Cartesian and fractional transforms.
    """
    n, file_path = structure

    atoms = read(file_path, index=0)
    Natoms = atoms.get_global_number_of_atoms()

    atomic_structure = AtomicStructure.from_ase(atoms)

    to_test = ['REF_BEC', 'REF_stress']

    for name in to_test:
        if name in atoms.arrays:
            pos = atoms.arrays[name].reshape((Natoms,3,3))
        elif name in atoms.info:
            pos = atoms.info[name].reshape((3,3))
        else:
            pytest.skip(f"{file_path}: missing '{name}'")

        frac_pos_test = atomic_structure.to_fractional(pos,rank=2)
        cart_test = atomic_structure.to_cartesian(frac_pos_test,rank=2)
        diff_cart = np.abs(pos - cart_test)
        max_cart = np.max(diff_cart)

        assert np.allclose(pos, cart_test, atol=ATOL), (
            f"\n[CARTESIAN ROUND-TRIP MISMATCH]"
            f"\nFile: {file_path}"
            f"\nStructure id: {n}"
            f"\nArray: {name}"
            f"\nShape: {pos.shape}"
            f"\nMax |Δ|: {max_cart:.3e}"
        )
        
if __name__ == "__main__":
    pytest.main([__file__])