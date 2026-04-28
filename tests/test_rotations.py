from ase.io import read
import pytest
import numpy as np
from ase import Atoms
from fd2bec import ATOL
from pathlib import Path
from fd2bec.tools import atoms2bec
from fd2bec.tensor import Dipole, BornCharge, Force, Stress, LatticeVectors

REF = Path(__file__).parent / "rotations/start.extxyz"
FILE = Path(__file__).parent / "rotations/rotated.extxyz"


def rotation_matrix_from_euler(alpha, beta, gamma):
    """
    Create rotation matrix from ZYX Euler angles:
        R = Rz(alpha) @ Ry(beta) @ Rx(gamma)

    Parameters
    ----------
    alpha, beta, gamma : float
        Euler angles (radians)
    degrees : bool
        If True, input angles are in degrees

    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """

    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])

    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])

    rx = np.array([[1, 0, 0], [0, cg, -sg], [0, sg, cg]])

    return rz @ ry @ rx


def reconstruct_bec(atoms: Atoms):
    return atoms2bec(atoms, "MACE_BEC")


@pytest.mark.parametrize("n", range(10))
def test_rotations_cellpar(n):

    reference = read(REF, index=0)
    # ref_as = AtomicStructure.from_ase(reference)
    ref_BEC = reconstruct_bec(reference)

    atoms = read(FILE, index=n)
    # atoms_as = AtomicStructure.from_ase(atoms)

    assert np.allclose(atoms.cell.cellpar(), reference.cell.cellpar(), atol=ATOL), (
        f"\n[ROTATION MISMATCH]"
        f"\nStructure id: {n}"
        f"\nCell parameters:\n{atoms.cell.cellpar()}\nReference Cell parameters:\n{reference.cell.cellpar()}"
        f"\nMax |Δ|: {np.max(np.abs(atoms.cell.cellpar() - reference.cell.cellpar())):.3e}"
    )


@pytest.mark.parametrize("n", range(10))
def test_rotations_manual(n):

    reference = read(REF, index=0)
    # ref_as = AtomicStructure.from_ase(reference)
    ref_BEC = reconstruct_bec(reference)

    atoms = read(FILE, index=n)
    # atoms_as = AtomicStructure.from_ase(atoms)

    alpha = atoms.info["euler_alpha"]
    beta = atoms.info["euler_beta"]
    gamma = atoms.info["euler_gamma"]

    R = rotation_matrix_from_euler(alpha, beta, gamma)
    cell = reference.cell.array @ R.T

    assert np.allclose(cell, atoms.cell.array, atol=ATOL), (
        f"\n[ROTATION MISMATCH]"
        f"\nStructure id: {n}"
        f"\nCell:\n{cell}\nReference Cell:\n{reference.cell.array}"
        f"\nMax |Δ|: {np.max(np.abs(cell - reference.cell.array)):.3e}"
    )

    for vector_arrays in ["MACE_forces"]:

        original = reference.arrays[vector_arrays]
        rotated = original @ R.T

        this = atoms.arrays[vector_arrays]

        assert np.allclose(rotated, this, atol=ATOL), (
            f"\n[ROTATION MISMATCH]"
            f"\nStructure id: {n}"
            f"\nVector: {vector_arrays}"
            f"\nMax |Δ|: {np.max(np.abs(rotated - reference.arrays[vector_arrays])):.3e}"
        )

    bec = reconstruct_bec(atoms)

    test_bec = np.einsum("ij,mjk,lk->mil", R, ref_BEC, R)

    assert np.allclose(test_bec, bec, atol=ATOL), (
        f"\n[ROTATION MISMATCH]"
        f"\nStructure id: {n}"
        f"\nBEC:\n{test_bec}\nReference BEC:\n{bec}"
        f"\nMax |Δ|: {np.max(np.abs(test_bec - bec)):.3e}"
    )

    Z = (np.kron(R, R) @ ref_BEC.reshape((len(atoms), 9)).T).T
    Z = Z.reshape((-1, 3, 3))
    assert np.allclose(test_bec, Z, atol=ATOL), (
        f"\n[ROTATION MISMATCH]"
        f"\nStructure id: {n}"
        f"\nBEC:\n{test_bec}\nReference BEC:\n{Z}"
        f"\nMax |Δ|: {np.max(np.abs(test_bec - bec)):.3e}"
    )


@pytest.mark.parametrize("n", range(10))
def test_rotations_tensors(n):

    reference = read(REF, index=0)
    ref_dipole = Dipole(data=reference.info["MACE_dipole"])
    ref_forces = Force(data=reference.arrays["MACE_forces"])
    ref_stress = Stress(data=reference.info["MACE_stress"])
    ref_bec = BornCharge(data=reconstruct_bec(reference))
    ref_cell = LatticeVectors(data=reference.cell)

    atoms = read(FILE, index=n)

    alpha = atoms.info["euler_alpha"]
    beta = atoms.info["euler_beta"]
    gamma = atoms.info["euler_gamma"]

    R = rotation_matrix_from_euler(alpha, beta, gamma)

    ref_dipole_rotated = ref_dipole.rotate(R)
    ref_forces_rotated = ref_forces.rotate(R)
    ref_stress_rotated = ref_stress.rotate(R)
    ref_bec_rotated = ref_bec.rotate(R)
    ref_cell_rotated = ref_cell.rotate(R)

    assert np.allclose(ref_dipole_rotated.data, atoms.info["MACE_dipole"], atol=ATOL)
    assert np.allclose(ref_forces_rotated.data, atoms.arrays["MACE_forces"], atol=ATOL)
    assert np.allclose(ref_stress_rotated.data, atoms.info["MACE_stress"], atol=ATOL)
    assert np.allclose(ref_bec_rotated.data, reconstruct_bec(atoms), atol=ATOL)
    assert np.allclose(ref_cell_rotated.data, atoms.cell.array, atol=ATOL)


if __name__ == "__main__":
    pytest.main([__file__])
