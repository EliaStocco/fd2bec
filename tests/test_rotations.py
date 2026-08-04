from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from fd2bec.io import read

from fd2bec import ATOL
from fd2bec.tensor import BornCharges, Dipole, Forces, Stress
from fd2bec.tools import atoms2bec

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


import pytest


@pytest.mark.parametrize("method", ["recursive", "flat"])
@pytest.mark.parametrize("n", range(10))
def test_rotations_tensors(n, method):

    atoms = read(FILE, index=n)

    R = rotation_matrix_from_euler(
        atoms.info["euler_alpha"],
        atoms.info["euler_beta"],
        atoms.info["euler_gamma"],
    )

    # ------------------------------------------------------------
    # Build tensors
    # ------------------------------------------------------------
    dipole = Dipole(data=atoms.info["REF_dipole"], cell=atoms.cell)
    forces = Forces(data=atoms.arrays["MACE_forces"], cell=atoms.cell)
    stress = Stress(data=atoms.info["MACE_stress"], cell=atoms.cell)
    bec = BornCharges(data=reconstruct_bec(atoms), cell=atoms.cell)
    # cell = LatticeVectors(data=atoms.cell, cell=atoms.cell)

    # ------------------------------------------------------------
    # Rotate using selected method
    # ------------------------------------------------------------
    dip_rot = dipole.rotate(R, method=method)
    f_rot = forces.rotate(R, method=method)
    s_rot = stress.rotate(R, method=method)
    b_rot = bec.rotate(R, method=method)
    # c_rot = cell.rotate(R, method=method)

    # ------------------------------------------------------------
    # Consistency check vs the OTHER method
    # ------------------------------------------------------------
    other_method = "flat" if method == "recursive" else "recursive"

    dip_rot_2 = dipole.rotate(R, method=other_method)
    f_rot_2 = forces.rotate(R, method=other_method)
    s_rot_2 = stress.rotate(R, method=other_method)
    b_rot_2 = bec.rotate(R, method=other_method)
    # c_rot_2 = cell.rotate(R, method=other_method)

    # ------------------------------------------------------------
    # Compare implementations
    # ------------------------------------------------------------
    assert np.allclose(dip_rot.data, dip_rot_2.data, atol=ATOL), "Dipole mismatch"
    assert np.allclose(f_rot.data, f_rot_2.data, atol=ATOL), "Forces mismatch"
    assert np.allclose(s_rot.data, s_rot_2.data, atol=ATOL), "Stress mismatch"
    assert np.allclose(b_rot.data, b_rot_2.data, atol=ATOL), "Born charge mismatch"
    # assert np.allclose(c_rot.data, c_rot_2.data, atol=ATOL), "Cell mismatch"


@pytest.mark.parametrize("method", ["recursive", "flat"])
@pytest.mark.parametrize("n", range(10))
def test_rotation_operator(n, method):

    atoms = read(FILE, index=n)

    R = rotation_matrix_from_euler(
        atoms.info["euler_alpha"],
        atoms.info["euler_beta"],
        atoms.info["euler_gamma"],
    )

    # ------------------------------------------------------------
    # Build tensors
    # ------------------------------------------------------------
    dipole = Dipole(data=atoms.info["REF_dipole"], cell=atoms.cell)
    forces = Forces(data=atoms.arrays["MACE_forces"], cell=atoms.cell)
    stress = Stress(data=atoms.info["MACE_stress"], cell=atoms.cell)
    bec = BornCharges(data=reconstruct_bec(atoms), cell=atoms.cell)
    # cell = LatticeVectors(data=atoms.cell, cell=atoms.cell)

    # ------------------------------------------------------------
    # Rotate using selected method
    # ------------------------------------------------------------
    dip_R = dipole.rotation_operator(R)
    f_R = forces.rotation_operator(R)
    s_R = stress.rotation_operator(R)
    b_R = bec.rotation_operator(R)
    # c_rot = cell.rotate(R, method=method)

    dip_rot = np.einsum("ij,...j->...i", dip_R, dipole.flatten())
    f_rot = np.einsum("ij,...j->...i", f_R, forces.flatten())
    s_rot = np.einsum("ij,...j->...i", s_R, stress.flatten())
    b_rot = np.einsum("ij,...j->...i", b_R, bec.flatten())

    assert np.allclose(dip_rot, dipole.contract(dip_R).flatten(), atol=ATOL), "Dipole mismatch"
    assert np.allclose(f_rot, forces.contract(f_R).flatten(), atol=ATOL), "Forces mismatch"
    assert np.allclose(s_rot, stress.contract(s_R).flatten(), atol=ATOL), "Stress mismatch"
    assert np.allclose(b_rot, bec.contract(b_R).flatten(), atol=ATOL), "Born charge mismatch"

    # ------------------------------------------------------------
    # Consistency check vs the OTHER method
    # ------------------------------------------------------------
    dip_rot_2 = dipole.rotate(R=R, method=method).flatten()
    f_rot_2 = forces.rotate(R=R, method=method).flatten()
    s_rot_2 = stress.rotate(R=R, method=method).flatten()
    b_rot_2 = bec.rotate(R=R, method=method).flatten()
    # c_rot_2 = cell.rotate(R, method=other_method)

    # ------------------------------------------------------------
    # Compare implementations
    # ------------------------------------------------------------
    assert np.allclose(dip_rot, dip_rot_2, atol=ATOL), "Dipole mismatch"
    assert np.allclose(f_rot, f_rot_2, atol=ATOL), "Forces mismatch"
    assert np.allclose(s_rot, s_rot_2, atol=ATOL), "Stress mismatch"
    assert np.allclose(b_rot, b_rot_2, atol=ATOL), "Born charge mismatch"
    # assert np.allclose(c_rot.data, c_rot_2.data, atol=ATOL), "Cell mismatch"


if __name__ == "__main__":
    pytest.main([__file__])
