#!/usr/bin/env python3
"""
Generate N random Euler-angle rotations of an atomic structure (including periodic cells),
store the Euler angles in atoms.info, and write all rotated structures to an output file.

Supports any ASE-readable format (e.g. xyz, extxyz, cif, traj, POSCAR, etc.).

Usage:
    python random_rotate_structures.py input_file output_file N [--degrees]

Examples:
    python random_rotate_structures.py input.cif rotated.extxyz 100
    python random_rotate_structures.py POSCAR rotated.traj 50 --degrees
"""

import argparse
import numpy as np
from ase.io import read, write


def rotation_matrix_from_euler(alpha, beta, gamma, degrees=False):
    """
    Create rotation matrix from ZYX Euler angles:
        R = Rz(alpha) @ Ry(beta) @ Rx(gamma)

    Parameters
    ----------
    alpha, beta, gamma : float
        Euler angles
    degrees : bool
        If True, input angles are in degrees

    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    if degrees:
        alpha, beta, gamma = np.radians([alpha, beta, gamma])

    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    rz = np.array([
        [ca, -sa, 0],
        [sa,  ca, 0],
        [0,    0, 1]
    ])

    ry = np.array([
        [cb, 0, sb],
        [0,  1, 0],
        [-sb, 0, cb]
    ])

    rx = np.array([
        [1,  0,   0],
        [0, cg, -sg],
        [0, sg,  cg]
    ])

    return rz @ ry @ rx


def random_euler_angles(n, degrees=False):
    """
    Generate N random Euler angles.
    Uniform sampling over angle ranges:
      alpha ∈ [0, 2π)
      beta  ∈ [0, π)
      gamma ∈ [0, 2π)
    """
    alpha = np.random.uniform(0, 2*np.pi, n)
    beta = np.random.uniform(0, np.pi, n)
    gamma = np.random.uniform(0, 2*np.pi, n)

    if degrees:
        return np.degrees(alpha), np.degrees(beta), np.degrees(gamma)

    return alpha, beta, gamma

from ase import Atoms
def rotate_atoms(atoms:Atoms, rotation_matrix):
    """
    Rotate both atomic positions and periodic cell around the center of mass.
    """
    rotated = atoms.copy()

    # Rotate positions about center of mass
    center = rotated.get_center_of_mass()
    positions = rotated.get_positions()
    rotated_positions = (positions - center) @ rotation_matrix.T + center
    rotated.set_positions(rotated_positions)

    # Rotate cell if periodic
    if rotated.cell is not None:
        rotated_cell = rotated.cell.array @ rotation_matrix.T
        rotated.set_cell(rotated_cell, scale_atoms=False)

    return rotated


def main():
    parser = argparse.ArgumentParser(
        description="Generate N random Euler-rotated structures from an input atomic structure."
    )
    parser.add_argument("input_file", help="Input atomic structure file")
    parser.add_argument("output_file", help="Output file for rotated structures")
    parser.add_argument("N", type=int, help="Number of random rotated structures")
    parser.add_argument(
        "--degrees",
        action="store_true",
        help="Store Euler angles in degrees instead of radians"
    )

    args = parser.parse_args()

    # Read first structure from file
    atoms = read(args.input_file)

    # Generate random angles
    alphas, betas, gammas = random_euler_angles(args.N, degrees=args.degrees)

    rotated_structures = []

    for i in range(args.N):
        alpha, beta, gamma = alphas[i], betas[i], gammas[i]

        R = rotation_matrix_from_euler(alpha, beta, gamma, degrees=args.degrees)
        rotated = rotate_atoms(atoms, R)

        # Store Euler angles in atoms.info
        rotated.info["euler_alpha"] = float(alpha)
        rotated.info["euler_beta"] = float(beta)
        rotated.info["euler_gamma"] = float(gamma)

        rotated_structures.append(rotated)

    # Write all structures
    write(args.output_file, rotated_structures)

    print(f"Saved {args.N} rotated structures to {args.output_file}")


if __name__ == "__main__":
    main()
