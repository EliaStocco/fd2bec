import argparse
from typing import Tuple

import numpy as np
from ase.io import read

from fd2bec import float_format
from fd2bec.atomic import AtomicStructure
from fd2bec.cli import cli

description = "Generate all cartesian displacements."


def prepare_args(descr):

    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="path to input structure (e.g. supercell.extxyz)",
    )
    parser.add_argument(
        "-a",
        "--amplitude",
        **argv,
        type=float,
        required=True,
        help="amplitude of the displacement",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        type=str,
        required=True,
        help="path to txt output file with cartesian displacements (e.g. displacement.txt)",
    )
    return parser


def atomic_structure2all_displacements(
    unit_cell: AtomicStructure, amplitude: float, use_delta_dipole: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate all symmetry inequivalent cartesian displacements
    for the given unit cell and amplitude."""

    N = 3 * len(unit_cell)
    displacements = np.eye(N) * amplitude

    if not use_delta_dipole:
        displacements = np.concatenate([np.zeros((1, 3 * len(unit_cell))), displacements], axis=0)

    return displacements, displacements


@cli(prepare_args, description)
def main(args):

    print(f"Reading input structure from {args.input} ... ", end="")
    atoms = read(args.input, index=0)
    unit_cell = AtomicStructure.from_ase(atoms)
    print("done")

    N = 3 * atoms.get_global_number_of_atoms()
    print(f"Generating all {N} displacements ... ", end="")
    displacements = atomic_structure2all_displacements(
        unit_cell, args.amplitude, use_delta_dipole=True
    )[0]
    print("done")

    print(f"Writing cartesian displacements to {args.output} ... ", end="")
    np.savetxt(args.output, displacements, fmt=float_format)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
