import argparse

import numpy as np

from fd2bec import float_format
from fd2bec.atomic import AtomicStructure
from fd2bec.cli import cli
from fd2bec.io import read

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
        required=False,
        help="amplitude of the displacement (default: %(default)s)",
        default=1e-3,
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


def atomic_structure2all_displacements(unit_cell: AtomicStructure, amplitude: float) -> np.ndarray:
    """Generate all symmetry inequivalent cartesian displacements
    for the given unit cell and amplitude."""

    N = 3 * len(unit_cell)
    plus = np.eye(N) * amplitude
    minus = -plus.copy()
    null = np.zeros((1, 3 * len(unit_cell)))

    return np.concatenate([plus, minus, null])


@cli(prepare_args, description)
def main(args):

    print(f"Reading input structure from {args.input} ... ", end="")
    atoms = read(args.input, index=0)
    unit_cell = AtomicStructure.from_ase(atoms)
    print("done")

    N = 3 * atoms.get_global_number_of_atoms()
    print(f"Generating all {N} displacements ... ", end="")
    displacements = atomic_structure2all_displacements(unit_cell, args.amplitude)
    print("done")

    print(f"Writing cartesian displacements to {args.output} ... ", end="")
    np.savetxt(args.output, displacements, fmt=float_format)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
