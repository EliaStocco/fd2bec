import numpy as np
from ase.io import read

from fd2bec import float_format
from fd2bec.cli import cli

description = "Generate all cartesian displacements."


def prepare_args(description):
    import argparse

    parser = argparse.ArgumentParser(description=description)
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


@cli(prepare_args, description)
def main(args):

    print(f"Reading input structure from {args.input} ... ", end="")
    atoms = read(args.input, index=0)
    print("done")

    N = 3 * atoms.get_global_number_of_atoms()
    print(f"Generating all {N} displacements ... ", end="")
    displacements = np.eye(N) * args.amplitude
    print("done")

    print(f"Writing cartesian displacements to {args.output} ... ", end="")
    np.savetxt(args.output, displacements, fmt=float_format)
    print("done")


if __name__ == "__main__":
    main()
