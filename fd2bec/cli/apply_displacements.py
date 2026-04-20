from fd2bec.cli import cli
from ase.io import read, write
import numpy as np

description = "Apply the displacements to an atomic structure."


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
        "-d",
        "--displacements",
        **argv,
        type=str,
        required=True,
        help="path to cartesian displacements (e.g. displacement.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        type=str,
        required=True,
        help="path to output displaced structure (e.g. supercell-displaced.extxyz)",
    )
    return parser


@cli(prepare_args, description)
def main(args):

    print(f"Reading input structure from {args.input} ... ", end="")
    atoms = read(args.input, index=0)
    print("done")

    print(f"Reading cartesian displacements from {args.displacements} ... ", end="")
    displacements = np.loadtxt(args.displacements)
    number = displacements.shape[0]
    print("done")
    assert displacements.shape == (number, atoms.get_global_number_of_atoms() * 3), (
        f"Displacement file shape mismatch\n"
        f"Expected shape: ({number}, {atoms.get_global_number_of_atoms() * 3})\n"
        f"Got shape: {displacements.shape}\n"
        f"File: {args.displacements}"
    )
    displacements = displacements.reshape(number, atoms.get_global_number_of_atoms(), 3)

    displaced_structures = [None] * number
    for i in range(number):
        displaced = atoms.copy()
        displaced.set_positions(displaced.get_positions() + displacements[i])
        displaced_structures[i] = displaced

    print(f"Writing displaced structures to {args.output} ... ", end="")
    write(args.output, displaced_structures)
    print("done")


if __name__ == "__main__":
    main()
