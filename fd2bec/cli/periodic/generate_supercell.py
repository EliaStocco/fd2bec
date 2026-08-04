import argparse

import numpy as np
from ase.build import make_supercell

from fd2bec.cli import cli, ilist
from fd2bec.io import read, write

description = """Generate supercell structures from a unit cell structure.
    -t/--type:
    str (default: "cell-major")
    how to order the atoms in the supercell

    "cell-major":
    [atom1_shift1, atom2_shift1, ..., atom1_shift2, atom2_shift2, ...]
    i.e. run first over all the atoms in cell1 and then move to cell2.

    "atom-major":
    [atom1_shift1, atom1_shift2, ..., atom2_shift1, atom2_shift2, ...]
    i.e. run first over atom1 in all the cells and then move to atom2.

    This may be the order preferred by most VASP users.
"""

choices = ["cell-major", "atom-major"]


def prepare_args(descr):

    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="path to input structure (e.g. unitcell.extxyz)",
    )
    parser.add_argument(
        "-s",
        "--supercell",
        **argv,
        type=ilist,
        required=True,
        help="supercell dimensions (e.g. -s 2 2 2)",
    )
    parser.add_argument(
        "-t",
        "--type",
        **argv,
        type=str,
        help=f"order type {choices}" + " (default: %(default)s)",
        default="atom-major",
        choices=choices,
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        type=str,
        required=True,
        help="path to output structure (e.g. supercell.extxyz)",
    )
    return parser


@cli(prepare_args, description)
def main(args):

    print(f"Reading input structure from {args.input} ... ", end="")
    atoms = read(args.input, index=0)
    print("done")

    print(f"Generating supercell with dimensions {args.supercell} ... ", end="")
    matrix = np.diag(args.supercell)
    supercell = make_supercell(atoms, matrix, wrap=False, order=args.type)
    print("done")

    print(f"Writing supercell structure to {args.output} ... ", end="")
    write(args.output, supercell)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
