from fd2bec import float_format
from fd2bec.cli import cli
from ase.io import read
import numpy as np

description = "Extract BEC from a extxyz file and convert it to a txt file."


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
        "-n",
        "--name",
        **argv,
        type=str,
        required=True,
        help="name of the 'info' (e.g. dipole",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        type=str,
        required=True,
        help="path to txt output file (e.g. dipole.txt)",
    )
    return parser


@cli(prepare_args, description)
def main(args):

    print(f"Reading input structure from {args.input} ... ", end="")
    atoms = read(args.input, index=0)
    print("done")

    print(f"Extracting '{args.name}' from the 'info' of the structures ... ", end="")
    arrays = atoms.arrays
    if args.name not in arrays:
        raise ValueError(
            f"'{args.name}' not found in the 'info' of the structure\n"
            f"Available keys: {list(arrays.keys())}"
        )
    bec = arrays[args.name]
    print("done")
    bec = bec.reshape(len(atoms), 9)

    print(f"Writing BEC to {args.output} ... ", end="")
    np.savetxt(args.output, bec, fmt=float_format)
    print("done")


if __name__ == "__main__":
    main()
