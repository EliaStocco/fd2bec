import argparse

import numpy as np

from fd2bec import float_format
from fd2bec.cli import cli
from fd2bec.io import read

description = "Extract an 'info' from a extxyz file and convert it to a txt file."


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
    structures = read(args.input, index=":")
    print("done")

    print(f"Extracting '{args.name}' from the 'info' of the structures ... ", end="")
    data = [None] * len(structures)
    for n, structure in enumerate(structures):
        info = structure.info
        if args.name not in info:
            raise ValueError(
                f"'{args.name}' not found in the 'info' of the structure\n"
                f"Available keys: {list(info.keys())}"
            )
        data[n] = info[args.name]
    print("done")

    data = np.asarray(data)
    data = data.reshape(len(structures), -1)  # flatten the last dimensions, if any

    print(f"Writing cartesian displacements to {args.output} ... ", end="")
    np.savetxt(args.output, data, fmt=float_format)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
