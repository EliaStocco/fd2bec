"""Generate a shared finite-strain dataset for both piezoelectric tensors."""

import argparse
from pathlib import Path

from fd2bec.cli import cli
from fd2bec.io import read, write
from fd2bec.piezoelectric import build_strained_structures

description = (
    "Generate one reference and positive/negative versions of all six symmetric "
    "strain modes for proper and improper piezoelectric calculations."
)


def prepare_args(descr):
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, required=True, help="periodic input structure")
    parser.add_argument(
        "-a",
        "--amplitude",
        **argv,
        type=float,
        default=5e-3,
        help="engineering-strain amplitude (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        default="strained-structures.extxyz",
        help="multi-frame extxyz output (default: %(default)s)",
    )
    return parser


@cli(prepare_args, description)
def main(args):
    output = Path(args.output)
    if output.suffix != ".extxyz":
        raise ValueError("The strained-structure dataset must be an extxyz file.")

    reference = read(args.input, index=0)
    structures = build_strained_structures(reference, args.amplitude)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, structures, format="extxyz")

    print(f"Generated {len(structures)} structures using amplitude {args.amplitude:g}.")
    print(f"Saved the shared proper/improper piezoelectric dataset to '{output}'.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
