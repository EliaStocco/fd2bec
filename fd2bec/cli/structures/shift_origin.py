"""Shift a periodic structure so its first atom is at the fractional origin."""

# Tested by pytest: tests/test_shift_origin.py

import argparse

from fd2bec.cli import cli
from fd2bec.io import read, write
from fd2bec.tools import shift_first_atom_to_origin

description = (
    "Translate a periodic structure so atom 0 has fractional coordinates (0, 0, 0), "
    "wrapping every atom into the unchanged input cell."
)


def prepare_args(descr: str):
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, required=True, help="path to input structure")
    parser.add_argument("-o", "--output", **argv, required=True, help="path to shifted structure")
    parser.add_argument(
        "--index",
        **argv,
        type=int,
        default=0,
        help="index of the input structure to shift (default: %(default)s)",
    )
    return parser


@cli(prepare_args, description)
def main(args: argparse.Namespace):
    """Read, translate, and write one periodic structure."""
    print(f"Reading structure {args.index} from {args.input} ... ", end="")
    atoms = read(args.input, index=args.index)
    print("done")

    origin = atoms.get_scaled_positions(wrap=False)[0]
    print(
        "Shifting fractional origin by "
        + "["
        + ", ".join(f"{-value:.12g}" for value in origin)
        + "] ... ",
        end="",
    )
    shifted = shift_first_atom_to_origin(atoms)
    print("done")

    print(f"Writing shifted structure to {args.output} ... ", end="")
    write(args.output, shifted)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
