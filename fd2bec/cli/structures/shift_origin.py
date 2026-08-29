"""Shift a periodic structure so its first atom is at the fractional origin."""

# Tested by pytest: tests/test_shift_origin.py

import argparse

from fd2bec.cli import cli, read_input_structures
from fd2bec.cli.parser import add_shared_argument
from fd2bec.io import write
from fd2bec.tools import shift_first_atom_to_origin

description = (
    "Translate a periodic structure so atom 0 has fractional coordinates (0, 0, 0), "
    "wrapping every atom into the unchanged input cell."
)


def prepare_args(descr: str):
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    add_shared_argument(parser, "input_structure")
    add_shared_argument(parser, "output_structure")
    add_shared_argument(parser, "structure_index")
    return parser


@cli(prepare_args, description)
def main(args: argparse.Namespace):
    """Read, translate, and write one periodic structure."""
    atoms = read_input_structures(args.input, index=args.index)

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
