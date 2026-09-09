"""Align and reorder an ASE structure to match a reference."""

# Tested by pytest: tests/test_sort_structure.py

import argparse

from fd2bec.cli import cli, read_input_structures
from fd2bec.cli.parser import add_shared_argument
from fd2bec.io import write
from fd2bec.structure_alignment import sort_atoms_like

description = "Align and reorder a structure so its atom order matches a reference structure."


def prepare_args(descr):
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-r",
        "--reference",
        **argv,
        required=True,
        help="path to the reference structure, whose atom order is retained",
    )
    add_shared_argument(parser, "input_structure")
    add_shared_argument(parser, "output_structure")
    parser.add_argument(
        "--atol",
        **argv,
        type=float,
        default=10,
        help=(
            "maximum positional mismatch (default: %(default)g). This is in "
            "fractional coordinates for periodic structures and Angstrom for molecules."
        ),
    )
    return parser


@cli(prepare_args, description)
def main(args):
    """Run the structure sorting command."""
    reference = read_input_structures(args.reference, label="reference structure")
    candidate = read_input_structures(args.input, label="structure to reorder")

    print("Aligning, matching, and reordering atoms ... ", end="")
    ordered = sort_atoms_like(reference, candidate, atol=args.atol)
    print("done")

    print(f"Writing reordered structure to {args.output} ... ", end="")
    write(args.output, ordered)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
