"""Align and reorder an ASE structure to match a reference."""

# Tested by pytest: tests/test_sort_structure.py

import argparse
from pathlib import Path

from fd2bec.cli import cli, read_input_structures
from fd2bec.cli.parser import add_shared_argument
from fd2bec.io import write
from fd2bec.structure_alignment import sort_atoms_like_with_indices, write_sorting_map

description = (
    "Align and reorder a structure so its atom order matches a reference structure, "
    "and save the resulting atom-order map.\n\n"
    "Example:\n"
    "  sort_structure -r reference.extxyz -i positions.extxyz -o positions-sorted.extxyz\n\n"
    "This also writes positions-sorted.sorting.json, which can reorder a velocity or "
    "momentum XYZ file with:\n"
    "  apply_sorting_map "
    "-m positions-sorted.sorting.json -i velocities.xyz -o velocities-sorted.xyz"
)


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
    parser.add_argument(
        "-m",
        "--sorting-map",
        "--sorting-info",
        dest="sorting_map",
        **argv,
        help=(
            "path to the JSON sorting map (default: <output>.sorting.json); it can be used "
            "to reorder velocity or momentum XYZ files"
        ),
    )
    return parser


@cli(prepare_args, description)
def main(args):
    """Run the structure sorting command."""
    reference = read_input_structures(args.reference, label="reference structure")
    candidate = read_input_structures(args.input, label="structure to reorder")

    print("Aligning, matching, and reordering atoms ... ", end="")
    ordered, sorting_indices = sort_atoms_like_with_indices(reference, candidate, atol=args.atol)
    print("done")

    print(f"Writing reordered structure to {args.output} ... ", end="")
    write(args.output, ordered)
    print("done")

    output = Path(args.output)
    sorting_map = (
        Path(args.sorting_map) if args.sorting_map else output.with_suffix(".sorting.json")
    )
    print(f"Writing sorting map to {sorting_map} ... ", end="")
    write_sorting_map(sorting_map, reference, candidate, sorting_indices)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
