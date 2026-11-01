"""Reorder an XYZ file using a sorting map produced by ``sort_structure``."""

import argparse

from fd2bec.cli import cli, read_input_structures
from fd2bec.cli.parser import add_shared_argument
from fd2bec.io import write
from fd2bec.structure_alignment import apply_sorting_map, read_sorting_map

description = (
    "Reorder every frame of an XYZ file using the atom-order map written by "
    "sort_structure. Coordinates are treated as data, so this can be used for "
    "velocity or momentum XYZ files."
)


def prepare_args(descr):
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-m",
        "--sorting-map",
        "--sorting-info",
        dest="sorting_map",
        **argv,
        required=True,
        help="JSON sorting map written by sort_structure",
    )
    add_shared_argument(parser, "input_structure")
    add_shared_argument(parser, "output_structure")
    return parser


@cli(prepare_args, description)
def main(args):
    """Read a data XYZ file, reorder every frame, and write the result."""
    sorting_indices = read_sorting_map(args.sorting_map)
    structures = read_input_structures(args.input, index=":", label="input structures")

    print("Applying sorting map ... ", end="")
    ordered = []
    for frame, atoms in enumerate(structures):
        try:
            ordered.append(apply_sorting_map(atoms, sorting_indices))
        except ValueError as error:
            raise ValueError(f"Cannot sort frame {frame}: {error}") from error
    print("done")

    print(f"Writing reordered structures to {args.output} ... ", end="")
    write(args.output, ordered)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
