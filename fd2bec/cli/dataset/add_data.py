"""Add numeric structure-level or per-atom data to an extxyz trajectory."""

# Tested by pytest: tests/test_add_data.py

import argparse
from pathlib import Path

from fd2bec.cli import cli, read_input_structures
from fd2bec.cli.parser import add_shared_argument
from fd2bec.io import add_extxyz_data, read_numeric_data, write

description = "Add numeric data from a text or CSV file to an extxyz trajectory."


def prepare_args(descr):
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    add_shared_argument(parser, "input_structure")
    add_shared_argument(parser, "data_name")
    add_shared_argument(parser, "data_file")
    add_shared_argument(parser, "data_destination")
    add_shared_argument(parser, "output_structure")
    return parser


@cli(prepare_args, description)
def main(args):
    """Read the trajectory, attach the supplied data, and write it back."""
    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.suffix.lower() != ".extxyz" or output_path.suffix.lower() != ".extxyz":
        raise ValueError("Both input and output must use the .extxyz extension.")

    structures = read_input_structures(input_path, index=":", input_format="extxyz")
    if len(structures) != 1:
        raise ValueError("add_data requires an extxyz input containing exactly one structure.")
    data = read_numeric_data(args.data)
    updated = add_extxyz_data(structures, data, args.name, args.what)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, updated[0], format="extxyz")
    print(f"Added {args.what} data '{args.name}' to '{output_path}'.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
