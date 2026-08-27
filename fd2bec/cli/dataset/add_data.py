"""Add numeric structure-level or per-atom data to an extxyz trajectory."""

# Tested by pytest: tests/test_add_data.py

import argparse
from pathlib import Path

from fd2bec.cli import cli
from fd2bec.io import add_extxyz_data, read, read_numeric_data, write

description = "Add numeric data from a text or CSV file to an extxyz trajectory."


def prepare_args(descr):
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, required=True, help="input extxyz file")
    parser.add_argument("-n", "--name", **argv, required=True, help="name of the new info or array")
    parser.add_argument("-d", "--data", **argv, required=True, help="numeric text or CSV data file")
    parser.add_argument(
        "-w",
        "--what",
        **argv,
        required=True,
        choices=("i", "info", "a", "array", "arrays"),
        help="store data as structure info ('i') or a per-atom array ('a')",
    )
    parser.add_argument("-o", "--output", **argv, required=True, help="output extxyz file")
    return parser


@cli(prepare_args, description)
def main(args):
    """Read the trajectory, attach the supplied data, and write it back."""
    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.suffix.lower() != ".extxyz" or output_path.suffix.lower() != ".extxyz":
        raise ValueError("Both input and output must use the .extxyz extension.")

    structures = read(input_path, format="extxyz", index=":")
    if len(structures) != 1:
        raise ValueError("add_data requires an extxyz input containing exactly one structure.")
    data = read_numeric_data(args.data)
    updated = add_extxyz_data(structures, data, args.name, args.what)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, updated[0], format="extxyz")
    print(f"Added {args.what} data '{args.name}' to '{output_path}'.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
