"""Add proper piezoelectric tensors from a text file to an extxyz structure."""

# Tested by pytest: tests/test_add_tensors.py

import argparse
from pathlib import Path

from fd2bec.cli import KEYWORDS, cli, read_input_structures
from fd2bec.cli.parser import add_shared_argument
from fd2bec.io import add_proper_piezoelectric_tensors, read_numeric_data, write

description = "Add proper piezoelectric tensors from text under REF_piezoelectric."


def prepare_args(descr):
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    add_shared_argument(parser, "input_structure")
    add_shared_argument(parser, "data_file")
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        help="output extxyz file (default: overwrite the input)",
    )
    return parser


@cli(prepare_args, description)
def main(args):
    """Read proper tensors, add them under the default key, and write the trajectory."""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    if input_path.suffix.lower() != ".extxyz" or output_path.suffix.lower() != ".extxyz":
        raise ValueError("Both input and output must use the .extxyz extension.")

    structures = read_input_structures(input_path, index=":", input_format="extxyz")
    if len(structures) != 1:
        raise ValueError("add_piezo requires an extxyz input containing exactly one structure.")
    updated = add_proper_piezoelectric_tensors(
        structures,
        read_numeric_data(args.data),
        key=KEYWORDS["piezoelectric"],
    )
    write(output_path, updated[0], format="extxyz")
    print(
        f"Added proper piezoelectric tensors under '{KEYWORDS['piezoelectric']}' "
        f"to '{output_path}'."
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
