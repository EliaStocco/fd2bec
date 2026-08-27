"""Add Born effective charges from a text file to an extxyz structure."""

# Tested by pytest: tests/test_add_tensors.py

import argparse
from pathlib import Path

from fd2bec.cli import KEYWORDS, cli
from fd2bec.io import add_born_effective_charges, read, read_numeric_data, write

description = "Add BECs from a text file under the default REF_BEC key."


def prepare_args(descr):
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, required=True, help="input extxyz file")
    parser.add_argument(
        "-d", "--data", **argv, required=True, help="BEC text file (n_atoms rows × 9)"
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        help="output extxyz file (default: overwrite the input)",
    )
    return parser


@cli(prepare_args, description)
def main(args):
    """Read BECs, add them under the default key, and write the trajectory."""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    if input_path.suffix.lower() != ".extxyz" or output_path.suffix.lower() != ".extxyz":
        raise ValueError("Both input and output must use the .extxyz extension.")

    structures = read(input_path, format="extxyz", index=":")
    if len(structures) != 1:
        raise ValueError("add_bec requires an extxyz input containing exactly one structure.")
    updated = add_born_effective_charges(
        structures,
        read_numeric_data(args.data),
        key=KEYWORDS["bec"],
    )
    write(output_path, updated[0], format="extxyz")
    print(f"Added BECs under '{KEYWORDS['bec']}' to '{output_path}'.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
