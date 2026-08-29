"""Display structure information and the space group detected by spglib."""

# Tested by pytest: tests/test_space_group.py

import argparse

import numpy as np

from fd2bec.cli import cli, read_input_structures
from fd2bec.cli.parser import add_shared_argument
from fd2bec.show import print_space_group, print_structure, print_symmetry_operations
from fd2bec.tools import ase2spglib_dataset

description = "Show the cell, atomic positions, and space-group information."


def prepare_args(descr):
    parser = argparse.ArgumentParser(description=descr)
    add_shared_argument(parser, "input_structure")
    add_shared_argument(parser, "symprec")
    parser.add_argument(
        "--show-operations",
        action="store_true",
        help="print all fractional-coordinate symmetry operations",
    )
    return parser


@cli(prepare_args, description)
def main(args):
    atoms = read_input_structures(args.input)

    print()
    print_structure(atoms)

    if not np.all(atoms.get_pbc()):
        print("\nThis structure is not fully periodic; no space group is computed.")
        return

    dataset = ase2spglib_dataset(atoms, symprec=args.symprec)
    if dataset is None:
        raise ValueError("spglib could not determine a space group for this structure.")

    print()
    print_space_group(dataset, atoms, args.symprec)
    print()
    if args.show_operations:
        print()
        print_symmetry_operations(dataset)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
