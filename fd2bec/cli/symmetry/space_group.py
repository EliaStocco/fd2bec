"""Display structure information and the space group detected by spglib."""

# Tested by pytest: tests/test_space_group.py

import argparse

import numpy as np

from fd2bec.cli import cli, read_input_structures
from fd2bec.cli.parser import add_shared_argument
from fd2bec.show import (
    print_point_operation,
    print_space_group,
    print_structure,
    print_symmetry_operations,
)
from fd2bec.symmetry import (
    character_table_frame,
    character_table_legend,
    gamma_character_table,
    point_group_generators,
)
from fd2bec.tools import ase2spglib_dataset

description = "Show the cell, atomic positions, and space-group information."


def prepare_args(descr):
    parser = argparse.ArgumentParser(description=descr)
    add_shared_argument(parser, "input_structure")
    add_shared_argument(parser, "symprec")
    parser.add_argument(
        "--show_operations",
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

    generators = point_group_generators(dataset.rotations)
    print(f"\nPoint-group generators ({len(generators)}):")
    if generators:
        for index, rotation in enumerate(generators, start=1):
            print_point_operation(rotation, label=f"{index})")
    else:
        print("  None (the point group contains only the identity).")

    table = gamma_character_table(dataset.rotations)
    df = character_table_frame(table)
    print("\nCharacter table:")
    # DataFrame.__repr__ abbreviates wide character tables with ``...``.
    # These tables are usually small enough to print completely, and every
    # conjugacy class is needed to interpret the irreducible representations.
    print(df.to_string())
    print(character_table_legend())

    if args.show_operations:
        print()
        print_symmetry_operations(dataset)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
