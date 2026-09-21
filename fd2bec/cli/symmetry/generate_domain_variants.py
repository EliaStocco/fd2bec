"""Generate parent-symmetry-related orientation variants of a periodic structure."""

import argparse
from collections import Counter

from fd2bec.cli import cli, read_input_structures
from fd2bec.cli.parser import add_shared_argument
from fd2bec.domain_variants import (
    common_space_group_symbol,
    generate_domain_variants_with_operations,
    lost_parent_point_operation_generators,
    lost_parent_point_operations,
    parent_point_operations,
    retained_parent_point_operations,
)
from fd2bec.io import write
from fd2bec.show import print_compact_structure, print_point_operation, print_structure
from fd2bec.symmetry import point_operation_name
from fd2bec.tools import ase2spglib_dataset

description = (
    "Generate all unique orientation-domain variants of a distorted periodic "
    "structure. The script applies the point-symmetry operations of a supplied "
    "parent-phase reference structure, then removes symmetry-equivalent duplicates. "
    "The resulting variants are written as separate frames in the output structure "
    "file."
)


def prepare_args(descr: str) -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    add_shared_argument(parser, "input_structure")
    add_shared_argument(parser, "output_structure")
    add_shared_argument(parser, "symprec")
    parser.add_argument(
        "-p",
        "--parent",
        metavar="\b",
        required=True,
        help="parent-phase reference structure defining the symmetry and coordinate axes",
    )
    parser.add_argument(
        "--show-differences",
        action="store_true",
        help="show cell and fractional-coordinate differences from the parent for each variant",
    )
    return parser


@cli(prepare_args, description)
def main(args: argparse.Namespace) -> None:
    """Read a structure and its parent, generate variants, and write all frames."""
    atoms = read_input_structures(args.input, label="distorted structure")
    parent = read_input_structures(args.parent, label="parent reference structure")
    print()
    print_structure(atoms, title="Distorted input structure")
    print()
    print_structure(parent, title="Parent reference structure")
    parent_dataset = ase2spglib_dataset(parent, symprec=args.symprec)
    if parent_dataset is None:
        raise ValueError("spglib could not determine the parent space group.")

    operations = parent_point_operations(parent, symprec=args.symprec)
    print(
        f"Using parent space group {parent_dataset.international} "
        f"with {len(operations)} point operations."
    )
    retained_operations = retained_parent_point_operations(atoms, parent, symprec=args.symprec)
    lost_operations = lost_parent_point_operations(atoms, parent, symprec=args.symprec)
    lost_generators = lost_parent_point_operation_generators(atoms, parent, symprec=args.symprec)
    print(
        f"The distorted input retains {len(retained_operations)} parent point operations "
        f"and loses {len(lost_operations)}."
    )
    if lost_operations:
        print("Lost parent point-operation types:")
        names = Counter(
            point_operation_name(rotation, coordinate_frame="parent")
            for rotation, _ in lost_operations
        )
        for name, count in sorted(names.items()):
            print(f"  {count} x {name}")
        print("\nSymmetry-breaking generators:")
        for index, (rotation, translation) in enumerate(lost_generators, start=1):
            print_point_operation(
                rotation, translation, label=f"{index})", coordinate_frame="parent"
            )
        print("  Together with the retained operations, these generate the parent point group.")
    print("\nGenerating parent-symmetry-related variants ... ", end="")
    variants_with_operations = generate_domain_variants_with_operations(
        atoms, parent, symprec=args.symprec
    )
    variants = [variant for variant, _, _ in variants_with_operations]
    print(f"done ({len(variants)} inequivalent variants)")
    space_group_symbol = common_space_group_symbol(variants, symprec=args.symprec)
    print(f"All generated structures have space group {space_group_symbol}.")
    print("\nGenerated structures:")
    for index, (variant, rotation, translation) in enumerate(variants_with_operations, start=1):
        print_compact_structure(
            variant,
            title=f"Variant {index}",
            space_group_symbol=space_group_symbol,
            parent=parent if args.show_differences else None,
        )
        print_point_operation(
            rotation, translation, label="parent operation", coordinate_frame="parent"
        )

    print(f"\nWriting variants to {args.output} ... ", end="")
    write(args.output, variants)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
