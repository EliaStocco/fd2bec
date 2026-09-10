"""Generate parent-symmetry-related orientation variants of a periodic structure."""

import argparse

from fd2bec.cli import cli, read_input_structures
from fd2bec.cli.parser import add_shared_argument
from fd2bec.domain_variants import generate_domain_variants, parent_point_operations
from fd2bec.io import write
from fd2bec.tools import ase2spglib_dataset

description = (
    "Generate the unique orientation-domain variants of a periodic structure from a "
    "parent reference structure."
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
    return parser


@cli(prepare_args, description)
def main(args: argparse.Namespace) -> None:
    """Read a structure and its parent, generate variants, and write all frames."""
    atoms = read_input_structures(args.input, label="distorted structure")
    parent = read_input_structures(args.parent, label="parent reference structure")
    parent_dataset = ase2spglib_dataset(parent, symprec=args.symprec)
    if parent_dataset is None:
        raise ValueError("spglib could not determine the parent space group.")

    operations = parent_point_operations(parent, symprec=args.symprec)
    print(
        f"Using parent space group {parent_dataset.international} "
        f"with {len(operations)} point operations."
    )
    print("Generating parent-symmetry-related variants ... ", end="")
    variants = generate_domain_variants(atoms, parent, symprec=args.symprec)
    print(f"done ({len(variants)} inequivalent variants)")

    print(f"Writing variants to {args.output} ... ", end="")
    write(args.output, variants)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
