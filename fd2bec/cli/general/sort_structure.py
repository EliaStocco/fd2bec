"""Reorder an ASE structure so that its atom indices match a reference."""

import argparse

import numpy as np
from ase import Atoms

from fd2bec.atomic import AtomicStructure
from fd2bec.cli import cli
from fd2bec.io import read, write


description = "Reorder a structure so its atom order matches a reference structure."
DEFAULT_ATOL = 1e-2


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
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        required=True,
        help="path to the structure to reorder",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        required=True,
        help="path for the reordered structure",
    )
    parser.add_argument(
        "--atol",
        **argv,
        type=float,
        default=DEFAULT_ATOL,
        help=(
            "maximum positional mismatch (default: %(default)g). This is in "
            "fractional coordinates for periodic structures and Angstrom for molecules."
        ),
    )
    return parser


def sort_atoms_like(reference: Atoms, candidate: Atoms, atol: float = DEFAULT_ATOL) -> Atoms:
    """Return ``candidate`` reordered to have the atom order of ``reference``.

    Slicing the original ASE object keeps its cell, PBC flags, info dictionary,
    constraints, and atom arrays (for example forces, tags, or charges).
    """
    reference_structure = AtomicStructure.from_ase(reference)
    candidate_structure = AtomicStructure.from_ase(candidate)
    order = np.argsort(reference_structure.get_atoms_mapping(candidate_structure, atol=atol))
    return candidate[order]


@cli(prepare_args, description)
def main(args):
    """Run the structure sorting command."""
    print(f"Reading reference structure from {args.reference} ... ", end="")
    reference = read(args.reference, index=0)
    print("done")

    print(f"Reading structure to reorder from {args.input} ... ", end="")
    candidate = read(args.input, index=0)
    print("done")

    print("Matching and reordering atoms ... ", end="")
    ordered = sort_atoms_like(reference, candidate, atol=args.atol)
    print("done")

    print(f"Writing reordered structure to {args.output} ... ", end="")
    write(args.output, ordered)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
