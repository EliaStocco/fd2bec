"""Reorder an ASE structure so that its atom indices match a reference."""

# Tested by pytest: tests/test_sort_structure.py

import argparse

import numpy as np
from ase import Atoms

from fd2bec.atomic import AtomicStructure
from fd2bec.cli import cli
from fd2bec.cli.structures.convert_format import is_ase_standard_cell
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
        default=10,
        help=(
            "maximum positional mismatch (default: %(default)g). This is in "
            "fractional coordinates for periodic structures and Angstrom for molecules."
        ),
    )
    return parser


def require_ase_standard_cell(reference: Atoms) -> None:
    """Raise an actionable error when a periodic reference cell is rotated."""
    if not is_ase_standard_cell(reference):
        raise ValueError(
            "Reference cell is not in ASE's lower-triangular standard form. "
            "Run `rotate_cell -i <reference> -o <rotated-reference>` first."
        )


def sort_atoms_like(reference: Atoms, candidate: Atoms, atol: float = DEFAULT_ATOL) -> Atoms:
    """Return ``candidate`` reordered to have the atom order of ``reference``.

    Slicing the original ASE object keeps its cell, PBC flags, info dictionary,
    constraints, and atom arrays (for example forces, tags, or charges).  For
    periodic structures, every reordered atom is also translated by lattice
    vectors to the image nearest to its corresponding reference atom.
    """
    require_ase_standard_cell(reference)
    reference_structure = AtomicStructure.from_ase(reference)
    candidate_structure = AtomicStructure.from_ase(candidate)
    mapping = reference_structure.get_atoms_mapping(candidate_structure, atol=atol)
    order = np.argsort(mapping)
    ordered = candidate[order]

    if reference_structure.pbc:
        reference_frac_pos = reference.get_scaled_positions(wrap=False)
        candidate_frac_pos = ordered.get_scaled_positions(wrap=False)
        displacement = candidate_frac_pos - reference_frac_pos
        nearest_displacement = displacement - np.floor(displacement + 0.5)
        ordered.set_scaled_positions(reference_frac_pos + nearest_displacement)

    return ordered


@cli(prepare_args, description)
def main(args):
    """Run the structure sorting command."""
    print(f"Reading reference structure from {args.reference} ... ", end="")
    reference = read(args.reference, index=0)
    print("done")

    require_ase_standard_cell(reference)

    print(f"Reading structure to reorder from {args.input} ... ", end="")
    candidate = read(args.input, index=0)
    print("done")

    print("Matching and reordering atoms ... ", end="")
    ordered = sort_atoms_like(reference, candidate, atol=args.atol)
    print("done")

    if np.all(reference.get_pbc()):
        # Translate the whole structure so the first atom matches the reference.
        reference_pos = reference.get_scaled_positions(wrap=False)
        pos = ordered.get_scaled_positions(wrap=False)
        pos -= pos[0] - reference_pos[0]

        # Move each atom to the periodic image nearest to its reference atom.
        diff = pos - reference_pos
        diff = np.mod(diff + 0.5, 1.0) - 0.5
        pos = reference_pos + diff
        ordered.set_scaled_positions(pos)

        assert np.all(np.abs(pos - reference_pos) <= 0.5), "error"

    print(f"Writing reordered structure to {args.output} ... ", end="")
    write(args.output, ordered)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
