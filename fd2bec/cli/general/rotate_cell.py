"""Rotate periodic structures into ASE's lower-triangular cell standard form."""

import argparse

import numpy as np
from ase import Atoms

from fd2bec.cli import cli
from fd2bec.io import read, write

description = "Rotate a periodic structure into ASE's lower-triangular cell form."
CELL_ATOL = 1e-10


def prepare_args(descr):
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, required=True, help="path to input structure")
    parser.add_argument("-o", "--output", **argv, required=True, help="path to output structure")
    return parser


def is_ase_standard_cell(atoms: Atoms, atol: float = CELL_ATOL) -> bool:
    """Return whether a periodic cell is in ASE's lower-triangular standard form."""
    if not np.all(atoms.pbc):
        return True

    standard_cell, _ = atoms.cell.standard_form()
    return np.allclose(atoms.cell.array, standard_cell.array, rtol=0.0, atol=atol)


def rotate_to_ase_standard_cell(atoms: Atoms) -> Atoms:
    """Return a copy of ``atoms`` with ASE's lower-triangular standard cell.

    The same rigid rotation is applied to the Cartesian positions, preserving
    fractional coordinates and the physical structure.
    """
    if not np.all(atoms.pbc):
        raise ValueError("Cell rotation requires a fully periodic structure.")

    rotated = atoms.copy()
    standard_cell, rotation = rotated.cell.standard_form()
    rotated.set_positions(rotated.get_positions() @ rotation.T)
    rotated.set_cell(standard_cell, scale_atoms=False)
    return rotated


@cli(prepare_args, description)
def main(args):
    """Run the cell-rotation command."""
    print(f"Reading input structure from {args.input} ... ", end="")
    atoms = read(args.input, index=0)
    print("done")

    print("Rotating cell into ASE standard form ... ", end="")
    rotated = rotate_to_ase_standard_cell(atoms)
    print("done")

    print(f"Writing rotated structure to {args.output} ... ", end="")
    write(args.output, rotated)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
