"""Display structure information and the space group detected by spglib."""

# Tested by pytest: tests/test_space_group.py

import argparse

import numpy as np

from fd2bec.cli import cli
from fd2bec.io import read
from fd2bec.tools import ase2spglib_dataset

description = "Show the cell, atomic positions, and space-group information."


def prepare_args(descr):
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="path to input structure (e.g. unitcell.extxyz)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        **argv,
        type=float,
        default=1e-3,
        help="symmetry tolerance passed to spglib (default: %(default)s)",
    )
    parser.add_argument(
        "--show-operations",
        action="store_true",
        help="print all fractional-coordinate symmetry operations",
    )
    return parser


def _text(value):
    """Convert spglib strings, including values returned as bytes, to text."""
    return value.decode() if isinstance(value, bytes) else str(value)


def _print_matrix(matrix, precision=8):
    for row in np.asarray(matrix):
        print("    " + "  ".join(f"{value:14.{precision}f}" for value in row))


def print_cell(atoms):
    """Print cell vectors, lattice parameters, and volume."""
    cell = np.asarray(atoms.cell.array, dtype=float)
    print("Cell vectors [Angstrom]:")
    for index, vector in enumerate(cell, start=1):
        print(f"  a{index}: " + "  ".join(f"{value:14.8f}" for value in vector))

    a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
    print("\nLattice parameters:")
    print(f"  a, b, c [Angstrom]       = {a:.8f}  {b:.8f}  {c:.8f}")
    print(f"  alpha, beta, gamma [deg] = {alpha:.8f} {beta:.8f} {gamma:.8f}")
    print(f"  volume [Angstrom^3]      = {atoms.get_volume():.8f}")


def print_positions(atoms):
    """Print Cartesian and fractional positions for every atom."""
    cartesian = np.asarray(atoms.get_positions(), dtype=float)
    fractional = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)
    print("Positions (Cartesian [Angstrom] and fractional):")
    print(
        "  index  atom           Rx           Ry           Rz           fx           fy           fz"
    )
    for index, (symbol, position, scaled) in enumerate(
        zip(atoms.get_chemical_symbols(), cartesian, fractional), start=1
    ):
        values = "  ".join(f"{value:11.6f}" for value in (*position, *scaled))
        print(f"  {index:5d}  {symbol:>4s}  {values}")


def _print_space_group(dataset, atoms, threshold):
    """Print a readable summary and details from a spglib symmetry dataset."""
    international = _text(dataset.international)
    # hall = _text(dataset.hall)
    # choice = _text(dataset.choice)
    pointgroup = _text(dataset.pointgroup)
    try:
        bravais = atoms.cell.get_bravais_lattice(eps=threshold)
        bravais_type = bravais.longname
    except (AttributeError, ValueError):
        bravais_type = "undetermined"

    centrosymmetric = any(
        np.array_equal(rotation, -np.eye(3, dtype=int)) for rotation in dataset.rotations
    )

    print("Space-group summary:")
    print(f"  International symbol         : {international}")
    print(f"  number                       : {dataset.number}")
    print(f"  Crystal class                : {pointgroup}")
    print(f"  Bravais lattice type         : {bravais_type}")
    print(f"  Number of symmetry operations: {len(dataset.rotations)}")
    print(f"  Centrosymmetric              : {'yes' if centrosymmetric else 'no'}")
    # print()
    # print("Space-group details:")
    # print(f"  symmetry threshold : {threshold:.6g}")
    # print(f"  number             : {dataset.number}")
    # print(f"  Hall number        : {dataset.hall_number}")
    # print(f"  Hall symbol        : {hall}")
    # if choice:
    #     print(f"  setting choice = {choice}")
    # print(f"  point group = {pointgroup}")
    # primitive_count = len(np.unique(dataset.mapping_to_primitive))
    # conventional_count = len(dataset.std_positions)
    # print(f"  primitive cell atoms = {primitive_count}")
    # print(f"  standardized conventional cell atoms = {conventional_count}")
    # print(f"  input cell is primitive = {len(atoms) == primitive_count}")


def print_symmetry_operations(dataset):
    """Print spglib operations in fractional coordinates."""
    print("Symmetry operations (fractional coordinates x' = R x + t):")
    for index, (rotation, translation) in enumerate(
        zip(dataset.rotations, dataset.translations), start=1
    ):
        print(f"  #{index}")
        print("    rotation:")
        _print_matrix(rotation, precision=0)
        print("    translation: " + "  ".join(f"{value: .8f}" for value in translation))


@cli(prepare_args, description)
def main(args):
    print(f"Reading input structure from {args.input} ... ", end="")
    atoms = read(args.input, index=0)
    print("done")

    print(f"\nStructure information ({atoms.get_chemical_formula()}):")
    print(f"  atoms = {len(atoms)}")
    print(f"  periodic boundary conditions = {atoms.get_pbc().tolist()}")
    print(f"  total mass [amu] = {atoms.get_masses().sum():.8f}")

    if not np.all(atoms.get_pbc()):
        print("\nThis structure is not fully periodic; no space group is computed.")
        print_positions(atoms)
        return

    print()
    print_cell(atoms)
    print()
    print_positions(atoms)

    dataset = ase2spglib_dataset(atoms, symprec=args.threshold)
    if dataset is None:
        raise ValueError("spglib could not determine a space group for this structure.")

    print()
    _print_space_group(dataset, atoms, args.threshold)
    print()
    if args.show_operations:
        print()
        print_symmetry_operations(dataset)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
