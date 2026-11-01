"""Display structure information and the space group detected by spglib."""

# Tested by pytest: tests/test_space_group.py

import argparse

import numpy as np

from fd2bec.cli import cli, read_input_structures
from fd2bec.cli.parser import add_shared_argument
from fd2bec.geometry import cart2frac
from fd2bec.io import write
from fd2bec.mathematics import zero_dipole_fractional_shifts
from fd2bec.show import print_space_group, print_structure
from fd2bec.tools import ase2spglib_dataset

description = "Show the cell, atomic positions, and space-group information."


def prepare_args(descr):
    parser = argparse.ArgumentParser(description=descr)
    add_shared_argument(parser, "input_structure")
    add_shared_argument(parser, "output_structure")
    parser.add_argument(
        "-oxn",
        "--oxidation-numbers",
        metavar="\b",
        required=False,
        default="oxn",
        help="ASE atoms.info key containing the oxidation numbers (default: %(default)s)",
    )
    add_shared_argument(parser, "symprec")
    return parser


@cli(prepare_args, description)
def main(args):
    atoms = read_input_structures(args.input)
    original_atoms = atoms.copy()

    print()
    print_structure(atoms)

    if not np.all(atoms.get_pbc()):
        raise ValueError("This structure is not fully periodic.")

    dataset = ase2spglib_dataset(atoms, symprec=args.symprec)
    if dataset is None:
        raise ValueError("spglib could not determine a space group for this structure.")

    print()
    fields = print_space_group(dataset, atoms, args.symprec)
    if fields["Centrosymmetric"] == "No":
        raise ValueError("The structure is not centrosymmetric; cannot standardize.")

    print("\nExtracting oxidation numbers from atoms.info ... ", end="")
    oxn = np.asarray(atoms.arrays[args.oxidation_numbers], dtype=float)
    if oxn.shape != (len(atoms),):
        raise ValueError(
            f"Oxidation numbers in atoms.info[{args.oxidation_numbers!r}] must have "
            f"shape ({len(atoms)},), not {oxn.shape}."
        )
    print("done")
    for i, atom in enumerate(atoms):
        print(f"\t{atom.symbol:<2}: {oxn[i]}")

    dipole = np.sum(oxn[:, np.newaxis] * atoms.get_positions(), axis=0)
    print(f"\nDipole moment (in e*Å): {dipole}")

    fractional_positions = atoms.get_scaled_positions(wrap=False)
    quanta = fractional_positions.T @ oxn
    print(f"Dipole quanta: {quanta}")
    fractional_dipole = cart2frac(cell=atoms.cell, v=dipole)
    if not np.allclose(quanta, fractional_dipole, atol=1e-8, rtol=0.0):
        raise RuntimeError(
            "Dipole quanta do not match the fractional coordinates of the dipole: "
            f"{quanta} != {fractional_dipole}."
        )

    print("Shifting atoms to cancel the ionic dipole ... ", end="")
    try:
        shifts = zero_dipole_fractional_shifts(oxn, fractional_positions)
        if not np.allclose(shifts, np.rint(shifts), atol=1e-8, rtol=0.0):
            raise ValueError(f"The shifts are not all integers: {shifts}.")
    except ValueError as error:
        output_atoms = original_atoms
        output_atoms.info["standardized-centrosymettric"] = "no"
        print(f"not possible ({error})")
        print("Saving the original structure.")
    else:
        atoms.set_scaled_positions(fractional_positions + shifts)
        dipole = np.sum(oxn[:, np.newaxis] * atoms.get_positions(), axis=0)
        polarization = dipole / atoms.get_volume()
        if not np.allclose(polarization, 0.0, atol=1e-8, rtol=0.0):
            raise RuntimeError(f"Shifts did not cancel the polarization: {polarization}.")
        output_atoms = atoms
        output_atoms.info["standardized-centrosymettric"] = "yes"
        print("done")
        print(f"Dipole after shifting (in e*Å): {dipole}")
        print(f"Polarization after shifting (in e/Å²): {polarization}")

        print("Structure after shifting:")
        print_structure(output_atoms)

    print(f"Writing structure to {args.output} ... ", end="")
    write(args.output, output_atoms)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
