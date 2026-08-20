# Tested by pytest: tests/test_generate_symmetry_inequivalent_displacements.py, tests/test_prepare_qe.py, tests/test_aims_workflow_wrappers.py

import argparse

import numpy as np

from fd2bec import float_format
from fd2bec.atomic import AtomicStructure
from fd2bec.cli import cli
from fd2bec.displacements import (
    all_cartesian_displacements,
    all_cell_displacements,
    displacements2structures,
    random_cartesian_displacements,
    symmetry_inequivalent_displacements,
    target_tensor,
    tensor_has_atomic_input,
    tensor_perturbation_shape,
)
from fd2bec.io import read, write
from fd2bec.show import print_displacement_input_structure, print_symmetry_selection

description = "Generate Cartesian atomic or cell displacements and displaced structures."


def prepare_args(descr):

    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="path to input structure",
    )
    parser.add_argument(
        "-a",
        "--amplitude",
        **argv,
        type=float,
        required=False,
        help="Cartesian displacement amplitude in Angstrom (default: %(default)s)",
        default=1e-3,
    )
    parser.add_argument(
        "-w",
        "--what",
        **argv,
        type=str,
        required=False,
        help="target quantity (default: %(default)s)",
        default="bec",
        choices=("bec", "piezo", "forces", "stress", "elastic", "force_constants"),
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--no-symmetry",
        action="store_true",
        help="disable symmetry reduction and use every Cartesian displacement",
    )
    selection.add_argument(
        "-n",
        "--number",
        **argv,
        type=int,
        help="number of normally distributed random displacements",
    )
    parser.add_argument(
        "--seed",
        **argv,
        type=int,
        help="random seed used with --number",
    )
    parser.add_argument(
        "-d",
        "--displacements-output",
        "--displacements",
        **argv,
        type=str,
        required=False,
        help="optional path to a flattened txt displacement table",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        type=str,
        required=True,
        help="path to the multi-frame extxyz output",
    )
    return parser


@cli(prepare_args, description)
def main(args):
    """Generate and save the selected displaced structures."""
    if args.amplitude <= 0:
        raise ValueError("The displacement amplitude must be positive.")
    if args.number is not None and args.number <= 0:
        raise ValueError("The number of random displacements must be positive.")
    if args.seed is not None and args.number is None:
        raise ValueError("--seed can only be used together with --number.")

    print(f"Reading input structure from {args.input} ... ", end="")
    atoms = read(args.input, index=0)
    print("done")
    print_displacement_input_structure(atoms)

    unit_cell = AtomicStructure.from_ase(atoms)
    number_of_atoms = len(unit_cell)

    print(f"Constructing {args.what} tensor ... ", end="")
    tensor = target_tensor(args.what, number_of_atoms)
    print("done")

    number_of_components = int(np.prod(tensor_perturbation_shape(tensor)))
    atomic_input = tensor_has_atomic_input(tensor)
    if args.number is not None:
        selected = random_cartesian_displacements(
            number=args.number,
            number_of_components=number_of_components,
            atomic=atomic_input,
            seed=args.seed,
        )
        print(
            f"Generated {len(selected)} normally distributed random "
            f"{'atomic' if atomic_input else 'lower-triangular cell'} displacements."
        )
    elif args.no_symmetry:
        if atomic_input:
            selected = all_cartesian_displacements(number_of_components)
        else:
            selected = all_cell_displacements()
        candidates = selected
        print(
            f"Symmetry disabled: selected all {len(selected) - 1} signed Cartesian "
            f"basis displacements; {len(selected)} structures including the reference."
        )
    else:
        selected, candidates = symmetry_inequivalent_displacements(unit_cell, tensor)

    selected = selected * args.amplitude

    if args.number is None and not args.no_symmetry:
        print(
            f"Found {len(selected) - 1} unique signed displacements from "
            f"{len(candidates)} symmetry-mode candidates."
        )
        print_symmetry_selection(unit_cell, selected, atomic=atomic_input)

    structures = displacements2structures(atoms, selected, atomic=atomic_input)

    if args.displacements_output is not None:
        print(f"Writing displacements to {args.displacements_output} ... ", end="")
        np.savetxt(args.displacements_output, selected, fmt=float_format)
        print("done")

    print(f"Writing {len(structures)} displaced structures to {args.output} ... ", end="")
    write(args.output, structures, format="extxyz")
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
