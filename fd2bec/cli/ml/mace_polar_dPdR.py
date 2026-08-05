"""Build a finite-displacement dipole dataset with MACE-POLAR."""

import argparse
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from ase import Atoms

from fd2bec.atomic import AtomicStructure
from fd2bec.cli import KEYWORDS, cli
from fd2bec.cli.displacements.generate_displacements import (
    atomic_structure2all_displacements,
    displacements2structures,
)
from fd2bec.io import read, write

description = (
    "Evaluate MACE-POLAR dipoles for Cartesian finite displacements of an "
    "isolated structure. The resulting extxyz dataset can be passed directly "
    "to dPdR2bec."
)


def prepare_args(descr):
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, required=True, help="isolated input structure")
    parser.add_argument(
        "-m",
        "--model",
        **argv,
        default="polar-1-m",
        help=("MACE-POLAR model name, local checkpoint, or URL (default: %(default)s)"),
    )
    parser.add_argument(
        "-a",
        "--amplitude",
        **argv,
        type=float,
        required=False,
        help="amplitude of the displacement (default: %(default)s)",
        default=1e-3,
    )
    parser.add_argument(
        "-d", "--device", **argv, default="cpu", help="torch device (default: %(default)s)"
    )
    parser.add_argument(
        "--default-dtype",
        choices=("float32", "float64"),
        default="float64",
        help="MACE inference precision (default: %(default)s)",
    )
    parser.add_argument(
        "--charge",
        type=float,
        default=None,
        help="total molecular charge; defaults to input metadata or 0",
    )
    parser.add_argument(
        "--spin",
        type=float,
        default=None,
        help="spin multiplicity; defaults to input metadata or 1",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        default="mace-polar-dataset.extxyz",
        help="output extxyz dataset (default: %(default)s)",
    )
    return parser


def _load_mace_polar():
    try:
        from mace.calculators import mace_polar
    except ImportError as error:
        raise ImportError(
            "MACE-POLAR support requires 'mace-torch>=0.3.16' and "
            "'graph-longrange>=0.4.0'. See fd2bec/cli/ml/README.md."
        ) from error
    return mace_polar


def _electronic_state(reference: Atoms, charge: Optional[float], spin: Optional[float]):
    return {
        "charge": reference.info.get("charge", 0.0) if charge is None else charge,
        "spin": reference.info.get("spin", 1.0) if spin is None else spin,
        "external_field": np.zeros(3),
    }


def build_displaced_structures(reference: Atoms, amplitude: float) -> List[Atoms]:
    """Return positive, negative, and zero Cartesian displacements."""
    if amplitude <= 0:
        raise ValueError("The displacement amplitude must be positive.")
    unit_cell = AtomicStructure.from_ase(reference)
    displacements = atomic_structure2all_displacements(unit_cell, amplitude)
    return displacements2structures(reference, displacements, atomic=True)


def evaluate_dipoles(
    reference: Atoms,
    structures: List[Atoms],
    model: str,
    device: str = "cpu",
    default_dtype: str = "float64",
    charge: Optional[float] = None,
    spin: Optional[float] = None,
    calculator_factory: Optional[Callable] = None,
) -> List[Atoms]:
    """Evaluate MACE-POLAR and attach fd2bec dipoles and displacements."""
    if any(reference.get_pbc()):
        raise ValueError(
            "MACE-POLAR total dipoles are only meaningful for non-periodic "
            "structures; periodic inputs cannot be used for dP/dR."
        )

    calculator_factory = calculator_factory or _load_mace_polar()
    calculator = calculator_factory(
        model=model,
        device=device,
        default_dtype=default_dtype,
    )
    state = _electronic_state(reference, charge, spin)
    reference_positions = reference.get_positions()
    evaluated = []

    for number, atoms in enumerate(structures, start=1):
        if any(atoms.get_pbc()):
            raise ValueError("All displaced structures must be non-periodic.")

        atoms.info.update(state)
        atoms.calc = calculator
        atoms.get_potential_energy()
        try:
            dipole = np.asarray(calculator.results["dipole"], dtype=float).reshape(3)
        except (KeyError, ValueError) as error:
            raise ValueError(
                "The MACE-POLAR calculation did not return a 3-vector dipole."
            ) from error
        if not np.all(np.isfinite(dipole)):
            raise ValueError(f"MACE-POLAR returned a non-finite dipole for structure {number}.")

        atoms.info[KEYWORDS["dipole"]] = dipole
        atoms.arrays[KEYWORDS["displacements"]] = atoms.get_positions() - reference_positions
        atoms.calc = None
        evaluated.append(atoms)

    return evaluated


@cli(prepare_args, description)
def main(args):
    output = Path(args.output)
    if output.suffix != ".extxyz":
        raise ValueError(f"'{output}' must be an extxyz file.")

    print(f"Reading reference structure from '{args.input}'")
    reference = read(args.input, index=0)
    if any(reference.get_pbc()):
        raise ValueError(
            "MACE-POLAR total dipoles are only meaningful for non-periodic "
            "structures; periodic inputs cannot be used for dP/dR."
        )

    structures = build_displaced_structures(reference, args.amplitude)
    print(
        f"Evaluating {len(structures)} structures with '{args.model}' "
        f"on '{args.device}' ({args.default_dtype})"
    )
    structures = evaluate_dipoles(
        reference=reference,
        structures=structures,
        model=args.model,
        device=args.device,
        default_dtype=args.default_dtype,
        charge=args.charge,
        spin=args.spin,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, structures, format="extxyz")
    print(f"Saved fd2bec dataset to '{output}'")
    print(f"Next run: dPdR2bec -i {output}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
