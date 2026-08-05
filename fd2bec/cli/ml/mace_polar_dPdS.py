"""Evaluate a finite-strain polarization proxy with MACE-POLAR."""

import argparse
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from ase import Atoms

from fd2bec.cli import KEYWORDS, cli
from fd2bec.cli.ml.mace_polar_dPdR import _electronic_state, _load_mace_polar
from fd2bec.io import read, write
from fd2bec.piezoelectric import build_strained_structures

description = (
    "Evaluate MACE-POLAR cell-dipole/volume polarization proxies for one shared "
    "set of strained periodic structures."
)


def prepare_args(descr):
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, required=True, help="periodic input structure")
    parser.add_argument(
        "-m",
        "--model",
        **argv,
        default="polar-1-m",
        help="MACE-POLAR model name, checkpoint, or URL (default: %(default)s)",
    )
    parser.add_argument(
        "-a",
        "--amplitude",
        **argv,
        type=float,
        required=False,
        help="amplitude of the cell displacement (default: %(default)s)",
        default=1e-3,
    )
    parser.add_argument("-d", "--device", **argv, default="cpu", help="torch device")
    parser.add_argument(
        "--default-dtype",
        choices=("float32", "float64"),
        default="float64",
        help="MACE inference precision (default: %(default)s)",
    )
    parser.add_argument("--charge", type=float, default=None, help="total cell charge")
    parser.add_argument("--spin", type=float, default=None, help="cell spin multiplicity")
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        default="mace-polar-piezoelectric.extxyz",
        help="polarized strained extxyz output (default: %(default)s)",
    )
    return parser


def evaluate_polarizations(
    reference: Atoms,
    structures: List[Atoms],
    model: str,
    device: str = "cpu",
    default_dtype: str = "float64",
    charge: Optional[float] = None,
    spin: Optional[float] = None,
    calculator_factory: Optional[Callable] = None,
) -> List[Atoms]:
    """Attach MACE-POLAR dipole/volume values as polarization proxies."""
    if not np.all(reference.get_pbc()):
        raise ValueError("The MACE-POLAR piezoelectric workflow requires full periodicity.")

    calculator_factory = calculator_factory or _load_mace_polar()
    calculator = calculator_factory(
        model=model,
        device=device,
        default_dtype=default_dtype,
    )
    state = _electronic_state(reference, charge, spin)
    evaluated = []

    for number, atoms in enumerate(structures, start=1):
        if not np.all(atoms.get_pbc()):
            raise ValueError("Every strained structure must be fully periodic.")
        atoms.info.update(state)
        atoms.calc = calculator
        atoms.get_potential_energy()
        try:
            dipole = np.asarray(calculator.results["dipole"], dtype=float).reshape(3)
        except (KeyError, ValueError) as error:
            raise ValueError(
                "The MACE-POLAR calculation did not return a three-component dipole."
            ) from error
        if not np.all(np.isfinite(dipole)):
            raise ValueError(f"MACE-POLAR returned a non-finite dipole for structure {number}.")

        atoms.info[KEYWORDS["dipole"]] = dipole
        atoms.info[KEYWORDS["polarization"]] = dipole / atoms.get_volume()
        atoms.calc = None
        evaluated.append(atoms)
    return evaluated


@cli(prepare_args, description)
def main(args):
    output = Path(args.output)
    if output.suffix != ".extxyz":
        raise ValueError("The MACE-POLAR piezoelectric dataset must be an extxyz file.")

    reference = read(args.input, index=0)
    structures = build_strained_structures(reference, args.amplitude)
    structures = evaluate_polarizations(
        reference,
        structures,
        model=args.model,
        device=args.device,
        default_dtype=args.default_dtype,
        charge=args.charge,
        spin=args.spin,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, structures, format="extxyz")
    print(f"Saved {len(structures)} MACE-POLAR strained structures to '{output}'.")
    print(f"Next run: dPdS2piezo -i {output} -r {args.input}")
    print(
        "Warning: dipole/volume is a model polarization proxy, not a periodic "
        "Berry-phase polarization."
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
