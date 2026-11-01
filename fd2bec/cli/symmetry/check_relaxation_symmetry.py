"""Check force and stress residuals from a symmetry-constrained relaxation.

The group-average projection gives the force or stress that can change a
structure without lowering its detected space group.  At convergence of a
relaxation constrained to that space group, this projected quantity should be
zero within the optimiser tolerance.  The complementary residual is reported
separately because it belongs to symmetry-lowering directions.
"""

import argparse

import numpy as np

from fd2bec.atomic import AtomicStructure
from fd2bec.cli import KEYWORDS, cli, positive_int, read_input_structures
from fd2bec.cli.parser import add_shared_argument
from fd2bec.relaxation import TensorProjectionReport, project_relaxation_tensor
from fd2bec.show import print_reference_structure, print_space_group
from fd2bec.tensor import Forces, Stress
from fd2bec.tensor_components import format_numeric_components
from fd2bec.tools import ase2spglib_dataset, tensor_from_atoms

description = """Report raw and symmetry-projected forces and stress for one structure.

At convergence of a relaxation constrained to the detected space group, the
symmetry-projected force and stress must be zero within the chosen tolerance.
The symmetry-breaking residual is also printed.  A non-zero residual is not,
by itself, evidence of a lower-symmetry instability: such an instability is a
curvature (Hessian/phonon) property, and an exact symmetric energy should have
no symmetry-breaking first derivative at a symmetric structure.
"""


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not parsed > 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def prepare_args(descr: str):
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    add_shared_argument(parser, "input_structure")
    add_shared_argument(parser, "structure_index")
    add_shared_argument(parser, "symprec")
    parser.add_argument(
        "--forces-key",
        **argv,
        default=KEYWORDS["forces"],
        help="per-atom forces array key (default: %(default)s)",
    )
    parser.add_argument(
        "--stress-key",
        **argv,
        default=KEYWORDS["stress"],
        help="structure-level stress key (default: %(default)s)",
    )
    parser.add_argument(
        "--tolerance",
        **argv,
        type=_positive_float,
        default=1e-5,
        help="absolute zero-check tolerance in the stored units (default: %(default)s)",
    )
    parser.add_argument(
        "--precision",
        **argv,
        type=positive_int,
        default=4,
        help="significant digits used to display tensor components (default: %(default)s)",
    )
    return parser


def _print_tensor(title: str, tensor, precision: int) -> None:
    print(f"\n{title} ({tensor.basis} basis):")
    tensor.print_components(format_numeric_components(tensor.data, precision))


def _print_summary(name: str, report: TensorProjectionReport, tolerance: float) -> None:
    preserving = report.maximum_symmetrized_component
    breaking = report.maximum_symmetry_breaking_component
    print(f"\n{name} residual summary:")
    print(f"  maximum |original component|            = {report.maximum_original_component:.4e}")
    print(f"  maximum |symmetry-preserving component| = {preserving:.4e}")
    print(f"  maximum |symmetry-breaking component|   = {breaking:.4e}")
    if preserving <= tolerance:
        print(f"  constrained-stationarity check: PASS (<= {tolerance:.1e} in stored units)")
    else:
        print(f"  constrained-stationarity check: FAIL (> {tolerance:.1e} in stored units)")
    if breaking <= tolerance:
        print(f"  symmetry-breaking residual: zero within {tolerance:.1e}")
    else:
        print(
            "  symmetry-breaking residual: non-zero; this is incompatible with an exact "
            "first derivative at the detected symmetry."
        )


def _report_tensor(
    name: str, structure: AtomicStructure, tensor, tolerance: float, precision: int
) -> None:
    report = project_relaxation_tensor(structure, tensor)
    _print_tensor(f"Original {name}", report.original, precision)
    _print_tensor(f"Symmetry-projected {name}", report.symmetrized, precision)
    _print_tensor(f"Symmetry-breaking {name} residual", report.symmetry_breaking, precision)
    _print_summary(name.capitalize(), report, tolerance)


@cli(prepare_args, description)
def main(args: argparse.Namespace):
    atoms = read_input_structures(args.input, index=args.index)
    if not np.all(atoms.get_pbc()):
        raise ValueError("This check requires a fully periodic structure.")

    print_reference_structure(atoms)
    dataset = ase2spglib_dataset(atoms, symprec=args.symprec)
    if dataset is None:
        raise ValueError("spglib could not determine a space group for this structure.")
    print()
    print_space_group(dataset, atoms, args.symprec)

    structure = AtomicStructure.from_ase(atoms, symprec=args.symprec)
    forces, forces_location = tensor_from_atoms(
        atoms,
        args.forces_key,
        "forces",
        Forces,
        Forces.template(len(atoms)),
        "cartesian",
    )
    stress, stress_location = tensor_from_atoms(
        atoms,
        args.stress_key,
        "stress",
        Stress,
        Stress.template(),
        "cartesian",
    )
    print(f"\nForces read from {forces_location}[{args.forces_key!r}].")
    _report_tensor("forces", structure, forces, args.tolerance, args.precision)
    print(f"\nStress read from {stress_location}[{args.stress_key!r}].")
    _report_tensor("stress", structure, stress, args.tolerance, args.precision)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
