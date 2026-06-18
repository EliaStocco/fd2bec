import argparse
from typing import Tuple

import numpy as np

from fd2bec import float_format
from fd2bec.atomic import AtomicStructure
from fd2bec.cli import cli
from fd2bec.io import read
from fd2bec.tensor import Tensor

description = "Generate all symmetry inequivalent cartesian displacements."


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
        help="amplitude of the displacement (default: %(default)s)",
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
        choices=["bec"],
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        type=str,
        required=False,
        help="path to txt output file with cartesian displacements (default: %(default)s)",
        default="displacements.txt",
    )
    return parser


def atomic_structure2unique_displacements(
    unit_cell: AtomicStructure, tensor: Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate all symmetry inequivalent cartesian displacements
    for the given unit cell and amplitude."""

    _, theta, theta_real = unit_cell.get_symmetrizer(tensor=tensor)

    theta_real = theta_real.reshape((-1, len(unit_cell), 3, 3))

    displacements = np.zeros((len(theta), 3 * len(unit_cell)))
    for n, t in enumerate(theta_real):
        displacements[n] = np.sum(t != 0, axis=2).flatten()

    displacements: np.ndarray = displacements / np.linalg.norm(displacements, axis=1)[:, None]

    displacements = np.concatenate(
        [np.zeros((1, 3 * len(unit_cell))), displacements, -displacements.copy()], axis=0
    )

    u = np.unique(displacements, axis=0)

    return u, displacements


@cli(prepare_args, description)
def main(args):

    print(f"Reading input structure from {args.input} ... ", end="")
    atoms = read(args.input, index=0)
    print("done")

    unit_cell = AtomicStructure.from_ase(atoms)
    Na = len(unit_cell)

    if args.what == "bec":
        print("Constructing Born Effective Charges ... ", end="")
        from fd2bec.tensor import BornCharges

        data = np.zeros((Na, 3, 3))
        x = BornCharges(data=data)
        print("done")

    u, d = atomic_structure2unique_displacements(unit_cell, tensor=x)
    u *= args.amplitude

    print(
        f"Found {u.shape[0]} symmetry inequivalent displacements out "
        + f"of {d.shape[0]} total displacements."
    )

    print(f"Writing cartesian displacements to {args.output} ... ", end="")
    np.savetxt(args.output, u, fmt=float_format)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
