import argparse
from pathlib import Path
from typing import List

import numpy as np
from ase import Atoms
from ase.io.formats import ioformats

from fd2bec.cli import cli
from fd2bec.io import read, write

writable_formats = sorted(name for name, fmt in ioformats.items() if fmt.can_write)


description = "Apply the displacements to an atomic structure."


def prepare_args(descr):

    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="path to input structure (e.g. supercell.extxyz)",
    )
    parser.add_argument(
        "-d",
        "--displacements",
        **argv,
        type=str,
        required=True,
        help="path to cartesian displacements (e.g. displacement.txt)",
    )
    parser.add_argument(
        "-f",
        "--format",
        **argv,
        type=str,
        required=False,
        help="ASE output format (default: %(default)s)",
        default="aims",
        choices=writable_formats,
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        type=str,
        required=False,
        help="folder that will contain all geometries (default: %(default)s)",
        default="geometries",
    )
    return parser


def displacements2atoms(atoms: Atoms, displacements: np.ndarray) -> List[Atoms]:
    """Apply the displacements to the input structure and return a list of displaced structures."""

    number = displacements.shape[0]
    displ = displacements.reshape(number, atoms.get_global_number_of_atoms(), 3)

    displaced_structures = [None] * number
    for i in range(number):
        displaced = atoms.copy()
        displaced.set_positions(displaced.get_positions() + displ[i])
        displaced_structures[i] = displaced

    return displaced_structures


@cli(prepare_args, description)
def main(args):

    print(f"Reading input structure from {args.input} ... ", end="")
    atoms = read(args.input, index=0)
    print("done")

    print(f"Reading cartesian displacements from {args.displacements} ... ", end="")
    displacements = np.loadtxt(args.displacements)
    number = displacements.shape[0]
    print("done")
    assert displacements.shape == (number, atoms.get_global_number_of_atoms() * 3), (
        f"Displacement file shape mismatch\n"
        f"Expected shape: ({number}, {atoms.get_global_number_of_atoms() * 3})\n"
        f"Got shape: {displacements.shape}\n"
        f"File: {args.displacements}"
    )

    displaced_structures = displacements2atoms(atoms, displacements)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Writing {len(displaced_structures)} displaced structures "
        f"to {output_dir} in {args.format} format ...",
        end="",
    )

    for i, atoms in enumerate(displaced_structures):
        if args.format == "aims":
            filename = output_dir / f"geometry.n={i}.in"
        else:
            filename = output_dir / f"structure.n={i}.{args.format}"

        write(
            str(filename),
            atoms,
            format=args.format,
        )

    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
