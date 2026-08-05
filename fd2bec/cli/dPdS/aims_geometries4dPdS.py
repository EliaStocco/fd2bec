"""Generate FHI-aims geometries for finite-strain piezoelectric calculations."""

import argparse
from pathlib import Path

import numpy as np

from fd2bec.cli import cli
from fd2bec.io import read, write
from fd2bec.piezoelectric import build_strained_structures

description = (
    "Generate FHI-aims geometry.in files for proper and improper "
    "piezoelectric tensors from one shared strain set."
)


def prepare_args(descr):
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, required=True, help="periodic input structure")
    parser.add_argument(
        "-a",
        "--amplitude",
        **argv,
        type=float,
        required=False,
        help="amplitude of the cell displacement (default: %(default)s)",
        default=1e-3,
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        default="piezoelectric-geometries",
        help="output folder (default: %(default)s)",
    )
    return parser


@cli(prepare_args, description, deprecated=True)
def main(args):
    reference = read(args.input, index=0)
    structures = build_strained_structures(reference, args.amplitude)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    strains = []
    for number, atoms in enumerate(structures):
        filename = output / f"geometry.n={number}.in"
        write(filename, atoms, format="aims")
        strains.append(np.asarray(atoms.info["strain"]).reshape(-1))
        print(f" - {number:3d}) {filename}")

    np.savetxt(output / "strains.txt", strains)
    print(f"Generated {len(structures)} FHI-aims geometries in '{output}'.")
    print("Use the same k-grid and Berry-phase polarization settings for every geometry.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
