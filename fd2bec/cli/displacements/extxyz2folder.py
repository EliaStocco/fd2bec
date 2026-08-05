"""Write every structure in a multi-frame extxyz file to a separate file."""

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
from ase import Atoms
from ase.io.formats import ioformats

from fd2bec.cli import cli
from fd2bec.io import read, write

ESPRESSO_GEOMETRY_FORMAT = "espresso-in"
ase_writable_formats = sorted(name for name, fmt in ioformats.items() if fmt.can_write)
output_formats = sorted(set(ase_writable_formats) | {ESPRESSO_GEOMETRY_FORMAT})

description = "Write each snapshot from a multi-frame extxyz file to a separate file."


def prepare_args(descr):
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        required=True,
        help="path to the multi-frame extxyz input",
    )
    parser.add_argument(
        "-f",
        "--format",
        **argv,
        default="aims",
        choices=output_formats,
        help="output format (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        default="geometries",
        help="output folder (default: %(default)s)",
    )
    return parser


def espresso_geometry(atoms: Atoms) -> str:
    """Return QE ``CELL_PARAMETERS`` and fractional ``ATOMIC_POSITIONS`` cards."""
    if not np.all(atoms.get_pbc()):
        raise ValueError("espresso-in geometry requires a fully periodic structure.")
    if abs(np.linalg.det(atoms.cell.array)) < 1e-14:
        raise ValueError("espresso-in geometry requires a non-singular cell.")

    lines = ["CELL_PARAMETERS angstrom"]
    for vector in atoms.cell.array:
        lines.append("  " + "  ".join(f"{value:.12f}" for value in vector))

    lines.extend(("", "ATOMIC_POSITIONS crystal"))
    scaled_positions = atoms.get_scaled_positions(wrap=False)
    for symbol, position in zip(atoms.get_chemical_symbols(), scaled_positions):
        coordinates = "  ".join(f"{value:.12f}" for value in position)
        lines.append(f"{symbol:<3s}  {coordinates}")

    return "\n".join(lines) + "\n"


def write_espresso_geometry(filename: Path, atoms: Atoms) -> None:
    """Write the geometry portion of a Quantum Espresso input file."""
    filename.write_text(espresso_geometry(atoms), encoding="utf-8")


def write_snapshots(structures: Iterable[Atoms], output: Path, output_format: str) -> List[Path]:
    """Write snapshots as ``geometry.n=<index>.in`` and return their paths."""
    structures = list(structures)
    if not structures:
        raise ValueError("The input contains no structures.")
    if output_format not in output_formats:
        raise ValueError(f"Unsupported output format: {output_format}.")

    output.mkdir(parents=True, exist_ok=True)
    filenames = []
    for index, atoms in enumerate(structures):
        filename = output / f"geometry.n={index}.in"
        if output_format == ESPRESSO_GEOMETRY_FORMAT:
            write_espresso_geometry(filename, atoms)
        else:
            write(str(filename), atoms, format=output_format)
        filenames.append(filename)
    return filenames


@cli(prepare_args, description)
def main(args):
    """Run the snapshot exporter."""
    print(f"Reading snapshots from {args.input} ... ", end="")
    structures = read(args.input, index=":")
    print(f"done ({len(structures)} structures)")

    output = Path(args.output)
    print(f"Writing {args.format} geometries to {output} ... ", end="")
    filenames = write_snapshots(structures, output, args.format)
    print(f"done ({len(filenames)} files)")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
