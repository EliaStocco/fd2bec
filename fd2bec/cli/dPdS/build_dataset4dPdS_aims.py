"""Build a polarized strained extxyz dataset from FHI-aims outputs."""

# Tested by pytest: tests/test_piezoelectric_aims.py

import argparse
import re
from pathlib import Path

import numpy as np

from fd2bec.cli import KEYWORDS, cli, extract_n
from fd2bec.io import read, write

description = "Extract FHI-aims dipoles from strained calculations for dPdS2piezo."


AIMS_POLARIZATION = re.compile(
    r"Cartesian Polarization\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


# FHI-aims prints C/m²; this converts to e/Å² before multiplying by volume.
C_PER_M2_TO_E_PER_ANGSTROM2 = 0.06241517271464743


def prepare_args(descr):
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, required=True, help="FHI-aims output folder")
    parser.add_argument(
        "--pattern",
        default="*.out",
        help="output filename glob inside --input (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        default="dataset.extxyz",
        help="polarized extxyz output (default: %(default)s)",
    )
    return parser


def extract_aims_polarization(path: Path) -> np.ndarray:
    """Return the final Cartesian polarization printed by FHI-aims."""
    polarization = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = AIMS_POLARIZATION.search(line)
            if match:
                polarization = np.asarray([float(value) for value in match.groups()])
    if polarization is None:
        raise ValueError(f"No Cartesian Polarization found in '{path}'.")
    return polarization


@cli(prepare_args, description)
def main(args):
    input_path = Path(args.input)
    files = sorted(input_path.glob(args.pattern), key=extract_n)
    if not files:
        raise FileNotFoundError(f"No files matching '{args.pattern}' in '{input_path}'.")

    structures = []
    for number, filename in enumerate(files):
        atoms = read(filename, format="aims-output", index=-1)
        polarization = extract_aims_polarization(filename) * C_PER_M2_TO_E_PER_ANGSTROM2
        dipole = polarization * atoms.get_volume()
        atoms.info[KEYWORDS["dipole"]] = dipole
        atoms.info["source"] = str(filename)
        structures.append(atoms)
        print(f" - {number:3d}) {filename}: {dipole.tolist()} e*Å")

    output = Path(args.output)
    if output.suffix != ".extxyz":
        raise ValueError("The dipole dataset must be an extxyz file.")
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, structures, format="extxyz")
    print(f"Saved {len(structures)} dipole strained structures to '{output}'.")
    print(f"Next run: dPdS2piezo -i {output} -r reference-geometry")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
