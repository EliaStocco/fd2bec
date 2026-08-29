"""Build a dP/dR dataset from total dipoles of isolated structures."""

import argparse
import os
import re
from pathlib import Path
from typing import List
from warnings import warn

import numpy as np
from ase import Atoms

from fd2bec.cli import KEYWORDS, cli, extract_n, read_input_structures
from fd2bec.cli.dPdR.build_dataset4dPdR import (
    FORMAT_REGISTRY,
    extract_vectors,
    filtered_temp_file,
    load_format,
)
from fd2bec.io import read, write

description = (
    "Build a dataset of total dipoles and Cartesian displacements for "
    "non-periodic structures.\n"
    "Use this script for molecules (PBC = False, False, False); polarization "
    "formats are intentionally not supported.\n\n"
    "The output is compatible with dPdR2bec.\n"
    f"Available formats: {', '.join(FORMAT_REGISTRY.keys())}\n"
)


def prepare_args(descr):
    parser = argparse.ArgumentParser(
        description=descr, formatter_class=argparse.RawTextHelpFormatter
    )
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, required=True, help="output-file list or folder")
    parser.add_argument("-r", "--reference", **argv, required=True, help="reference structure")
    parser.add_argument(
        "-f",
        "--format",
        **argv,
        required=True,
        help="dipole format key or JSON format definition",
    )
    parser.add_argument("-o", "--output", **argv, required=True, help="output extxyz")
    return parser


def input_files(path: Path):
    if path.is_dir():
        return sorted((file for file in path.iterdir() if file.is_file()), key=extract_n)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as handle:
            return [
                Path(line.strip())
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
    raise FileNotFoundError(f"Input '{path}' is invalid.")


def validate_nonperiodic_structure(reference: Atoms, atoms: Atoms, filename: Path):
    if any(reference.get_pbc()):
        raise ValueError(
            "The reference structure is periodic. "
            "Use build_dataset4dPdR.py for periodic structures."
        )
    if any(atoms.get_pbc()):
        raise ValueError(
            f"Structure '{filename}' is periodic. This script only accepts "
            "PBC = False, False, False."
        )
    if atoms.get_chemical_symbols() != reference.get_chemical_symbols():
        raise ValueError(
            f"Structure '{filename}' does not have the same ordered chemical "
            "symbols as the reference."
        )


@cli(prepare_args, description)
def main(args):
    fmt = load_format(args.format)
    if fmt["type"] != "dipole":
        raise ValueError(
            "This non-periodic builder requires a format with type='dipole'. "
            "Polarization is only defined for periodic structures."
        )

    regex = re.compile(fmt["regex"])
    factor = fmt["factor"]
    reference = read_input_structures(args.reference, label="reference structure")
    if any(reference.get_pbc()):
        raise ValueError(
            "The reference structure is periodic. "
            "Use build_dataset4dPdR.py for periodic structures."
        )

    structures: List[Atoms] = []
    dipoles = []
    filenames = []
    for filename in input_files(Path(args.input)):
        tmp_file = None
        try:
            print(f" - reading file '{filename}'")
            tmp_file = filtered_temp_file(filename, fmt)
            dipole = extract_vectors(tmp_file, regex) * factor
            atoms = read(tmp_file)
            validate_nonperiodic_structure(reference, atoms, filename)
            structures.append(atoms)
            dipoles.append(dipole)
            filenames.append(filename)
        except Exception as error:
            warn(f"Exception while reading '{filename}'. File skipped.")
            print(error)
        finally:
            if tmp_file is not None:
                os.remove(tmp_file)

    if not structures:
        raise ValueError("No readable non-periodic structures were found.")

    dipoles = np.asarray(dipoles)
    print(f"n. read files: {len(structures)}")
    print(f"Using conversion factor to e·Å: {factor}")

    reference_positions = reference.get_positions()
    for atoms, dipole in zip(structures, dipoles):
        atoms.info[KEYWORDS["dipole"]] = dipole
        atoms.arrays[KEYWORDS["displacements"]] = atoms.get_positions() - reference_positions

    print(f"Saving non-periodic dataset to file '{args.output}'")
    write(args.output, structures)


if __name__ == "__main__":
    main()
