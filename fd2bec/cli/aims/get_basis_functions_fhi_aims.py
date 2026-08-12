"""Create the FHI-aims basis-function section for a structure."""

# Tested by pytest: tests/test_create_fhi_aims.py

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional

from fd2bec.cli import cli
from fd2bec.io import read

description = "Create an FHI-aims species.in file from the elements in a structure."


def prepare_args(descr):
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        default="geometry.in",
        help="input structure (default: %(default)s)",
    )
    parser.add_argument(
        "--input-format", "--input_format", **argv, default=None, help="optional ASE input format"
    )
    parser.add_argument(
        "-b", "--basis", **argv, default="light", help="FHI-aims basis set (default: %(default)s)"
    )
    parser.add_argument(
        "-f",
        "--folder",
        **argv,
        default=None,
        help="FHI-aims folder; otherwise use the environment",
    )
    parser.add_argument(
        "-v",
        "--variable",
        **argv,
        default="AIMS_PATH",
        help="environment variable containing the FHI-aims folder (default: %(default)s)",
    )
    parser.add_argument(
        "-o", "--output", **argv, default=None, help="output file (default: species.<basis>.in)"
    )
    return parser


def _aims_species_folder(aims_folder: Path, basis: str) -> Path:
    """Locate the standard FHI-aims species directory."""
    candidates = (
        aims_folder / "species_defaults" / "defaults_2020" / basis,
        aims_folder.parent / "species_defaults" / "defaults_2020" / basis,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find the FHI-aims '{basis}' species directory. Searched:\n{searched}"
    )


def _species_files(species_folder: Path, symbol: str) -> List[Path]:
    """Return matching standard species files for one chemical symbol."""
    files = sorted(species_folder.glob(f"*_{symbol}_*"))
    if not files:
        raise FileNotFoundError(f"No species file found for '{symbol}' in '{species_folder}'.")
    if len(files) > 1:
        matches = ", ".join(path.name for path in files)
        raise ValueError(f"More than one species file found for '{symbol}': {matches}")
    return files


def _unique_species(symbols: Iterable[str]) -> List[str]:
    """Return chemical symbols once, in deterministic order."""
    return sorted(set(symbols))


def create_species_file(
    input_file: str,
    basis: str = "light",
    aims_folder: Optional[str] = None,
    variable: str = "AIMS_PATH",
    output: Optional[str] = None,
    input_format: Optional[str] = None,
) -> Path:
    """Create a species file and return its path."""
    structure = read(input_file, index=0, format=input_format)
    species = _unique_species(structure.get_chemical_symbols())
    if not species:
        raise ValueError("The input structure contains no atoms.")

    folder_value = aims_folder or os.environ.get(variable)
    if not folder_value:
        raise ValueError(f"FHI-aims folder not provided; use --folder or set ${variable}.")
    species_folder = _aims_species_folder(Path(folder_value).expanduser(), basis)

    output_path = Path(output or f"species.{basis}.in")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as target:
        for symbol in species:
            target.write(_species_files(species_folder, symbol)[0].read_text(encoding="utf-8"))
    return output_path


@cli(prepare_args, description)
def main(args):
    """Create an FHI-aims species file from the input structure."""
    output = create_species_file(
        input_file=args.input,
        basis=args.basis,
        aims_folder=args.folder,
        variable=args.variable,
        output=args.output,
        input_format=args.input_format,
    )
    print(f"Wrote FHI-aims basis functions to {output}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
