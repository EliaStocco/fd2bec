"""Add species-based oxidation numbers to every structure in an extxyz file."""

# Tested by pytest: tests/test_add_oxidation_numbers.py

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from ase import Atoms

from fd2bec.cli import cli
from fd2bec.io import read, write

description = "Add oxidation numbers from a species-to-charge JSON file to extxyz arrays."


def prepare_args(descr):
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, required=True, help="input extxyz file")
    parser.add_argument(
        "-c",
        "--charges",
        **argv,
        required=True,
        help='JSON species mapping, for example {"Ba": 2, "Ti": 4, "O": -2}',
    )
    parser.add_argument(
        "-n",
        "--name",
        **argv,
        default="Qs",
        help="name of the per-atom extxyz array (default: %(default)s)",
    )
    parser.add_argument("-o", "--output", **argv, required=True, help="output extxyz file")
    parser.add_argument(
        "--allow-non-neutral",
        action="store_true",
        help="allow structures whose oxidation numbers do not sum to zero",
    )
    parser.add_argument(
        "--neutrality-tolerance",
        **argv,
        type=float,
        default=1e-8,
        help="absolute neutrality tolerance (default: %(default)s)",
    )
    return parser


def read_oxidation_numbers(filename) -> Dict[str, float]:
    """Read and validate a species-to-oxidation-number JSON object."""
    with Path(filename).open("r", encoding="utf-8") as stream:
        mapping = json.load(stream)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("The oxidation-number JSON must be a non-empty object.")

    oxidation_numbers = {}
    for symbol, value in mapping.items():
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("Every oxidation-number key must be a chemical-symbol string.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Oxidation number for '{symbol}' must be numeric.")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Oxidation number for '{symbol}' must be finite.")
        if not value.is_integer():
            warnings.warn(
                f"Oxidation number for '{symbol}' is not an integer: {value}.",
                UserWarning,
                stacklevel=2,
            )
        oxidation_numbers[symbol] = value
    return oxidation_numbers


def add_oxidation_numbers(
    structures: Iterable[Atoms],
    oxidation_numbers: Dict[str, float],
    name: str = "Qs",
    require_neutral: bool = True,
    tolerance: float = 1e-8,
) -> List[Atoms]:
    """Attach mapped oxidation numbers to copies of all structures."""
    if not name:
        raise ValueError("The oxidation-number array name cannot be empty.")
    if tolerance < 0:
        raise ValueError("The neutrality tolerance cannot be negative.")

    output = []
    for index, atoms in enumerate(structures):
        symbols = atoms.get_chemical_symbols()
        missing = sorted(set(symbols) - set(oxidation_numbers))
        if missing:
            raise ValueError(
                f"Structure {index} contains species absent from the JSON mapping: "
                + ", ".join(missing)
            )
        values = np.asarray([oxidation_numbers[symbol] for symbol in symbols], dtype=float)
        total = float(values.sum())
        if require_neutral and not np.isclose(total, 0.0, atol=tolerance, rtol=0.0):
            raise ValueError(
                f"Structure {index} is not oxidation-number neutral: sum = {total:.12g}. "
                "Use --allow-non-neutral if this is intentional."
            )

        charged = atoms.copy()
        charged.set_array(name, values)
        output.append(charged)
    if not output:
        raise ValueError("The input extxyz contains no structures.")
    return output


@cli(prepare_args, description)
def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.suffix != ".extxyz" or output_path.suffix != ".extxyz":
        raise ValueError("Both input and output must use the .extxyz extension.")

    oxidation_numbers = read_oxidation_numbers(args.charges)
    structures = read(input_path, format="extxyz", index=":")
    charged = add_oxidation_numbers(
        structures,
        oxidation_numbers,
        name=args.name,
        require_neutral=not args.allow_non_neutral,
        tolerance=args.neutrality_tolerance,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, charged, format="extxyz")
    mapping = ", ".join(f"{symbol}={value:g}" for symbol, value in oxidation_numbers.items())
    print(f"Oxidation numbers: {mapping}")
    print(f"Added array '{args.name}' to {len(charged)} structure(s) in '{output_path}'.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
