"""Shared argparse argument definitions for fd2bec command-line interfaces."""

import argparse

from fd2bec import SYMPREC


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not parsed > 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


SHARED_ARGUMENTS = {
    "cartesian_amplitude": (
        ("-a", "--amplitude"),
        {
            "metavar": "\b",
            "type": _positive_float,
            "default": 1e-3,
            "help": (
                "Cartesian atomic or cell displacement amplitude in Angstrom (default: %(default)s)"
            ),
        },
    ),
    "data_destination": (
        ("-w", "--what"),
        {
            "metavar": "\b",
            "required": True,
            "choices": ("i", "info", "a", "array", "arrays"),
            "help": "store data as structure info ('i') or a per-atom array ('a')",
        },
    ),
    "data_file": (
        ("-d", "--data"),
        {
            "metavar": "\b",
            "required": True,
            "help": "path to the input numeric data file",
        },
    ),
    "data_name": (
        ("-n", "--name"),
        {
            "metavar": "\b",
            "required": True,
            "help": "name of the structure info or per-atom array",
        },
    ),
    "input_structure": (
        ("-i", "--input"),
        {
            "metavar": "\b",
            "required": True,
            "help": "path to the input atomic structure",
        },
    ),
    "displacement_target": (
        ("-w", "--what"),
        {
            "metavar": "\b",
            "choices": (
                "bec",
                "piezo",
                "forces",
                "stress",
                "elastic",
                "force_constants",
            ),
            "default": "bec",
            "help": "target quantity (default: %(default)s)",
        },
    ),
    "output_structure": (
        ("-o", "--output"),
        {
            "metavar": "\b",
            "required": True,
            "help": "path to the output atomic structure",
        },
    ),
    "response_quantity": (
        ("-w", "--what"),
        {
            "metavar": "\b",
            "choices": ("bec", "piezo"),
            "default": "bec",
            "help": "response quantity (default: %(default)s)",
        },
    ),
    "structure_index": (
        ("--index",),
        {
            "metavar": "\b",
            "type": int,
            "default": 0,
            "help": "index of the input structure (default: %(default)s)",
        },
    ),
    "strain_amplitude": (
        ("-a", "--amplitude"),
        {
            "metavar": "\b",
            "type": _positive_float,
            "default": 1e-3,
            "help": "dimensionless strain amplitude (default: %(default)s)",
        },
    ),
    "symprec": (
        ("-sp", "--symprec"),
        {
            "metavar": "\b",
            "type": _positive_float,
            "default": SYMPREC,
            "help": "symmetry tolerance in Angstrom (default: %(default)s)",
        },
    ),
}


def add_shared_argument(parser: argparse.ArgumentParser, name: str) -> argparse.Action:
    """Add a named, centrally defined argument to ``parser``."""
    try:
        flags, settings = SHARED_ARGUMENTS[name]
    except KeyError as error:
        choices = ", ".join(sorted(SHARED_ARGUMENTS))
        raise ValueError(f"Unknown shared argument {name!r}; choose from: {choices}") from error
    return parser.add_argument(*flags, **settings)
