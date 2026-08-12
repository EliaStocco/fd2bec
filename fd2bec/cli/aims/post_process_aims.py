# Tested by pytest: tests/test_aims_workflow_wrappers.py

import argparse
import subprocess
import sys
from pathlib import Path

from fd2bec.cli import cli

description = "Post process calculations from FHI-aims."

# This wrapper is intentionally limited to Born effective charges (dP/dR).
# Piezoelectric AIMS calculations must instead use the dPdS post-processing
# workflow: build_dataset4dPdS_aims followed by dPdS2piezo.


def prepare_args(descr):

    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="input structure",
    )
    parser.add_argument(
        "--results",
        **argv,
        default="results",
        help="folder containing aims.n=<index>.out files (default: %(default)s)",
    )
    parser.add_argument(
        "--format",
        **argv,
        default="aims_polarization",
        help="build_dataset4dPdR format key or JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset",
        **argv,
        default="dataset.extxyz",
        help="assembled polarized dataset (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        default=".",
        help="folder for BEC and charge files (default: %(default)s)",
    )
    parser.add_argument(
        "--log",
        **argv,
        default="fd2bec-log.pp.txt",
        help="subcommand log file (default: %(default)s)",
    )
    return parser


def postprocess_commands(args):
    """Build the dataset, BEC fitting, and charge conversion commands."""
    dataset = Path(args.dataset)
    output = Path(args.output)
    bec = output / "bec.txt"
    charges = output / "charges.txt"
    return (
        [
            sys.executable,
            "-m",
            "fd2bec.cli.dPdR.build_dataset4dPdR",
            "-i",
            str(args.results),
            "-r",
            str(args.input),
            "-f",
            str(args.format),
            "-o",
            str(dataset),
        ],
        [
            sys.executable,
            "-m",
            "fd2bec.cli.dPdR.dPdR2bec",
            "-i",
            str(dataset),
            "-o",
            str(output),
        ],
        [
            sys.executable,
            "-m",
            "fd2bec.cli.general.bec2charges",
            "-i",
            str(bec),
            "-o",
            str(charges),
        ],
    )


@cli(prepare_args, description)
def main(args):
    """Build an AIMS displacement dataset and evaluate its Born charges."""
    results = Path(args.results)
    if not results.is_dir():
        raise FileNotFoundError(f"AIMS results folder not found: '{results}'.")
    if not any(path.is_file() for path in results.iterdir()):
        raise ValueError(f"AIMS results folder is empty: '{results}'.")

    dataset = Path(args.dataset)
    dataset.parent.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    log_file = Path(args.log)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as stream:
        for command in postprocess_commands(args):
            subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                text=True,
            )

    print(f"Dataset: '{dataset}'")
    print(f"Born Effective Charges: '{output / 'bec.txt'}'")
    print(f"Scalar charges: '{output / 'charges.txt'}'")
    print(f"Subcommand details: '{log_file}'")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
