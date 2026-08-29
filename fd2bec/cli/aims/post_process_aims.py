# Tested by pytest: tests/test_aims_workflow_wrappers.py

import argparse
import subprocess
import sys
from pathlib import Path

from fd2bec.cli import cli
from fd2bec.cli.parser import add_shared_argument

description = "Post process Born-charge or piezoelectric calculations from FHI-aims."


def prepare_args(descr):

    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    add_shared_argument(parser, "input_structure")
    add_shared_argument(parser, "response_quantity")
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
        help=(
            "build_dataset4dPdR format key or JSON file used for BEC "
            "calculations (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--pattern",
        **argv,
        default="aims.n=*.out",
        help=("output filename glob used for piezoelectric calculations (default: %(default)s)"),
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
        help="folder for fitted tensor files (default: %(default)s)",
    )
    parser.add_argument(
        "--log",
        **argv,
        default="fd2bec-log.pp.txt",
        help="subcommand log file (default: %(default)s)",
    )
    add_shared_argument(parser, "symprec")
    return parser


def postprocess_commands(args):
    """Build the mode-specific dataset and tensor fitting commands."""
    dataset = Path(args.dataset)
    output = Path(args.output)

    if getattr(args, "what", "bec") == "piezo":
        return (
            [
                sys.executable,
                "-m",
                "fd2bec.cli.dPdS.build_dataset4dPdS_aims",
                "-i",
                str(args.results),
                "--pattern",
                str(getattr(args, "pattern", "aims.n=*.out")),
                "-o",
                str(dataset),
            ],
            [
                sys.executable,
                "-m",
                "fd2bec.cli.dPdS.dPdS2piezo",
                "-i",
                str(dataset),
                "-r",
                str(args.input),
                "-sp",
                str(args.symprec),
                "-o",
                str(output),
            ],
        )

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
            "-sp",
            str(args.symprec),
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
    """Build an AIMS finite-difference dataset and evaluate its response tensor."""
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
    if args.what == "piezo":
        print(f"Piezoelectric tensor: '{output / 'piezoelectric.extxyz'}'")
        print(f"Improper piezoelectric tensor: '{output / 'improper-piezoelectric.txt'}'")
        print(f"Proper piezoelectric tensor: '{output / 'proper-piezoelectric.txt'}'")
        print(
            "Direct-fit proper piezoelectric tensor: "
            f"'{output / 'proper-piezoelectric-direct.txt'}'"
        )
    else:
        print(f"Born Effective Charges: '{output / 'bec.extxyz'}'")
        print(f"Born Effective Charges: '{output / 'bec.txt'}'")
        print(f"Scalar charges: '{output / 'charges.txt'}'")
    print(f"Subcommand details: '{log_file}'")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
