import argparse
import subprocess
from pathlib import Path

from fd2bec.cli import cli

description = "Post process calculations from FHI-aims."


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
    return parser


@cli(prepare_args, description)
def main(args):

    log_file = Path("fd2bec-log.pp.txt")
    with log_file.open("w") as f:
        cmd = [
            "build_dataset4dPdR",
            "-i",
            "results",
            "-r",
            args.input,
            "-f",
            "aims_polarization",
            "-o",
            "dataset.extxyz",
        ]
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        cmd = ["dPdR2bec", "-i", "dataset.extxyz"]
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        cmd = ["bec2charges", "-i", "bec.txt", "-o", "charges.txt"]
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)

    print("You can find the Born Effective Charges in 'bec.txt'")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
