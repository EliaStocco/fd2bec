import argparse
import subprocess
from importlib import resources
from pathlib import Path

import numpy as np
from ase.io import read
from numpy.linalg import norm

from fd2bec.cli import cli

description = "Prepare calculations for FHI-aims."


def suggest_kgrid(input_file: str, k_density: float = 5.0):
    atoms = read(input_file)
    cell = atoms.get_cell()

    reciprocal = 2 * np.pi * np.linalg.inv(cell.T)

    kgrid = []
    for b in reciprocal:
        Ni = int(np.ceil(k_density * norm(b)))  # <-- FIXED
        kgrid.append(max(1, Ni))

    return tuple(kgrid)


def run_script(script_name: str, input_file: str):
    """
    Example wrapper for external scripts.
    """
    cmd = [script_name, input_file]
    subprocess.run(cmd, check=True)


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

    assert args.input != "geometry.in", (
        "Please change the filename of your input structure because this file will be overwritten later on."
    )

    log_file = Path("fd2bec-log.txt")
    with log_file.open("w") as f:
        cmd = ["generate_symmetry_inequivalent_displacements", "-i", args.input]
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        cmd = ["apply_displacements", "-i", args.input, "-d", "displacements.txt"]
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)

    folder = Path("geometries")
    file_count = sum(1 for f in folder.iterdir() if f.is_file()) - 1

    # --- step 2: copy + modify template ---
    with (
        resources.files("fd2bec.cli.aims")
        .joinpath("aims_template.sh")
        .open("r", encoding="utf-8") as src
    ):
        content = src.read()

    # replace placeholder
    content = content.replace("NNN", str(file_count))

    # write output
    dst = Path(".") / "sourceme.sh"
    dst.write_text(content, encoding="utf-8")

    print("Recommended k-grid")
    kx, ky, kz = suggest_kgrid(args.input)
    print(f"k_grid {kx} {ky} {kz}")

    print("\nRecommended keywords for computing the polarization")
    print(f"output polarization 1 {10 * kx} {ky} {kz}")
    print(f"output polarization 2 {kx} {10 * ky} {kz}")
    print(f"output polarization 3 {kx} {ky} {10 * kz}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
