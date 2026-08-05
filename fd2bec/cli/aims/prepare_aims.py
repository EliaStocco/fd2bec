import argparse
import subprocess
import sys
from importlib import resources
from pathlib import Path

import numpy as np
from ase.io import read
from numpy.linalg import norm

from fd2bec.cli import cli
from fd2bec.io import read as fd2bec_read

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
        "-w",
        "--what",
        **argv,
        choices=("bec", "piezo"),
        default="bec",
        help="quantity for which displacements are generated (default: %(default)s)",
    )
    parser.add_argument(
        "-a",
        "--amplitude",
        **argv,
        type=float,
        default=1e-3,
        help="Cartesian displacement amplitude in Angstrom (default: %(default)s)",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--no-symmetry",
        action="store_true",
        help="generate every signed Cartesian basis displacement",
    )
    selection.add_argument(
        "-n",
        "--number",
        **argv,
        type=int,
        help="generate this many normally distributed random displacements",
    )
    parser.add_argument("--seed", **argv, type=int, help="random seed used with --number")
    parser.add_argument(
        "--structures-output",
        **argv,
        default="displaced-structures.extxyz",
        help="multi-frame displaced structure file (default: %(default)s)",
    )
    parser.add_argument(
        "--displacements-output",
        **argv,
        default="displacements.txt",
        help="flattened displacement table (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        default="geometries",
        help="folder for FHI-aims geometry files (default: %(default)s)",
    )
    parser.add_argument(
        "--log",
        **argv,
        default="fd2bec-log.txt",
        help="subcommand log file (default: %(default)s)",
    )
    return parser


def preparation_commands(args):
    """Build the displacement-generation and geometry-export commands."""
    generate = [
        sys.executable,
        "-m",
        "fd2bec.cli.displacements.generate_displacements",
        "-i",
        str(args.input),
        "-w",
        str(args.what),
        "-a",
        str(args.amplitude),
        "-d",
        str(args.displacements_output),
        "-o",
        str(args.structures_output),
    ]
    if args.no_symmetry:
        generate.append("--no-symmetry")
    elif args.number is not None:
        generate.extend(("--number", str(args.number)))
        if args.seed is not None:
            generate.extend(("--seed", str(args.seed)))

    export = [
        sys.executable,
        "-m",
        "fd2bec.cli.displacements.extxyz2folder",
        "-i",
        str(args.structures_output),
        "-f",
        "aims",
        "-o",
        str(args.output),
    ]
    return generate, export


@cli(prepare_args, description)
def main(args):
    """Prepare displaced FHI-aims geometries and the batch helper script."""
    input_path = Path(args.input)
    if input_path.name == "geometry.in":
        raise ValueError(
            "Rename the input structure: the AIMS workflow uses 'geometry.in' as a work file."
        )
    if args.seed is not None and args.number is None:
        raise ValueError("--seed can only be used together with --number.")

    commands = preparation_commands(args)
    log_file = Path(args.log)
    for filename in (
        Path(args.structures_output),
        Path(args.displacements_output),
        log_file,
    ):
        filename.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as stream:
        for command in commands:
            subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                text=True,
            )

    structures = fd2bec_read(args.structures_output, index=":")
    if not structures:
        raise ValueError("Displacement generation produced no structures.")
    last_index = len(structures) - 1

    # --- step 2: copy + modify template ---
    with (
        resources.files("fd2bec.cli.aims")
        .joinpath("aims_template.sh")
        .open("r", encoding="utf-8") as src
    ):
        content = src.read()

    # replace placeholder
    content = content.replace("NNN", str(last_index))

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

    print("\nPlease add the following lines in your submission script:")
    print("export AIMS=/path/to/your/aims/executable")
    print("source sourceme.sh")

    print("\nYou need to provide a control.in in this folder")
    print("and remember to add 'output polarization' as suggested above.")
    print(f"Prepared {len(structures)} {args.what} geometries in '{args.output}'.")
    print(f"Subcommand details were written to '{log_file}'.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
