"""Prepare Quantum ESPRESSO SCF and Berry-phase NSCF calculations."""

# Tested by pytest: tests/test_prepare_qe.py

import argparse
import re
import subprocess
import sys
from importlib import resources
from pathlib import Path

from fd2bec.cli import cli
from fd2bec.cli.parser import add_shared_argument
from fd2bec.io import read

description = "Prepare Quantum ESPRESSO polarization calculations."
GEOMETRY_MARKER = "! FD2BEC"


def prepare_args(descr):
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    add_shared_argument(parser, "input_structure")
    parser.add_argument(
        "-t",
        "--template",
        **argv,
        required=True,
        help="SCF input template containing the '! FD2BEC' marker",
    )
    add_shared_argument(parser, "response_quantity")
    add_shared_argument(parser, "cartesian_amplitude")
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
        "-o",
        "--output",
        **argv,
        default="qe-calculations",
        help="preparation folder (default: %(default)s)",
    )
    parser.add_argument(
        "--structures-output",
        **argv,
        help="generated multi-frame extxyz (default: OUTPUT/displaced-structures.extxyz)",
    )
    parser.add_argument(
        "--displacements-output",
        **argv,
        help="displacement table (default: OUTPUT/displacements.txt)",
    )
    parser.add_argument(
        "--log",
        **argv,
        help="generator/export log (default: OUTPUT/fd2bec-log.txt)",
    )
    parser.add_argument(
        "--nppstr-factor",
        **argv,
        type=int,
        default=10,
        help="nppstr divided by the k-grid along gdir (default: %(default)s)",
    )
    parser.add_argument(
        "--script",
        **argv,
        help="sourceable run script (default: OUTPUT/sourceme.sh)",
    )
    add_shared_argument(parser, "symprec")
    return parser


def preparation_commands(args, structures_output, displacements_output, geometries_output):
    """Build commands for displacement generation and QE geometry export."""
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
        "-sp",
        str(args.symprec),
        "-d",
        str(displacements_output),
        "-o",
        str(structures_output),
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
        str(structures_output),
        "-f",
        "espresso-in",
        "-o",
        str(geometries_output),
    ]
    return generate, export


def automatic_k_grid(scf_input: str):
    """Extract the three mesh sizes from a ``K_POINTS automatic`` card."""
    match = re.search(
        r"(?im)^\s*K_POINTS\s*(?:\(?\s*automatic\s*\)?|\{\s*automatic\s*\})\s*"
        r"(?:!.*)?\n\s*(\d+)\s+(\d+)\s+(\d+)",
        scf_input,
    )
    if match is None:
        raise ValueError("The SCF template must contain a 'K_POINTS automatic' card.")
    k_grid = tuple(int(value) for value in match.groups())
    if any(value < 1 for value in k_grid):
        raise ValueError("Every automatic k-grid dimension must be positive.")
    return k_grid


def nscf_template(scf_input: str, gdir: int, nppstr: int) -> str:
    """Create one Berry-phase NSCF template from an SCF input template."""
    if gdir not in (1, 2, 3):
        raise ValueError("gdir must be 1, 2, or 3.")
    if scf_input.count(GEOMETRY_MARKER) != 1:
        raise ValueError(f"The SCF template must contain exactly one '{GEOMETRY_MARKER}' marker.")

    calculation = re.compile(r"(?im)^(\s*calculation\s*=\s*)(['\"]?)scf\2(\s*,?\s*(?:!.*)?)$")
    content, replacements = calculation.subn(r"\1'nscf'\3", scf_input, count=1)
    if replacements != 1:
        raise ValueError("Could not find calculation = 'scf' in the &CONTROL namelist.")

    berry = f"  lberry = .true.\n  gdir = {gdir}\n  nppstr = {nppstr}\n  {GEOMETRY_MARKER}"
    return content.replace(GEOMETRY_MARKER, berry)


def prepare_qe_files(input_file, scf_template, output, nppstr_factor=10):
    """Write QE templates for a prepared extxyz; return its size and k-grid."""
    if nppstr_factor < 1:
        raise ValueError("--nppstr-factor must be positive.")

    template_path = Path(scf_template)
    content = template_path.read_text(encoding="utf-8")
    if content.count(GEOMETRY_MARKER) != 1:
        raise ValueError(f"The SCF template must contain exactly one '{GEOMETRY_MARKER}' marker.")
    k_grid = automatic_k_grid(content)

    structures = read(input_file, index=":")
    if not structures:
        raise ValueError("The extxyz input contains no structures.")

    output = Path(output)
    templates = output / "templates"
    templates.mkdir(parents=True, exist_ok=True)
    (templates / "scf.in").write_text(content, encoding="utf-8")
    for gdir, k_points in enumerate(k_grid, start=1):
        nppstr = nppstr_factor * k_points
        (templates / f"nscf.g={gdir}.in").write_text(
            nscf_template(content, gdir, nppstr), encoding="utf-8"
        )
    return len(structures), k_grid


def write_run_script(script_path: Path) -> None:
    """Copy the generic QE helper to its destination."""
    script_path = script_path.absolute()

    with (
        resources.files("fd2bec.cli.qe")
        .joinpath("qe_template.sh")
        .open("r", encoding="utf-8") as stream
    ):
        script = stream.read()
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script, encoding="utf-8")


@cli(prepare_args, description)
def main(args):
    """Generate displacements and all files used by the QE run helper."""
    if args.seed is not None and args.number is None:
        raise ValueError("--seed can only be used together with --number.")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    structures_output = Path(args.structures_output or output / "displaced-structures.extxyz")
    displacements_output = Path(args.displacements_output or output / "displacements.txt")
    geometries_output = output / "geometries"
    log_file = Path(args.log or output / "fd2bec-log.txt")
    for path in (structures_output, displacements_output, log_file):
        path.parent.mkdir(parents=True, exist_ok=True)

    commands = preparation_commands(
        args, structures_output, displacements_output, geometries_output
    )
    with log_file.open("w", encoding="utf-8") as stream:
        for command in commands:
            subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                text=True,
            )

    number, k_grid = prepare_qe_files(structures_output, args.template, output, args.nppstr_factor)

    script_path = Path(args.script) if args.script else output / "sourceme.sh"
    write_run_script(script_path)

    print(f"Prepared {number} {args.what} structure(s) in '{output}'.")
    print(f"Automatic k-grid: {k_grid[0]} {k_grid[1]} {k_grid[2]}")
    print(
        "nppstr: "
        + " ".join(str(args.nppstr_factor * value) for value in k_grid)
        + " for gdir = 1, 2, 3"
    )
    print("In your submission script, define QE and source the helper:")
    print("export QE=/path/to/pw.x")
    print(f"source {script_path}")
    print(f"Displacement details were written to '{log_file}'.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
