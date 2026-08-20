# Tested by pytest: tests/test_prepare_aims_control.py, tests/test_aims_workflow_wrappers.py

import argparse
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from numpy.linalg import norm

from fd2bec.cli import cli
from fd2bec.cli.aims.get_basis_functions_fhi_aims import create_species_file
from fd2bec.io import read
from fd2bec.io import read as fd2bec_read

description = "Prepare calculations for FHI-aims."


CONTROL_FILE = Path("control.in")


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
    parser.add_argument(
        "--k-density",
        **argv,
        type=float,
        default=8.0,
        help=(
            "reciprocal-space k-grid density, ignored when --k-grid is given (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--k-grid",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
        help="explicit SCF k-grid; overrides --k-density",
    )
    parser.add_argument(
        "--k_density_polarization",
        "--k-density-polarization",
        dest="k_density_polarization",
        **argv,
        type=float,
        default=10.0,
        help=(
            "absolute reciprocal-space k-grid density along each polarization "
            "direction, ignored when --k-grid-polarization is given "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--k_grid_polarization",
        "--k-grid-polarization",
        "--polarization-k-grid",
        dest="k_grid_polarization",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
        help="explicit polarization k-grid; overrides --k-density-polarization",
    )
    parser.add_argument(
        "--basis",
        **argv,
        default="light",
        help="FHI-aims basis set used when control.in has no species blocks (default: %(default)s)",
    )
    parser.add_argument(
        "--aims-folder",
        "--folder",
        dest="aims_folder",
        **argv,
        help="FHI-aims folder used to find species defaults",
    )
    parser.add_argument(
        "--aims-variable",
        "--variable",
        dest="aims_variable",
        **argv,
        default="AIMS_PATH",
        help="environment variable containing the FHI-aims folder (default: %(default)s)",
    )
    parser.add_argument(
        "--use-csc",
        "--csc",
        dest="use_csc",
        action="store_true",
        help="reuse the first converged density through ELSI CSC restart files",
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


def suggest_kgrid(input_file: str, k_density: float = 5.0):
    if k_density <= 0:
        raise ValueError("k-grid density must be positive.")
    atoms = read(input_file)
    cell = atoms.get_cell()

    reciprocal = 2 * np.pi * np.linalg.inv(cell.T)

    kgrid = []
    for b in reciprocal:
        Ni = int(np.ceil(k_density * norm(b)))  # <-- FIXED
        kgrid.append(max(1, Ni))

    return tuple(kgrid)


def polarization_kgrids(
    k_grid: Tuple[int, int, int], polarization_grid: Tuple[int, int, int]
) -> Tuple[Tuple[int, ...], ...]:
    """Return Berry-phase meshes from an absolute polarization mesh.

    At least one extra k-point is always added in the active direction, so
    every polarization mesh has more points than the SCF mesh.
    """
    kx, ky, kz = k_grid
    polarization_kx, polarization_ky, polarization_kz = polarization_grid
    return (
        (1, max(kx + 1, polarization_kx), ky, kz),
        (2, kx, max(ky + 1, polarization_ky), kz),
        (3, kx, ky, max(kz + 1, polarization_kz)),
    )


def _keyword(line: str, words: int = 1) -> Tuple[str, ...]:
    """Return the active keyword tokens, ignoring blank and comment lines."""
    stripped = line.lstrip()
    if not stripped or stripped.startswith("#"):
        return ()
    return tuple(stripped.split()[:words])


def _has_species_section(control_file: Path) -> bool:
    """Whether control.in already contains an active FHI-aims species block."""
    return any(
        _keyword(line)[0:1] == ("species",)
        for line in control_file.read_text(encoding="utf-8").splitlines()
    )


def ensure_basis_functions(
    control_file: Path,
    input_file: str,
    basis: str = "light",
    aims_folder: str = None,
    variable: str = "AIMS_PATH",
) -> Optional[Path]:
    """Append generated species blocks when control.in has none.

    Return the generated species file, or ``None`` when control.in already
    contains species blocks.
    """
    if _has_species_section(control_file):
        return None

    species_file = control_file.with_name(f"species.{basis}.in")
    create_species_file(
        input_file=input_file,
        basis=basis,
        aims_folder=aims_folder,
        variable=variable,
        output=str(species_file),
    )
    control_text = control_file.read_text(encoding="utf-8")
    species_text = species_file.read_text(encoding="utf-8")
    separator = "" if not control_text or control_text.endswith("\n") else "\n"
    control_file.write_text(
        control_text + separator + species_text,
        encoding="utf-8",
    )
    return species_file


def _parse_integer_values(line: str, keyword: str, count: int) -> Tuple[int, ...]:
    values = line.split()
    if len(values) < count + 1:
        raise ValueError(f"Malformed '{keyword}' line in control.in: {line.rstrip()}")
    try:
        return tuple(int(value) for value in values[1 : count + 1])
    except ValueError as error:
        raise ValueError(f"Malformed '{keyword}' line in control.in: {line.rstrip()}") from error


def _parse_polarization(line: str) -> Tuple[int, ...]:
    values = line.split()
    if len(values) < 6:
        raise ValueError(f"Malformed 'output polarization' line in control.in: {line.rstrip()}")
    try:
        return tuple(int(value) for value in values[2:6])
    except ValueError as error:
        raise ValueError(
            f"Malformed 'output polarization' line in control.in: {line.rstrip()}"
        ) from error


def _parse_density(line: str) -> float:
    values = line.split()
    if len(values) < 2:
        raise ValueError(f"Malformed 'k_grid_density' line in control.in: {line.rstrip()}")
    try:
        return float(values[1])
    except ValueError as error:
        raise ValueError(
            f"Malformed 'k_grid_density' line in control.in: {line.rstrip()}"
        ) from error


def _replace_control_block(
    lines: Iterable[str], replacements: List[str], keywords: Tuple[str, ...]
) -> List[str]:
    """Remove active control keywords and insert replacements at their first position."""
    kept = []
    first_removed = None
    for line in lines:
        tokens = _keyword(line, words=2)
        if tokens and (tokens[0] in keywords or " ".join(tokens) in keywords):
            if first_removed is None:
                first_removed = len(kept)
            continue
        kept.append(line)

    if first_removed is None:
        first_removed = next(
            (
                index
                for index, line in enumerate(kept)
                if line.strip() and not line.lstrip().startswith("#")
            ),
            0,
        )
    kept[first_removed:first_removed] = [f"{line}\n" for line in replacements]
    return kept


def update_control_file(
    control_file: Path,
    input_file: str,
    k_density: float,
    k_density_polarization: float = 10.0,
    k_grid: Optional[Tuple[int, int, int]] = None,
    k_grid_polarization: Optional[Tuple[int, int, int]] = None,
) -> Tuple[Tuple[int, int, int], bool]:
    """Normalize k-grid and polarization settings in ``control.in``.

    Existing ``k_grid``, ``k_grid_density``, and ``output polarization`` lines
    are compared to the requested settings and replaced as one consistent
    block. This avoids duplicate settings and handles controls that specify
    only ``k_grid_density``. Explicit SCF and polarization grids take
    precedence over the grids derived from their respective densities.
    """
    if k_grid is None:
        k_grid = tuple(suggest_kgrid(input_file, k_density))
    else:
        k_grid = tuple(k_grid)
        if len(k_grid) != 3 or any(value <= 0 for value in k_grid):
            raise ValueError("Every explicit k-grid dimension must be positive.")
    lines = control_file.read_text(encoding="utf-8").splitlines(keepends=True)
    existing_k_grids = []
    existing_densities = []
    existing_polarizations = []
    for line in lines:
        tokens = _keyword(line, words=2)
        if tokens and tokens[0] == "k_grid":
            existing_k_grids.append(_parse_integer_values(line, "k_grid", 3))
        elif tokens and tokens[0] == "k_grid_density":
            density = _parse_density(line)
            existing_densities.append(density)
        elif tokens == ("output", "polarization"):
            existing_polarizations.append(_parse_polarization(line))

    if k_grid_polarization is None:
        polarization_grid = tuple(suggest_kgrid(input_file, k_density_polarization))
    else:
        polarization_grid = tuple(k_grid_polarization)
        if len(polarization_grid) != 3 or any(value <= 0 for value in polarization_grid):
            raise ValueError("Every explicit polarization k-grid dimension must be positive.")
        if any(
            polarization_value <= scf_value
            for polarization_value, scf_value in zip(polarization_grid, k_grid)
        ):
            raise ValueError(
                "Every explicit polarization k-grid dimension must exceed "
                "the corresponding SCF k-grid dimension."
            )
    expected_polarizations = polarization_kgrids(k_grid, polarization_grid)
    settings_match = (
        all(old_grid == k_grid for old_grid in existing_k_grids)
        and all(np.isclose(density, k_density) for density in existing_densities)
        and existing_polarizations == list(expected_polarizations)
    )

    replacements = ["k_grid " + " ".join(str(value) for value in k_grid)]
    replacements.extend(
        "output polarization " + " ".join(str(value) for value in values)
        for values in expected_polarizations
    )
    updated = _replace_control_block(
        lines,
        replacements,
        ("k_grid", "k_grid_density", "output polarization"),
    )
    control_file.write_text("".join(updated), encoding="utf-8")
    return k_grid, settings_match


def _replace_keyword(lines: Iterable[str], keyword: str, replacement: str) -> List[str]:
    """Replace all active instances of a keyword with one canonical line."""
    return _replace_control_block(lines, [replacement], (keyword,))


def write_control_templates(control_file: Path) -> Tuple[Path, Path, Path]:
    """Create the three control files consumed by ``aims_template.sh``.

    ``control.general.in`` is the regular calculation.  The other two files
    enable the CSC restart workflow for the first and subsequent geometries.
    """
    lines = control_file.read_text(encoding="utf-8").splitlines(keepends=True)
    general_control = control_file.with_name("control.general.in")
    first_control = control_file.with_name("control.first.in")
    other_control = control_file.with_name("control.other.in")

    general_lines = _replace_control_block(lines, [], ("elsi_restart",))
    first_lines = _replace_keyword(lines, "elsi_restart", "elsi_restart write scf_converged")
    other_lines = _replace_keyword(lines, "elsi_restart", "elsi_restart read")

    general_control.write_text("".join(general_lines), encoding="utf-8")
    first_control.write_text("".join(first_lines), encoding="utf-8")
    other_control.write_text("".join(other_lines), encoding="utf-8")
    return general_control, first_control, other_control


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
    if args.k_grid is None and args.k_density <= 0:
        raise ValueError("--k-density must be positive.")
    if args.k_grid is not None and any(value <= 0 for value in args.k_grid):
        raise ValueError("Every --k-grid dimension must be positive.")
    if args.k_grid_polarization is not None and any(
        value <= 0 for value in args.k_grid_polarization
    ):
        raise ValueError("Every --k-grid-polarization dimension must be positive.")
    if args.k_grid_polarization is None and args.k_density_polarization <= 0:
        raise ValueError("--k_density_polarization must be positive.")
    if not CONTROL_FILE.is_file():
        raise FileNotFoundError("prepare_aims requires a control.in in the current directory.")

    species_file = ensure_basis_functions(
        CONTROL_FILE,
        args.input,
        basis=args.basis,
        aims_folder=args.aims_folder,
        variable=args.aims_variable,
    )
    if species_file is None:
        print("Found basis functions in control.in; leaving them unchanged.")
    else:
        print(f"Added basis functions from {species_file} to control.in.")

    k_grid, settings_match = update_control_file(
        CONTROL_FILE,
        args.input,
        args.k_density,
        args.k_density_polarization,
        k_grid=args.k_grid,
        k_grid_polarization=args.k_grid_polarization,
    )
    print(f"Updated control.in k-grid: {' '.join(str(value) for value in k_grid)}")
    print("Updated control.in Berry-phase polarization settings.")
    if settings_match:
        print("Existing control.in settings already matched the requested values.")
    control_templates = write_control_templates(CONTROL_FILE)
    print("Generated control templates: " + ", ".join(path.name for path in control_templates))

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
    requested_csc = args.use_csc
    if len(structures) == 1:
        args.use_csc = False
        if requested_csc:
            print("CSC restart disabled because only one geometry was generated.")
    use_csc = args.use_csc

    # --- step 2: copy + modify template ---
    with (
        resources.files("fd2bec.cli.aims")
        .joinpath("aims_template.sh")
        .open("r", encoding="utf-8") as src
    ):
        content = src.read()

    # Set the default while allowing a submission script to override it.
    content = content.replace("USE_CSC_DEFAULT", "true" if use_csc else "false")

    # write output
    dst = Path(".") / "sourceme.sh"
    dst.write_text(content, encoding="utf-8")

    print("Recommended k-grid")
    kx, ky, kz = k_grid
    print(f"k_grid {kx} {ky} {kz}")

    print("\nRecommended keywords for computing the polarization")
    if args.k_grid_polarization is None:
        polarization_grid = tuple(suggest_kgrid(args.input, args.k_density_polarization))
    else:
        polarization_grid = tuple(args.k_grid_polarization)
    for polarization in polarization_kgrids(k_grid, polarization_grid):
        print("output polarization " + " ".join(str(value) for value in polarization))

    print("\nPlease add the following lines in your submission script:")
    print("export AIMS=/path/to/your/aims/executable")
    print("source sourceme.sh")

    print(
        "\nThe provided control.in was updated with the selected k-grid and polarization settings."
    )
    print(f"Prepared {len(structures)} {args.what} geometries in '{args.output}'.")
    print(f"Subcommand details were written to '{log_file}'.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
