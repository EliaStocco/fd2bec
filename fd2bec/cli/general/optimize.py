import argparse
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from ase import Atoms
from ase.calculators.socketio import SocketIOCalculator
from ase.constraints import FixSymmetry
from ase.filters import UnitCellFilter
from ase.optimize import BFGS

from fd2bec.cli import cli, read_input_structures, str2bool
from fd2bec.cli.parser import add_shared_argument
from fd2bec.io import write

description = "Run an ASE optimizer with constrained symmetries."
CALCULATOR_RESULT_KEYS = ("energy", "free_energy", "stress", "forces")


def prepare_args(descr):
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    add_shared_argument(parser, "input_structure")
    parser.add_argument(
        "-if",
        "--input_format",
        "--input-format",
        **argv,
        required=False,
        type=str,
        help="input file format (default: %(default)s)",
        default=None,
    )
    socket_group = parser.add_mutually_exclusive_group(required=True)
    socket_group.add_argument(
        "-p", "--port", **argv, type=int, help="TCP port on which to listen; selects TCP mode"
    )
    socket_group.add_argument(
        "-u",
        "--unixsocket",
        **argv,
        type=str,
        help="UNIX-domain socket name/path; selects UNIX mode",
    )
    parser.add_argument(
        "-f",
        "--fmax",
        **argv,
        required=False,
        type=float,
        help="force convergence threshold in eV/A (default: %(default)s)",
        default=0.05,
    )
    parser.add_argument(
        "-r",
        "--restart",
        **argv,
        required=False,
        type=str,
        help="file to restart the optimization from (default: %(default)s)",
        default=None,
    )
    parser.add_argument(
        "-ms",
        "--max-steps",
        "--maxstep",
        dest="max_steps",
        **argv,
        required=False,
        type=int,
        help="maximum number of optimizer steps (default: %(default)s)",
        default=100,
    )

    parser.add_argument(
        "-cs",
        "--constrain-symmetry",
        "--constrain_symmetry",
        dest="constrain_symmetry",
        action="store_true",
        help="preserve the initial symmetry",
    )
    add_shared_argument(parser, "symprec")
    parser.add_argument(
        "-rc",
        "--relax-cell",
        "--relax_cell",
        dest="relax_cell",
        action="store_true",
        help="relax the cell as well as atomic positions",
    )
    parser.add_argument(
        "--print-cell",
        **argv,
        required=False,
        type=str2bool,
        help="print cell and stress after each cell-relaxation step (default: %(default)s)",
        default=False,
    )

    parser.add_argument(
        "-cc",
        "--cell_constraints",
        "--cell-constraints",
        **argv,
        nargs="+",
        required=False,
        type=str,
        help=(
            "fixed deformation-gradient components: first letter is the "
            "lattice direction and second is Cartesian (e.g. az bz cz)"
        ),
        default=None,
    )

    parser.add_argument(
        "-o",
        "--output",
        **argv,
        required=False,
        type=str,
        help="final optimized structure in extxyz format (default: %(default)s)",
        default="final.extxyz",
    )
    parser.add_argument(
        "-t",
        "--history",
        **argv,
        required=False,
        type=str,
        help="full optimization history in extxyz format (default: %(default)s)",
        default="optimization-history.extxyz",
    )

    return parser


class ConstrainedUnitCellFilter(UnitCellFilter):
    """
    UnitCellFilter with selected deformation-gradient components fixed.

    cell_mask:
        3x3 array:
        1 -> relax this component
        0 -> keep this component fixed

        Rows correspond to lattice directions a,b,c
        Columns correspond to x,y,z Cartesian components.

    Example:
        Fix z components of all lattice vectors:

        mask[:,2] = 0
    """

    def __init__(self, atoms: Atoms, cell_mask=None, **kwargs):
        super().__init__(atoms, **kwargs)
        if cell_mask is None:
            cell_mask = np.ones((3, 3), dtype=bool)
        cell_mask = np.array(cell_mask, dtype=bool, copy=True)
        if cell_mask.shape != (3, 3):
            raise ValueError(f"cell_mask must have shape (3,3), got {cell_mask.shape}")
        self.cell_mask = cell_mask
        self.orig_cell = atoms.cell.array.copy()

    def _apply_constraints(self, positions: np.ndarray) -> None:
        """Reset fixed components of the filter's deformation gradient.

        ``UnitCellFilter`` stores ``cell_factor * deformation_gradient`` in
        its final three pseudo-atom positions, not the physical cell vectors.
        A fixed component must therefore be restored to the corresponding
        component of the identity deformation gradient.  The transpose is
        required because :meth:`set_positions` transposes the stored
        deformation gradient before applying ``cell_mask``.
        """
        deformation = positions[-3:]
        undeformed = self.cell_factor * np.eye(3)
        fixed = (~self.cell_mask).T
        deformation[fixed] = undeformed[fixed]

    def get_positions(self) -> np.ndarray:
        positions = super().get_positions()
        self._apply_constraints(positions)
        return positions

    def get_forces(self, **kwargs) -> np.ndarray:
        """Return forces with fixed deformation components projected out."""
        forces = super().get_forces(**kwargs)
        fixed = (~self.cell_mask).T
        forces[-3:][fixed] = 0.0
        return forces

    def set_positions(self, new: np.ndarray, **kwargs):
        """
        new is an array with shape (natoms+3,3).

        the first natoms rows are the positions of the atoms, the last
        three rows are the deformation tensor used to change the cell shape.

        the new cell is first set from original cell transformed by the new
        deformation gradient, then the positions are set with respect to the
        current cell by transforming them with the same deformation gradient
        """

        # Optimizers may reuse their positions array, so do not mutate it.
        new = np.array(new, copy=True)
        self._apply_constraints(new)

        natoms = len(self.atoms)
        new_atom_positions = new[:natoms]
        new_deform_grad = new[natoms:] / self.cell_factor
        deform = (new_deform_grad - np.eye(3)).T * self.mask
        deform[~self.cell_mask] = 0.0
        # Set the new cell from the original cell and the new
        # deformation gradient.  Both current and final structures should
        # preserve symmetry, so if set_cell() calls FixSymmetry.adjust_cell(),
        # it should be OK
        newcell = self.orig_cell @ (np.eye(3) + deform)

        self.atoms.set_cell(newcell, scale_atoms=True)
        # Set the positions from the ones passed in (which are without the
        # deformation gradient applied) and the new deformation gradient.
        # This should also preserve symmetry, so if set_positions() calls
        # FixSymmetyr.adjust_positions(), it should be OK
        self.atoms.set_positions(new_atom_positions @ (np.eye(3) + deform), **kwargs)

        # FixSymmetry and the component mask are separate projections.  Fail
        # clearly if an atomic constraint changes a component fixed here.
        actual_deform = np.linalg.solve(self.orig_cell, self.atoms.cell.array) - np.eye(3)
        if not np.allclose(actual_deform[~self.cell_mask], 0.0, atol=1e-10, rtol=0.0):
            raise RuntimeError(
                "An atomic/cell constraint changed a fixed cell deformation "
                "component; the symmetry and cell constraints are incompatible."
            )


def parse_cell_constraints(
    constraints: Optional[List[str]],
) -> Optional[np.ndarray]:
    if constraints is None:
        return None

    # Start with everything free
    mask = np.ones((3, 3), dtype=bool)

    vectors = {"a": 0, "b": 1, "c": 2}
    components = {"x": 0, "y": 1, "z": 2}

    for item in constraints:
        item = item.lower()
        if len(item) != 2:
            raise ValueError(f"Invalid cell constraint '{item}'. Expected format: ax, by, cz, ...")

        if item[0] not in vectors:
            raise ValueError(f"Unknown lattice vector '{item[0]}' in '{item}'. Allowed: a, b, c.")

        if item[1] not in components:
            raise ValueError(
                f"Unknown Cartesian component '{item[1]}' in '{item}'. Allowed: x, y, z."
            )
        vector = vectors[item[0]]
        component = components[item[1]]

        # False means fixed
        if not mask[vector, component]:
            raise ValueError(f"Cell component '{item}' specified more than once.")
        mask[vector, component] = False

    return mask


def validate_args(args) -> None:
    """Validate combinations and values not expressible with argparse alone."""
    if args.fmax <= 0:
        raise ValueError("--fmax must be greater than zero")
    if args.symprec <= 0:
        raise ValueError("--symprec must be greater than zero")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be greater than zero")
    if args.cell_constraints and not args.relax_cell:
        raise ValueError("--cell-constraints requires --relax-cell true")
    if args.unixsocket is not None and not args.unixsocket.strip():
        raise ValueError("--unixsocket cannot be empty")
    if args.unixsocket is None:
        if not (1025 <= args.port <= 65535):
            raise ValueError("--port must be between 1025 and 65535")
    if Path(args.output).suffix.lower() != ".extxyz":
        raise ValueError("--output must use the .extxyz extension")
    if Path(args.history).suffix.lower() != ".extxyz":
        raise ValueError("--history must use the .extxyz extension")
    if Path(args.output).resolve() == Path(args.history).resolve():
        raise ValueError("--output and --history must refer to different files")


def write_extxyz_snapshot(atoms: Atoms, output: str) -> None:
    """Overwrite ``output`` with one structure in extxyz format."""
    write(output, atoms, format="extxyz")


def remove_stored_calculator_results(atoms: Atoms) -> List[str]:
    """Remove results loaded from the input so new calculator results do not collide."""
    removed = []
    for key in CALCULATOR_RESULT_KEYS:
        if key in atoms.info:
            atoms.info.pop(key)
            removed.append(key)
        if key in atoms.arrays:
            atoms.arrays.pop(key)
            removed.append(key)
    return sorted(set(removed))


class ExtxyzHistoryWriter:
    """Write the first snapshot from scratch, then append subsequent snapshots."""

    def __init__(self, atoms: Atoms, output: str):
        self.atoms = atoms
        self.output = output
        self.append = False

    def __call__(self) -> None:
        write(self.output, self.atoms, format="extxyz", append=self.append)
        self.append = True


@cli(prepare_args, description)
def main(args):
    """Run the ASE BFGS optimization."""

    validate_args(args)

    # ------------------#
    atoms: Atoms = read_input_structures(args.input, input_format=args.input_format)
    print("Number of atoms:", atoms.get_global_number_of_atoms())

    removed_results = remove_stored_calculator_results(atoms)
    if removed_results:
        print("Removed stored calculator results:", ", ".join(removed_results))

    # ------------------#
    if args.constrain_symmetry:
        print("Setting symmetry constraints ... ", end="")
        symmetry_constraint = FixSymmetry(
            atoms,
            symprec=args.symprec,
            verbose=True,
        )
        atoms.set_constraint([*atoms.constraints, symmetry_constraint])
        print("done")

    # ------------------#
    unit_cell_filter = None
    if args.relax_cell:
        print("Preparing cell-relaxation filter ... ", end="")
        cell_mask = parse_cell_constraints(args.cell_constraints)
        if cell_mask is None:
            unit_cell_filter = UnitCellFilter(atoms)
        else:
            print("\nCell constraint mask:")
            print(cell_mask.astype(int))
            unit_cell_filter = ConstrainedUnitCellFilter(
                atoms,
                cell_mask=cell_mask,
            )
        print("done")

    with tempfile.TemporaryDirectory(prefix="fd2bec-optimize-") as temporary_directory:
        temporary_trajectory = Path(temporary_directory) / "optimization.traj"
        print("Allocating BFGS optimizer ... ", end="")
        opt = BFGS(
            atoms if unit_cell_filter is None else unit_cell_filter,
            restart=args.restart,
            trajectory=str(temporary_trajectory),
        )
        print("done")
        print(f"Writing the full optimization history to {args.history}.")
        opt.attach(ExtxyzHistoryWriter(atoms, args.history), interval=1)

        if unit_cell_filter is not None and args.print_cell:

            def print_cell_and_stress():
                print("Cell:")
                print(np.round(atoms.cell.array, 3).tolist())
                stress = atoms.get_stress(voigt=False)
                print("Stress tensor (eV/A^3):")
                print(np.round(stress, 3).tolist())
                print()

            opt.attach(print_cell_and_stress, interval=1)

        if args.unixsocket is not None:
            socket_parameters = {"unixsocket": args.unixsocket}
            print(f"UNIX socket: {args.unixsocket}")
        else:
            socket_parameters = {"port": args.port}
            print(f"TCP socket: listening on port {args.port}")

        print("Running BFGS optimizer ...")
        try:
            with SocketIOCalculator(**socket_parameters) as calc:
                atoms.calc = calc
                opt.run(fmax=args.fmax, steps=args.max_steps)
        finally:
            opt.close()

    print("Finished running BFGS optimizer")
    # Clear constraints right before writing
    atoms.set_constraint()

    print(f"Writing final structure to {args.output} ... ", end="")
    write_extxyz_snapshot(atoms, args.output)
    print("done")


# ---------------------------------------#
if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
