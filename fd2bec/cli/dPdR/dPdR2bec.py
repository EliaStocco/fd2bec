# Tested by pytest: tests/test_aims_workflow_wrappers.py

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from ase import Atoms

from fd2bec import ATOL, float_format
from fd2bec.acoustic_sum_rule import write_acoustic_sum_rule_diagnostics
from fd2bec.atomic import AtomicStructure
from fd2bec.cli import KEYWORDS, cli, read_input_structures, str2bool
from fd2bec.cli.parser import add_shared_argument
from fd2bec.cli.tools import matrix_norm
from fd2bec.displacement_cache import (
    born_charge_mode_cache_metadata,
    restore_born_charge_mode_cache,
    save_born_charge_mode_cache,
)
from fd2bec.displacements import displacement_directions, displacement_magnitudes
from fd2bec.fit_diagnostics import (
    FrameResidualDiagnostics,
    frame_residual_diagnostics,
    retained_equation_mask,
    write_dipole_fit_residual_report,
)
from fd2bec.io import write_tensor_extxyz
from fd2bec.linear_system import LinearSystem
from fd2bec.show import print_born_charges
from fd2bec.tensor import BornCharges

description = "Compute the Born Effective Charges as derivative of polarization/dipole w.r.t. nuclear displacements."


def apply_acoustic_sum_rule(bec: np.ndarray):
    """Return the atomic Born-charge sum and ASR-corrected tensors.

    The correction is distributed uniformly over atoms, so the corrected
    tensors satisfy ``sum_I Z*_I = 0`` without changing their response-space
    shape.
    """
    bec = np.asarray(bec, dtype=float)
    if bec.ndim != 3 or bec.shape[1:] != (3, 3):
        raise ValueError(f"BEC data must have shape (N, 3, 3), got {bec.shape}.")
    if len(bec) == 0:
        raise ValueError("At least one atomic Born-charge tensor is required.")
    asr_sum = np.sum(bec, axis=0)
    corrected = bec - asr_sum[None, :, :] / len(bec)
    return asr_sum, corrected


def print_dipole_point_residuals(
    diagnostics: FrameResidualDiagnostics,
    directions: np.ndarray,
    displacement_norms: np.ndarray,
    maximum_atomic_displacements: np.ndarray,
) -> None:
    """Print per-frame residual diagnostics as a consistently aligned table."""
    directions = np.asarray(directions, dtype=object)
    displacement_norms = np.asarray(displacement_norms, dtype=float)
    maximum_atomic_displacements = np.asarray(maximum_atomic_displacements, dtype=float)
    expected_shape = diagnostics.residual_norms.shape
    if (
        directions.shape != expected_shape
        or displacement_norms.shape != expected_shape
        or maximum_atomic_displacements.shape != expected_shape
    ):
        raise ValueError("There must be one direction and displacement magnitude per frame.")

    contributions = diagnostics.squared_residual_contributions
    table = pd.DataFrame(
        {
            "frame": np.arange(len(directions)),
            "kind": ["reference"] + ["displacement"] * (len(directions) - 1),
            "direction": directions,
            "disp. norm": [f"{value:.6e}" for value in displacement_norms],
            "max atom disp.": [f"{value:.6e}" for value in maximum_atomic_displacements],
            "residual norm": [f"{value:.6e}" for value in diagnostics.residual_norms],
            "contribution": [
                "n/a" if not np.isfinite(value) else f"{100.0 * value:.6f}%"
                for value in contributions
            ],
            "flag": ["*" if value else "" for value in diagnostics.outliers],
        }
    )
    print("\nDipole/polarization fit residuals by dataset frame:")
    print(table.to_string(index=False))
    print("*: high-residual outlier (modified z-score > 3.5 based on the MAD)")


def prepare_args(descr):

    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="path to extxyz file with all structures produced by 'build_dataset4dPdR.py' (e.g. structures.extxyz)",
    )
    parser.add_argument(
        "-r",
        "--retry",
        type=str2bool,
        required=False,
        help="retry linear fit without bad data (default: %(default)s)",
        default=False,
    )
    parser.add_argument(
        "--refit-outliers",
        action="store_true",
        help="force a refit after excluding all Cartesian equations of robust per-frame outliers",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        type=str,
        required=False,
        help="folder for the output files (default: %(default)s)",
        default=".",
    )
    parser.add_argument(
        "--cache-dir",
        **argv,
        default=".fd2bec",
        help="folder used for cached symmetry data (default: %(default)s)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="do not read or write the symmetry-data cache",
    )
    add_shared_argument(parser, "symprec")
    return parser


@cli(prepare_args, description)
def main(args):

    assert Path(args.input).suffix == ".extxyz", f"'{args.input}' must be an extxyz file."

    structures: List[Atoms] = read_input_structures(args.input, index=":", input_format="extxyz")

    Ns = len(structures)
    Na = structures[0].get_global_number_of_atoms()
    print("n. structures: ", Ns)
    print("n. atoms: ", Na)

    key = KEYWORDS["dipole"]
    print(f"Extracting dipole from '{key}' ... ", end="")
    dipole = np.asarray([atoms.info[key] for atoms in structures])
    print("done")
    print("dipole.shape: ", dipole.shape)

    key = KEYWORDS["displacements"]
    print(f"Extracting displacements from '{key}' ... ", end="")
    displacements = np.asarray([atoms.arrays[key] for atoms in structures])
    print("done")
    print("displacements.shape: ", displacements.shape)
    directions = displacement_directions(displacements)

    print("Reconstructing reference structure ... ", end="")
    positions = np.asarray([atoms.get_positions() for atoms in structures])
    reference = positions - displacements
    print("done")
    var = np.var(reference, axis=0)
    var_norm = np.linalg.norm(var)
    if var_norm > ATOL:
        raise ValueError("There has been a problem while reconstrucing the reference structure.")

    ref_pos = np.mean(reference, axis=0)
    reference_atoms = Atoms(
        positions=ref_pos,
        cell=structures[0].get_cell(),
        pbc=structures[0].get_pbc(),
        symbols=structures[0].get_chemical_symbols(),
    )
    reference = AtomicStructure.from_ase(reference_atoms, symprec=args.symprec)

    print("Preparing Born Charges and symmetrization ... ", end="")
    bec = BornCharges(data=np.zeros((Na, 3, 3)), cell=reference.cell)
    cache_metadata = born_charge_mode_cache_metadata(args.symprec)
    component_modes = None
    if not args.no_cache:
        component_modes = restore_born_charge_mode_cache(
            args.cache_dir, cache_metadata, reference_atoms
        )
    if component_modes is None:
        _, _, component_modes = reference.get_symmetry_modes(bec)
        if not args.no_cache:
            cache_path = save_born_charge_mode_cache(
                args.cache_dir, cache_metadata, reference_atoms, component_modes
            )
            print(f"cached in '{cache_path}'", end="; ")
    else:
        print(f"reused from '{args.cache_dir}'", end="; ")
    mode_basis = component_modes.T
    A = np.kron(displacements.reshape((Ns, -1)), np.eye(3))
    b = dipole.flatten()
    print("done")
    n_unknown = mode_basis.shape[1]

    print("\nMatrix shapes:")
    print(" - b.shape:", b.shape)
    print(" - A.shape:", A.shape)
    print(" - mode_basis.shape:", mode_basis.shape)

    print("\nMatrix shapes with symmetrization:")
    A = A @ mode_basis
    print(" - b.shape:", b.shape)
    print(" - A.shape:", A.shape)

    print("\nMatrix for total dipoles:")
    tmp = np.tile(-np.eye(3), Ns).T
    A = np.hstack([A, tmp])
    print(" - b.shape:", b.shape)
    print(" - A.shape:", A.shape)

    print("Preparing linear system ... ", end="")
    ls = LinearSystem(A=A, b=b)
    print("done")

    print("Solving linear systems  ... ", end="")
    ls.solve()
    print("done")

    info = ls.summary()
    frame_diagnostics = frame_residual_diagnostics(ls.residual_vector, group_size=3)
    displacement_norms, maximum_atomic_displacements = displacement_magnitudes(displacements)
    print_dipole_point_residuals(
        frame_diagnostics,
        directions,
        displacement_norms,
        maximum_atomic_displacements,
    )
    folder = Path(args.output)
    print("Writing dipole-fit visual report ... ", end="")
    image, report = write_dipole_fit_residual_report(
        folder,
        frame_diagnostics,
        directions,
        displacement_norms,
        maximum_atomic_displacements,
        n_equations=ls.n_rows,
        n_unknowns=ls.n_cols,
        rank=ls.rank,
        condition_number=ls.condition_number,
        relative_residual=ls.relative_residual[0],
    )
    print(f"done ({image}, {report})")

    if np.any(frame_diagnostics.outliers) and args.refit_outliers:
        outlier_frames = np.flatnonzero(frame_diagnostics.outliers)
        print("Refitting without outlier dataset frames: " + ", ".join(map(str, outlier_frames)))
        mask = retained_equation_mask(frame_diagnostics.outliers, group_size=3)
        A = A[mask, :]
        b = b[mask]

        print("Preparing linear system ... ", end="")
        ls = LinearSystem(A=A, b=b)
        print("done")

        print("Solving linear systems  ... ", end="")
        ls.solve()
        print("done")
        ls.summary()

        refit_diagnostics = frame_residual_diagnostics(ls.residual_vector, group_size=3)
        refit_directions = directions[~frame_diagnostics.outliers]
        refit_displacement_norms = displacement_norms[~frame_diagnostics.outliers]
        refit_maximum_atomic_displacements = maximum_atomic_displacements[
            ~frame_diagnostics.outliers
        ]
        print("Writing outlier-excluded dipole-fit visual report ... ", end="")
        image, report = write_dipole_fit_residual_report(
            folder,
            refit_diagnostics,
            refit_directions,
            refit_displacement_norms,
            refit_maximum_atomic_displacements,
            n_equations=ls.n_rows,
            n_unknowns=ls.n_cols,
            rank=ls.rank,
            condition_number=ls.condition_number,
            relative_residual=ls.relative_residual[0],
            stem="dipole-fit-refit",
            frame_indices=np.flatnonzero(~frame_diagnostics.outliers),
        )
        print(f"done ({image}, {report})")
    elif np.any(frame_diagnostics.outliers):
        print("Outlier-excluded refit disabled; retaining all dataset frames.")
    elif info["quality"] == "BAD" and args.retry:
        print("Removing bad data and retrying to solve the linear system")
        mask = info["good_mask"]
        A = A[mask, :]
        b = b[mask]

        print("Preparing linear system ... ", end="")
        ls = LinearSystem(A=A, b=b)
        print("done")

        print("Solving linear systems  ... ", end="")
        ls.solve()
        print("done")

        ls.summary()

    print("Extracting Born Charges  ... ", end="")
    bec = mode_basis @ ls.x[:n_unknown]
    bec = bec.reshape((Na, 3, 3))
    print("done\n")

    print_born_charges(reference, bec)

    print("Checking Acoustic Sum Rule:")
    # The acoustic sum rule is a sum over atoms. Keep that sum for the
    # diagnostic/output file, but subtract its per-atom average when applying
    # the correction so that the corrected tensors sum exactly to zero.
    asr_sum, bec_with_asr = apply_acoustic_sum_rule(bec)
    asr_mean = np.mean(bec, axis=0)
    norm = matrix_norm(asr_mean)
    print(f"||mean_I Z*_I|| = {norm:10.5f} | this value should be almost zero")

    file = folder / "bec-no-asr.txt"
    print(f"Writing Born Charges to {file} ... ", end="")
    np.savetxt(file, bec.reshape((Na, 9)), fmt=float_format)
    print("done")

    file = folder / "asr.txt"
    print(f"Writing sum of all Born Charges to {file} ... ", end="")
    np.savetxt(file, asr_sum, fmt=float_format)
    print("done")

    diagnostics = folder / "asr-diagnostics.txt"
    print(f"Writing acoustic sum-rule diagnostics to {diagnostics} ... ", end="")
    write_acoustic_sum_rule_diagnostics(
        diagnostics,
        bec,
        reference_atoms.get_chemical_symbols(),
        reference_atoms.get_positions(),
        tolerance=ATOL,
    )
    print("done")
    if norm > ATOL:
        msg = (
            f"Mean Born-charge acoustic sum-rule residual is not satisfied: "
            f"{norm:.5e} > {ATOL:.1e}. "
            f"Inspect {diagnostics}."
        )
        raise ValueError(msg)

    file = folder / "bec.txt"
    print(f"Writing sum of Born Charges with ASR applied to {file} ... ", end="")
    np.savetxt(file, bec_with_asr.reshape((Na, 9)), fmt=float_format)
    print("done")

    file = folder / "bec.extxyz"
    key = KEYWORDS["bec"]
    print(f"Writing Born Charges to {file} under '{key}' ... ", end="")
    write_tensor_extxyz(file, reference_atoms, bec_with_asr, key, per_atom=True)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
