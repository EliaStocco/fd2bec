"""Fit proper and improper piezoelectric tensors from one strained dataset."""

import argparse
import json
import warnings
from pathlib import Path
from typing import List

import numpy as np
from ase import Atoms

from fd2bec import ATOL, float_format
from fd2bec.atomic import AtomicStructure
from fd2bec.cli import KEYWORDS, cli
from fd2bec.io import read
from fd2bec.mathematics import rotate_rank3
from fd2bec.piezoelectric import (
    canonical_piezoelectric_modes,
    evaluate_dipole_lattice_derivative,
    evaluate_proper_piezoelectric_direct,
    piezoelectric_symbolic_matrix,
    piezoelectric_to_voigt,
    proper_piezoelectric_symmetry_basis,
    strains_from_cells,
)
from fd2bec.show import print_reference_structure

description = (
    "Evaluate full and clamped piezoelectric tensors from strained structures. "
    "Each frame must contain a dipole in e*Angstrom."
)

VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")
CARTESIAN_LABELS = ("x", "y", "z")


def _display_number(value, precision=6, zero_tolerance=5e-10):
    """Return a fixed-width-friendly number without negative numerical zero."""
    value = 0.0 if abs(value) < zero_tolerance else value
    return f"{value:.{precision}f}"


def print_voigt_tensor(tensor, precision=6):
    """Print a numerical 3x6 tensor with Cartesian and Voigt labels."""
    tensor = np.asarray(tensor, dtype=float)
    if tensor.shape != (3, 6):
        raise ValueError(f"Expected a 3x6 tensor, got {tensor.shape}.")
    width = precision + 8
    print(" " * 6 + "".join(f"{label:>{width}}" for label in VOIGT_LABELS))
    for axis, row in zip(CARTESIAN_LABELS, tensor):
        values = "".join(f"{_display_number(value, precision):>{width}}" for value in row)
        print(f"P_{axis:<4s}{values}")


def print_lattice_tensor(tensor, precision=6):
    """Print a rank-3 lattice-basis tensor as three labeled 3x3 slices."""
    tensor = np.asarray(tensor, dtype=float)
    if tensor.shape != (3, 3, 3):
        raise ValueError(f"Expected a 3x3x3 tensor, got {tensor.shape}.")
    width = precision + 8
    for component, block in zip(CARTESIAN_LABELS, tensor):
        print(f"P_{component} component:")
        print(" " * 7 + "".join(f"{label:>{width}}" for label in CARTESIAN_LABELS))
        for axis, row in zip(CARTESIAN_LABELS, block):
            values = "".join(f"{_display_number(value, precision):>{width}}" for value in row)
            print(f"  {axis:<5s}{values}")


def prepare_args(descr):
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i", "--input", **argv, required=True, help="multi-frame extxyz dataset with dipoles"
    )
    parser.add_argument(
        "-r",
        "--reference",
        **argv,
        default=None,
        help="unstrained reference structure; defaults to the first input frame",
    )
    parser.add_argument(
        "--dipole-keyword",
        default=KEYWORDS["dipole"],
        help="info key for dipoles in e*Angstrom (default: %(default)s)",
    )
    parser.add_argument(
        "--no-symmetry",
        action="store_true",
        help="disable crystal-symmetry constraints in the clamped-tensor fit",
    )
    parser.add_argument(
        "--clamped",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "require identical fractional coordinates in every frame; use "
            "--no-clamped for internally relaxed structures"
        ),
    )
    parser.add_argument(
        "--conventional-axes",
        action="store_true",
        help=(
            "rotate reported and saved Cartesian tensors into spglib's "
            "conventional crystallographic axes"
        ),
    )
    parser.add_argument(
        "--no-unwrap",
        action="store_true",
        help="disable default periodic dipole branch alignment",
    )
    parser.add_argument(
        "--agreement-tolerance",
        type=float,
        default=1e-6,
        help="absolute tolerance comparing the two proper tensors (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        default="piezoelectric",
        help="output folder (default: %(default)s)",
    )
    return parser


def dipoles_to_polarizations(structures, dipole_keyword=KEYWORDS["dipole"]):
    """Convert per-frame e*Angstrom dipoles to polarization in e/Angstrom²."""
    polarizations = []
    for index, atoms in enumerate(structures):
        if dipole_keyword not in atoms.info:
            raise ValueError(
                f"Structure {index} has no '{dipole_keyword}' dipole in atoms.info. "
                "Dipoles must be supplied in e*Angstrom."
            )
        volume = atoms.get_volume()
        if volume <= 0:
            raise ValueError(f"Structure {index} has a non-positive cell volume.")
        vector = np.asarray(atoms.info[dipole_keyword], dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(
                f"Structure {index} contains an invalid three-component e*Angstrom dipole."
            )
        polarizations.append(vector / volume)
    return np.asarray(polarizations)


def fractional_coordinates_are_clamped(structures: List[Atoms], reference: Atoms, atol=ATOL):
    """Return whether all frames retain the reference fractional coordinates."""
    reference_positions = reference.get_scaled_positions(wrap=False)
    for index, atoms in enumerate(structures):
        positions = atoms.get_scaled_positions(wrap=False)
        if positions.shape != reference_positions.shape:
            raise ValueError(
                f"Structure {index} has {len(positions)} atoms, but the reference has "
                f"{len(reference_positions)}."
            )
        difference = positions - reference_positions
        difference -= np.rint(difference)
        if not np.allclose(difference, 0.0, atol=atol, rtol=0.0):
            return False
    return True


def validate_clamped_coordinates(structures, reference, clamped):
    """Validate that the coordinate behavior agrees with the selected workflow."""
    coordinates_are_clamped = fractional_coordinates_are_clamped(structures, reference)
    if clamped and not coordinates_are_clamped:
        raise ValueError(
            "--clamped requires identical fractional coordinates in every structure. "
            "Use --no-clamped for an internally relaxed dataset."
        )
    if not clamped and coordinates_are_clamped:
        warnings.warn(
            "--no-clamped was selected, but every structure has the same fractional "
            "coordinates as the reference.",
            UserWarning,
            stacklevel=2,
        )
    return coordinates_are_clamped


@cli(prepare_args, description)
def main(args):
    if args.agreement_tolerance < 0:
        raise ValueError("--agreement-tolerance must be non-negative.")
    structures = read(args.input, format="extxyz", index=":")
    if not structures:
        raise ValueError("The input dataset contains no structures.")
    reference = read(args.reference, index=0) if args.reference else structures[0].copy()
    print_reference_structure(reference)
    validate_clamped_coordinates(structures, reference, args.clamped)

    fitted_polarizations = dipoles_to_polarizations(structures, args.dipole_keyword)
    print(
        f"Converted '{args.dipole_keyword}' dipoles in e*Angstrom to polarization "
        "using each cell volume."
    )
    cells = np.asarray([atoms.cell.array for atoms in structures])
    strains = strains_from_cells(reference.cell.array, cells)
    unwrap_enabled = not args.no_unwrap
    if unwrap_enabled:
        from fd2bec.piezoelectric import unwrap_polarizations

        reference_index = int(np.argmin(np.linalg.norm(strains.reshape(len(strains), -1), axis=1)))
        fitted_polarizations = unwrap_polarizations(
            fitted_polarizations,
            cells,
            reference_index,
            quantum_scale=1.0,
        )
        print("Aligned dipole branches using each snapshot's cell-dependent quantum.")
    else:
        print("Warning: dipole branch alignment was disabled; periodic dipoles can be multivalued.")

    unit_cell = AtomicStructure.from_ase(reference)
    if args.no_symmetry:
        # Eighteen independent components after enforcing j-k strain symmetry.
        full_modes = np.eye(27).reshape((27, 3, 3, 3))
        full_modes = 0.5 * (full_modes + full_modes.swapaxes(2, 3))
        flattened = full_modes.reshape((27, 27)).T
        left, singular_values, _ = np.linalg.svd(flattened, full_matrices=False)
        direct_basis = left[:, singular_values > 1e-10]
    else:
        direct_basis = proper_piezoelectric_symmetry_basis(unit_cell)
    dataset = unit_cell._spglib_dataset  # pylint: disable=protected-access
    space_group_symbol = dataset.international
    if isinstance(space_group_symbol, bytes):
        space_group_symbol = space_group_symbol.decode()
    coordinate_rotation = (
        np.asarray(dataset.std_rotation_matrix, dtype=float)
        if args.conventional_axes
        else np.eye(3)
    )
    reported_basis = (
        np.column_stack(
            [
                np.einsum(
                    "ai,bj,ck,ijk->abc",
                    coordinate_rotation,
                    coordinate_rotation,
                    coordinate_rotation,
                    mode.reshape((3, 3, 3)),
                ).reshape(-1)
                for mode in direct_basis.T
            ]
        )
        if direct_basis.shape[1]
        else direct_basis.copy()
    )
    symmetry_modes, independent_components = canonical_piezoelectric_modes(reported_basis)
    symbolic_matrix = piezoelectric_symbolic_matrix(reported_basis)
    coefficient_names = [
        f"e{int(component) // 6 + 1}{int(component) % 6 + 1}"
        for component in independent_components
    ]
    print(
        f"Space group: {dataset.number} ({space_group_symbol}); "
        f"point group {dataset.pointgroup}; "
        f"{len(dataset.rotations)} symmetry operations; "
        f"{direct_basis.shape[1]} allowed clamped-piezoelectric parameters."
    )
    names = ", ".join(coefficient_names) if coefficient_names else "none"
    print(f"Selected independent coefficient representatives: {names}.")
    if args.conventional_axes:
        print("Rotating reported and saved tensors into conventional crystallographic axes.")
        print("Cartesian coordinate rotation (conventional <- input):")
        print(np.array2string(coordinate_rotation, precision=8, suppress_small=True))

    fitted_dipoles = fitted_polarizations * np.abs(np.linalg.det(cells))[:, None]
    dipole_fit = evaluate_dipole_lattice_derivative(
        fitted_dipoles,
        cells,
        reference.cell.array,
        proper_symmetry_basis=None if args.no_symmetry else direct_basis,
    )
    result = dipole_fit.result
    print("\nDipole/lattice linear system:")
    print(f" - b.shape: {dipole_fit.linear_system.b.shape}")
    print(f" - A.shape: {dipole_fit.linear_system.A.shape}")
    print(f" - rank: {dipole_fit.linear_system.rank}")
    dipole_fit.linear_system.summary()

    inverse_reference_cell = np.linalg.inv(reference.cell.array)
    displacement_gradients = np.asarray(
        [(inverse_reference_cell @ cell).T - np.eye(3) for cell in cells]
    )
    rotations = 0.5 * (displacement_gradients - displacement_gradients.swapaxes(1, 2))
    clamped_tensor, clamped_reference, clamped_rank, clamped_rms = (
        evaluate_proper_piezoelectric_direct(
            fitted_polarizations,
            strains,
            direct_basis,
            rotations=rotations,
            cell=reference.cell.array,
        )
    )
    clamped_voigt = piezoelectric_to_voigt(clamped_tensor.data)
    full_clamped_difference = float(np.max(np.abs(result.proper_voigt - clamped_voigt)))
    full_clamped_agreement = bool(
        np.allclose(
            result.proper_voigt,
            clamped_voigt,
            atol=args.agreement_tolerance,
            rtol=1e-5,
        )
    )
    if args.clamped and not full_clamped_agreement:
        warnings.warn(
            "The full and clamped piezoelectric tensors differ for a clamped dataset: "
            f"maximum absolute difference {full_clamped_difference:.6e} exceeds the "
            f"requested tolerance {args.agreement_tolerance:.6e}.",
            UserWarning,
            stacklevel=2,
        )

    reported_improper_voigt = piezoelectric_to_voigt(
        rotate_rank3(result.improper.data, coordinate_rotation)
    )
    reported_full_voigt = piezoelectric_to_voigt(
        rotate_rank3(result.proper.data, coordinate_rotation)
    )
    reported_clamped_voigt = piezoelectric_to_voigt(
        rotate_rank3(clamped_tensor.data, coordinate_rotation)
    )
    coefficient_values = {
        symbol: float(reported_clamped_voigt.reshape(-1)[component])
        for symbol, component in zip(coefficient_names, independent_components)
    }
    reported_dipole_strain = rotate_rank3(dipole_fit.dipole_strain_derivative, coordinate_rotation)
    reported_reference_polarization = coordinate_rotation @ result.reference_polarization
    # full_lattice_basis = result.proper.to(basis="fractional").data

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(output / "improper-piezoelectric.txt", reported_improper_voigt, fmt=float_format)
    np.savetxt(output / "full-piezoelectric.txt", reported_full_voigt, fmt=float_format)
    np.savetxt(output / "clamped-piezoelectric.txt", reported_clamped_voigt, fmt=float_format)
    np.savetxt(
        output / "dipole-strain-derivative.txt",
        piezoelectric_to_voigt(reported_dipole_strain),
        fmt=float_format,
    )
    np.savetxt(
        output / "dipole-lattice-derivative.txt",
        dipole_fit.dipole_lattice_derivative.reshape((3, 9)),
        fmt=float_format,
    )
    np.savetxt(
        output / "reference-polarization.txt",
        reported_reference_polarization[None, :],
        fmt=float_format,
    )
    with (output / "fit.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "voigt_order": ["xx", "yy", "zz", "yz", "xz", "xy"],
                "shear_convention": ["exx", "eyy", "ezz", "2eyz", "2exz", "2exy"],
                "rank": int(result.rank),
                "residual_rms": result.residual_rms,
                "dipole_keyword": args.dipole_keyword,
                "branch_unwrapping_enabled": unwrap_enabled,
                "dipole_unit": "e*Angstrom",
                "polarization_unit": "e/Angstrom^2",
                "clamped": args.clamped,
                "symmetry_enabled": not args.no_symmetry,
                "coordinate_frame": "conventional" if args.conventional_axes else "input",
                "coordinate_rotation_conventional_from_input": coordinate_rotation.tolist(),
                "space_group_number": int(dataset.number),
                "space_group_symbol": str(space_group_symbol),
                "point_group_symbol": str(dataset.pointgroup),
                "independent_coefficient_names": coefficient_names,
                "independent_coefficient_values": coefficient_values,
                "symmetry_operations": int(len(dataset.rotations)),
                "symmetry_pattern": symbolic_matrix.tolist(),
                "symmetry_symbolic_matrix": symbolic_matrix.tolist(),
                "symmetry_modes": symmetry_modes.tolist(),
                "symmetry_mode_independent_components": [
                    int(component) for component in independent_components
                ],
                "clamped_tensor_parameters": int(direct_basis.shape[1]),
                "clamped_tensor_rank": clamped_rank,
                "clamped_tensor_residual_rms": clamped_rms,
                "clamped_reference_polarization": (
                    coordinate_rotation @ clamped_reference
                ).tolist(),
                "full_clamped_comparison_applicable": args.clamped,
                "full_and_clamped_tensors_agree": (
                    full_clamped_agreement if args.clamped else None
                ),
                "full_clamped_tensor_max_abs_difference": full_clamped_difference,
                "full_clamped_tensor_agreement_tolerance": args.agreement_tolerance,
                "dipole_lattice_fit_rank": int(dipole_fit.linear_system.rank),
                "dipole_lattice_fit_residual_rms": result.residual_rms,
            },
            handle,
            indent=2,
        )

    piezoelectric_unit = "e/Angstrom^2"
    # lattice_basis_unit = "e/Angstrom"
    print("Voigt order: xx, yy, zz, yz, xz, xy.")
    print("Engineering strain uses [exx, eyy, ezz, 2eyz, 2exz, 2exy].")
    print("\nSymmetry-allowed clamped piezoelectric matrix [3x6]:")
    frame_label = "conventional" if args.conventional_axes else "input"
    print(f"(letters are independent parameters in the {frame_label} Cartesian axes)")
    symbolic_width = max(1, max(len(value) for value in symbolic_matrix.flat))
    for row in symbolic_matrix:
        print("[ " + "  ".join(f"{value:>{symbolic_width}}" for value in row) + " ]")
    print(f"\nImproper piezoelectric tensor [3x6, {piezoelectric_unit}]:")
    print_voigt_tensor(reported_improper_voigt)
    print(
        "\nFull piezoelectric tensor from the dipole/lattice linear system "
        f"[3x6, {piezoelectric_unit}]:"
    )
    print_voigt_tensor(reported_full_voigt)
    # print(
    #     f"\nFull piezoelectric tensor in the reference lattice basis [3x3x3, {lattice_basis_unit}]:"
    # )
    # print("(full fractional-basis tensor; no Cartesian engineering-Voigt contraction)")
    # print_lattice_tensor(full_lattice_basis)
    print(f"\nClamped ProperPiezoelectricTensor fit [3x6, {piezoelectric_unit}]:")
    print_voigt_tensor(reported_clamped_voigt)
    print(f"\nClamped piezoelectric coefficient [{piezoelectric_unit}]:")
    if coefficient_values:
        for symbol, value in coefficient_values.items():
            print(f"{symbol}: {value:.6g}")
    elif coefficient_values == {}:
        print("none")
    print()
    if args.clamped:
        agreement_label = "AGREE" if full_clamped_agreement else "DO NOT AGREE"
        print(
            f"Full/clamped tensor check: {agreement_label}; maximum |difference| = "
            f"{full_clamped_difference:.6e} {piezoelectric_unit} "
            f"(absolute tolerance {args.agreement_tolerance:.6e} {piezoelectric_unit})."
        )
    else:
        print("Full/clamped tensor agreement check skipped for internally relaxed structures.")
    print("Converse stress convention: delta_sigma_V = - proper_e.T @ electric_field.")
    print(
        f"Clamped fit: {direct_basis.shape[1]} symmetry-allowed tensor parameters; "
        f"rank {clamped_rank}; RMS residual {clamped_rms:.6e}."
    )
    print(f"Fit rank: {result.rank}; RMS residual: {result.residual_rms:.6e}")
    print(f"Saved improper, full, and clamped piezoelectric tensors to '{output}'.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
