"""Fit proper and improper piezoelectric tensors from one strained dataset."""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from fd2bec import float_format
from fd2bec.atomic import AtomicStructure
from fd2bec.cli import KEYWORDS, cli
from fd2bec.io import read
from fd2bec.mathematics import rotate_rank3
from fd2bec.piezoelectric import (
    E_PER_ANGSTROM2_TO_C_PER_M2,
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
    "Evaluate proper and improper piezoelectric tensors from the same set of "
    "polarized strained structures. Dipole inputs must be in e*Angstrom and "
    "are converted to polarization by dividing by the cell volume in Angstrom^3."
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
        "-i", "--input", **argv, required=True, help="polarized multi-frame extxyz dataset"
    )
    parser.add_argument(
        "-r",
        "--reference",
        **argv,
        default=None,
        help="unstrained reference structure; defaults to the first input frame",
    )
    parser.add_argument(
        "-p",
        "--polarizations",
        **argv,
        default=None,
        help="optional N x 3 polarization text file",
    )
    parser.add_argument(
        "-k",
        "--keyword",
        **argv,
        default=KEYWORDS["polarization"],
        help="polarization info key when --polarizations is omitted (default: %(default)s)",
    )
    parser.add_argument(
        "--quantity",
        choices=("auto", "polarization", "dipole"),
        default="auto",
        help=(
            "input vector type; auto prefers polarization and falls back to "
            "dipole [e*Angstrom]/volume [Angstrom^3] (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--dipole-keyword",
        default=KEYWORDS["dipole"],
        help=("info key for dipoles in e*Angstrom used in auto/dipole mode (default: %(default)s)"),
    )
    parser.add_argument(
        "--no-symmetry",
        action="store_true",
        help="disable crystal-symmetry constraints in the direct proper-tensor fit",
    )
    parser.add_argument(
        "--conventional_axes",
        action="store_true",
        help=(
            "rotate reported and saved Cartesian tensors into spglib's "
            "conventional crystallographic axes"
        ),
    )
    parser.add_argument(
        "--no-unwrap",
        action="store_true",
        help="disable default polarization/dipole branch alignment",
    )
    parser.add_argument(
        "--polarization-unit",
        choices=("e/angstrom^2", "C/m^2"),
        default="e/angstrom^2",
        help="input/output polarization unit (default: %(default)s)",
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


def attach_polarizations(
    structures,
    *,
    quantity="auto",
    polarization_keyword=KEYWORDS["polarization"],
    dipole_keyword=KEYWORDS["dipole"],
    polarization_unit="e/angstrom^2",
):
    """Ensure every structure has polarization, converting dipole/volume if needed."""
    if quantity not in ("auto", "polarization", "dipole"):
        raise ValueError(f"Unsupported input quantity: '{quantity}'.")
    if polarization_unit not in ("e/angstrom^2", "C/m^2"):
        raise ValueError(f"Unsupported polarization unit: '{polarization_unit}'.")

    converted = 0
    for index, atoms in enumerate(structures):
        use_polarization = quantity != "dipole" and polarization_keyword in atoms.info
        if use_polarization:
            vector = np.asarray(atoms.info[polarization_keyword], dtype=float)
        elif quantity != "polarization" and dipole_keyword in atoms.info:
            volume = atoms.get_volume()
            if volume <= 0:
                raise ValueError(f"Structure {index} has a non-positive cell volume.")
            vector = np.asarray(atoms.info[dipole_keyword], dtype=float) / volume
            if polarization_unit == "C/m^2":
                vector = vector * E_PER_ANGSTROM2_TO_C_PER_M2
            atoms.info[polarization_keyword] = vector
            converted += 1
        else:
            expected = (
                f"'{polarization_keyword}'"
                if quantity == "polarization"
                else (
                    f"'{dipole_keyword}'"
                    if quantity == "dipole"
                    else f"'{polarization_keyword}' or '{dipole_keyword}'"
                )
            )
            raise ValueError(f"Structure {index} has no {expected} vector in atoms.info.")

        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"Structure {index} contains an invalid three-component vector.")
    return converted


@cli(prepare_args, description)
def main(args):
    if args.agreement_tolerance < 0:
        raise ValueError("--agreement-tolerance must be non-negative.")
    structures = read(args.input, format="extxyz", index=":")
    if not structures:
        raise ValueError("The input dataset contains no structures.")
    reference = read(args.reference, index=0) if args.reference else structures[0].copy()
    print_reference_structure(reference)

    if args.polarizations:
        input_vector_type = "polarization"
        polarizations = np.loadtxt(args.polarizations, ndmin=2)
        if polarizations.shape != (len(structures), 3):
            raise ValueError(
                f"Expected polarizations with shape ({len(structures)}, 3), "
                f"got {polarizations.shape}."
            )
        for atoms, polarization in zip(structures, polarizations):
            atoms.info[args.keyword] = polarization
        converted_dipoles = 0
    else:
        all_polarizations = all(args.keyword in atoms.info for atoms in structures)
        all_dipoles = all(args.dipole_keyword in atoms.info for atoms in structures)
        if args.quantity == "polarization":
            input_vector_type = "polarization"
        elif args.quantity == "dipole":
            input_vector_type = "dipole"
        elif all_polarizations:
            input_vector_type = "polarization"
        elif all_dipoles:
            input_vector_type = "dipole"
        else:
            raise ValueError(
                "Automatic input detection requires every frame to contain the same "
                f"'{args.keyword}' polarization or '{args.dipole_keyword}' dipole field."
            )
        converted_dipoles = attach_polarizations(
            structures,
            quantity=input_vector_type,
            polarization_keyword=args.keyword,
            dipole_keyword=args.dipole_keyword,
            polarization_unit=args.polarization_unit,
        )

    if converted_dipoles:
        print(
            f"Converted '{args.dipole_keyword}' to '{args.keyword}' as dipole/cell volume "
            f"for {converted_dipoles} structure(s)."
        )

    quantum_scale = E_PER_ANGSTROM2_TO_C_PER_M2 if args.polarization_unit == "C/m^2" else 1.0
    cells = np.asarray([atoms.cell.array for atoms in structures])
    strains = strains_from_cells(reference.cell.array, cells)
    fitted_polarizations = np.asarray([atoms.info[args.keyword] for atoms in structures])
    unwrap_enabled = not args.no_unwrap
    if unwrap_enabled:
        from fd2bec.piezoelectric import unwrap_polarizations

        reference_index = int(np.argmin(np.linalg.norm(strains.reshape(len(strains), -1), axis=1)))
        fitted_polarizations = unwrap_polarizations(
            fitted_polarizations,
            cells,
            reference_index,
            quantum_scale=quantum_scale,
        )
        print("Aligned polarization branches using each snapshot's cell-dependent quantum.")
    elif input_vector_type == "dipole":
        print(
            "Warning: dipole branch alignment was disabled with --no-unwrap; periodic "
            "dipoles can be multivalued."
        )

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
        f"{direct_basis.shape[1]} allowed proper-piezoelectric parameters."
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
    direct_proper, direct_reference, direct_rank, direct_rms = evaluate_proper_piezoelectric_direct(
        fitted_polarizations,
        strains,
        direct_basis,
        rotations=rotations,
        cell=reference.cell.array,
    )
    direct_voigt = piezoelectric_to_voigt(direct_proper.data)
    proper_difference = float(np.max(np.abs(result.proper_voigt - direct_voigt)))
    proper_agreement = bool(
        np.allclose(
            result.proper_voigt,
            direct_voigt,
            atol=args.agreement_tolerance,
            rtol=1e-5,
        )
    )
    if not proper_agreement:
        warnings.warn(
            "The Vanderbilt and direct proper tensors differ: "
            f"maximum absolute difference {proper_difference:.6e} exceeds the "
            f"requested tolerance {args.agreement_tolerance:.6e}.",
            UserWarning,
            stacklevel=2,
        )

    reported_improper_voigt = piezoelectric_to_voigt(
        rotate_rank3(result.improper.data, coordinate_rotation)
    )
    reported_proper_voigt = piezoelectric_to_voigt(
        rotate_rank3(result.proper.data, coordinate_rotation)
    )
    reported_direct_voigt = piezoelectric_to_voigt(
        rotate_rank3(direct_proper.data, coordinate_rotation)
    )
    coefficient_values = {
        symbol: float(reported_direct_voigt.reshape(-1)[component])
        for symbol, component in zip(coefficient_names, independent_components)
    }
    reported_dipole_strain = rotate_rank3(dipole_fit.dipole_strain_derivative, coordinate_rotation)
    reported_reference_polarization = coordinate_rotation @ result.reference_polarization
    # proper_lattice_basis = result.proper.to(basis="fractional").data

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(output / "improper-piezoelectric.txt", reported_improper_voigt, fmt=float_format)
    np.savetxt(output / "proper-piezoelectric.txt", reported_proper_voigt, fmt=float_format)
    np.savetxt(output / "proper-piezoelectric-direct.txt", reported_direct_voigt, fmt=float_format)
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
                "polarization_keyword": args.keyword,
                "dipole_keyword": args.dipole_keyword,
                "input_quantity": args.quantity,
                "detected_input_quantity": input_vector_type,
                "branch_unwrapping_enabled": unwrap_enabled,
                "polarization_unit": args.polarization_unit,
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
                "direct_proper_parameters": int(direct_basis.shape[1]),
                "direct_proper_rank": direct_rank,
                "direct_proper_residual_rms": direct_rms,
                "direct_reference_polarization": (coordinate_rotation @ direct_reference).tolist(),
                "proper_tensors_agree": proper_agreement,
                "proper_tensor_max_abs_difference": proper_difference,
                "proper_tensor_agreement_tolerance": args.agreement_tolerance,
                "dipole_lattice_fit_rank": int(dipole_fit.linear_system.rank),
                "dipole_lattice_fit_residual_rms": result.residual_rms,
            },
            handle,
            indent=2,
        )

    # matrix_format = {"precision": 6, "suppress_small": True, "max_line_width": 200}
    piezoelectric_unit = "C/m^2" if args.polarization_unit == "C/m^2" else "e/Angstrom^2"
    # lattice_basis_unit = "C*Angstrom/m^2" if args.polarization_unit == "C/m^2" else "e/Angstrom"
    print("Voigt order: xx, yy, zz, yz, xz, xy.")
    print("Engineering strain uses [exx, eyy, ezz, 2eyz, 2exz, 2exy].")
    print("\nSymmetry-allowed proper piezoelectric matrix [3x6]:")
    frame_label = "conventional" if args.conventional_axes else "input"
    print(f"(letters are independent parameters in the {frame_label} Cartesian axes)")
    symbolic_width = max(1, max(len(value) for value in symbolic_matrix.flat))
    for row in symbolic_matrix:
        print("[ " + "  ".join(f"{value:>{symbolic_width}}" for value in row) + " ]")
    # print("\nSymmetry-allowed proper piezoelectric modes [3x6]:")
    # print(f"(canonical modes expressed in the {frame_label} Cartesian axes)")
    # if not len(symmetry_modes):
    #     print("No symmetry-allowed modes: the proper piezoelectric tensor is zero.")
    # cartesian = "xyz"
    # for index, (mode, component) in enumerate(zip(symmetry_modes, independent_components)):
    #     label = chr(ord("a") + index) if index < 26 else f"a{index + 1}"
    #     polarization_axis, voigt_column = divmod(component, 6)
    #     anchor = f"e_{cartesian[polarization_axis]},{VOIGT_LABELS[voigt_column]}"
    #     print(f"Mode {label} ({anchor} = 1):")
    #     print_voigt_tensor(mode)
    # print("\nd(dipole)/d(lattice vectors) [3x9, input axes]:")
    # print(
    #     np.array2string(
    #         dipole_fit.dipole_lattice_derivative.reshape((3, 9)),
    #         **matrix_format,
    #     )
    # )
    print(f"\nImproper piezoelectric tensor [3x6, {piezoelectric_unit}]:")
    print_voigt_tensor(reported_improper_voigt)
    print(f"\nProper piezoelectric tensor from Vanderbilt correction [3x6, {piezoelectric_unit}]:")
    print_voigt_tensor(reported_proper_voigt)
    # print(
    #     "\nProper piezoelectric tensor in the reference lattice basis "
    #     f"[3x3x3, {lattice_basis_unit}]:"
    # )
    # print("(full fractional-basis tensor; no Cartesian engineering-Voigt contraction)")
    # print_lattice_tensor(proper_lattice_basis)
    print(f"\nDirect ProperPiezoelectricTensor fit [3x6, {piezoelectric_unit}]:")
    print_voigt_tensor(reported_direct_voigt)
    print(f"\nPiezoelectric coefficient [{piezoelectric_unit}]:")
    if coefficient_values:
        for symbol, value in coefficient_values.items():
            print(f"{symbol}: {value:.6g}")
    elif coefficient_values == {}:
        print("none")
    agreement_label = "AGREE" if proper_agreement else "DO NOT AGREE"
    print()
    print(
        f"Proper-tensor check: {agreement_label}; maximum |difference| = "
        f"{proper_difference:.6e} {piezoelectric_unit} "
        f"(absolute tolerance {args.agreement_tolerance:.6e} {piezoelectric_unit})."
    )
    print("Converse stress convention: delta_sigma_V = - proper_e.T @ electric_field.")
    print(
        f"Direct fit: {direct_basis.shape[1]} symmetry-allowed tensor parameters; "
        f"rank {direct_rank}; RMS residual {direct_rms:.6e}."
    )
    print(f"Fit rank: {result.rank}; RMS residual: {result.residual_rms:.6e}")
    print(
        f"Saved improper, Vanderbilt proper, and direct proper piezoelectric tensors to '{output}'."
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
