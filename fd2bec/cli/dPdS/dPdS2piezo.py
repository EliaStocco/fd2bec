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
from fd2bec.piezoelectric import (
    E_PER_ANGSTROM2_TO_C_PER_M2,
    evaluate_dipole_lattice_derivative,
    evaluate_proper_piezoelectric_direct,
    piezoelectric_symbolic_matrix,
    piezoelectric_to_voigt,
    proper_piezoelectric_symmetry_basis,
    strains_from_cells,
)

description = (
    "Evaluate proper and improper piezoelectric tensors from the same set of "
    "polarized strained structures. Dipole inputs must be in e*Angstrom and "
    "are converted to polarization by dividing by the cell volume in Angstrom^3."
)


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
        help=(
            "info key for dipoles in e*Angstrom used in auto/dipole mode " "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--no-symmetry",
        action="store_true",
        help="disable crystal-symmetry constraints in the direct proper-tensor fit",
    )
    parser.add_argument(
        "--no-unwrap",
        action="store_true",
        help="disable polarization-quantum branch alignment",
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

    if args.polarizations:
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
        converted_dipoles = attach_polarizations(
            structures,
            quantity=args.quantity,
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
    if not args.no_unwrap:
        from fd2bec.piezoelectric import unwrap_polarizations

        reference_index = int(np.argmin(np.linalg.norm(strains.reshape(len(strains), -1), axis=1)))
        fitted_polarizations = unwrap_polarizations(
            fitted_polarizations,
            cells,
            reference_index,
            quantum_scale=quantum_scale,
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
    symbolic_pattern = piezoelectric_symbolic_matrix(direct_basis)

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

    direct_proper, direct_reference, direct_rank, direct_rms = evaluate_proper_piezoelectric_direct(
        fitted_polarizations,
        strains,
        direct_basis,
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

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(output / "improper-piezoelectric.txt", result.improper_voigt, fmt=float_format)
    np.savetxt(output / "proper-piezoelectric.txt", result.proper_voigt, fmt=float_format)
    np.savetxt(output / "proper-piezoelectric-direct.txt", direct_voigt, fmt=float_format)
    np.savetxt(
        output / "dipole-strain-derivative.txt",
        piezoelectric_to_voigt(dipole_fit.dipole_strain_derivative),
        fmt=float_format,
    )
    np.savetxt(
        output / "dipole-lattice-derivative.txt",
        dipole_fit.dipole_lattice_derivative.reshape((3, 9)),
        fmt=float_format,
    )
    np.savetxt(
        output / "reference-polarization.txt",
        result.reference_polarization[None, :],
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
                "polarization_unit": args.polarization_unit,
                "symmetry_enabled": not args.no_symmetry,
                "symmetry_pattern": symbolic_pattern.tolist(),
                "direct_proper_parameters": int(direct_basis.shape[1]),
                "direct_proper_rank": direct_rank,
                "direct_proper_residual_rms": direct_rms,
                "direct_reference_polarization": direct_reference.tolist(),
                "proper_tensors_agree": proper_agreement,
                "proper_tensor_max_abs_difference": proper_difference,
                "proper_tensor_agreement_tolerance": args.agreement_tolerance,
                "dipole_lattice_fit_rank": int(dipole_fit.linear_system.rank),
                "dipole_lattice_fit_residual_rms": result.residual_rms,
            },
            handle,
            indent=2,
        )

    matrix_format = {"precision": 6, "suppress_small": True, "max_line_width": 200}
    print("Voigt order: xx, yy, zz, yz, xz, xy.")
    print("Engineering strain uses [exx, eyy, ezz, 2eyz, 2exz, 2exy].")
    print("\nSymmetry-allowed proper piezoelectric pattern [3x6]:")
    width = max(3, max(len(value) for value in symbolic_pattern.reshape(-1)))
    for row in symbolic_pattern:
        print("[" + " ".join(f"{value:>{width}}" for value in row) + "]")
    print("\nd(dipole)/d(lattice vectors) [3x9]:")
    print(
        np.array2string(
            dipole_fit.dipole_lattice_derivative.reshape((3, 9)),
            **matrix_format,
        )
    )
    print("\nImproper piezoelectric tensor [3x6]:")
    print(np.array2string(result.improper_voigt, **matrix_format))
    print("\nProper piezoelectric tensor from Vanderbilt correction [3x6]:")
    print(np.array2string(result.proper_voigt, **matrix_format))
    print("\nDirect ProperPiezoelectricTensor fit [3x6]:")
    print(np.array2string(direct_voigt, **matrix_format))
    agreement_label = "AGREE" if proper_agreement else "DO NOT AGREE"
    print()
    print(
        f"Proper-tensor check: {agreement_label}; maximum |difference| = "
        f"{proper_difference:.6e} (absolute tolerance {args.agreement_tolerance:.6e})."
    )
    print("Converse stress convention: delta_sigma_V = - proper_e.T @ electric_field.")
    print(
        f"Direct fit: {direct_basis.shape[1]} symmetry-allowed tensor parameters; "
        f"rank {direct_rank}; RMS residual {direct_rms:.6e}."
    )
    print(f"Fit rank: {result.rank}; RMS residual: {result.residual_rms:.6e}")
    print(
        "Saved improper, Vanderbilt proper, and direct proper piezoelectric "
        f"tensors to '{output}'."
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
