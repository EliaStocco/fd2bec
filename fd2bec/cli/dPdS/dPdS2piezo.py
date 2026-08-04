"""Fit proper and improper piezoelectric tensors from one strained dataset."""

import argparse
import json
from pathlib import Path

import numpy as np

from fd2bec import float_format
from fd2bec.cli import KEYWORDS, cli
from fd2bec.io import read
from fd2bec.piezoelectric import E_PER_ANGSTROM2_TO_C_PER_M2, evaluate_piezoelectric_from_structures

description = (
    "Evaluate proper and improper piezoelectric tensors from the same set of "
    "polarized strained structures."
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
        "-o",
        "--output",
        **argv,
        default="piezoelectric",
        help="output folder (default: %(default)s)",
    )
    return parser


@cli(prepare_args, description)
def main(args):
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

    quantum_scale = E_PER_ANGSTROM2_TO_C_PER_M2 if args.polarization_unit == "C/m^2" else 1.0
    result = evaluate_piezoelectric_from_structures(
        structures,
        reference,
        polarization_key=args.keyword,
        unwrap=not args.no_unwrap,
        polarization_quantum_scale=quantum_scale,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(output / "improper-piezoelectric.txt", result.improper_voigt, fmt=float_format)
    np.savetxt(output / "proper-piezoelectric.txt", result.proper_voigt, fmt=float_format)
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
                "rank": result.rank,
                "residual_rms": result.residual_rms,
                "polarization_keyword": args.keyword,
                "polarization_unit": args.polarization_unit,
            },
            handle,
            indent=2,
        )

    print("Voigt order: xx, yy, zz, yz, xz, xy (engineering shear).")
    print(f"Fit rank: {result.rank}; RMS residual: {result.residual_rms:.6e}")
    print(f"Saved proper and improper piezoelectric tensors to '{output}'.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
