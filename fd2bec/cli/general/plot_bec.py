"""Plot directional responses of one atom's Born effective charge tensor."""

import argparse

from fd2bec.born_charges_plot import (
    born_charge_from_atoms,
    born_charges_from_atoms,
    plot_born_charge_response_comparison,
    plot_born_charge_structure_comparison,
    plot_born_charge_surface,
)
from fd2bec.cli import KEYWORDS, cli, read_input_structures
from fd2bec.piezoelectric_plot import RESPONSE_COMPONENTS

description = (
    "Create directional Born-effective-charge surfaces for one atom in an extxyz structure.\n"
    "Plot the signed longitudinal effective charge, total effective-charge magnitude, its "
    "transverse magnitude, or all three side by side."
)


def nonnegative_int(value: str) -> int:
    """Parse a non-negative integer for an atom index."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def prepare_args(descr):
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument("-i", "--input", **argv, required=True, help="input extxyz structure")
    parser.add_argument(
        "-k",
        "--keyword",
        default=KEYWORDS["bec"],
        help="per-atom BEC array key (default: %(default)s)",
    )
    atom_selection = parser.add_mutually_exclusive_group(required=True)
    atom_selection.add_argument(
        "-a",
        "--atom",
        type=nonnegative_int,
        help="zero-based atom index selecting one Born effective charge",
    )
    atom_selection.add_argument(
        "--all-atoms",
        action="store_true",
        help="write one row per atom, with longitudinal, total, and transverse columns",
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        default="born-charge-surface.png",
        help="output image path (default: %(default)s)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=91,
        help="number of polar grid points; azimuth uses twice this number (default: %(default)s)",
    )
    parser.add_argument(
        "--unit",
        default="e",
        help="label for the Born-effective-charge unit (default: %(default)s)",
    )
    parser.add_argument(
        "--max-abs",
        type=float,
        default=None,
        help=(
            "shared positive maximum absolute response for the lobe radius and colour scale; "
            "useful when comparing figures"
        ),
    )
    parser.add_argument(
        "--response",
        choices=(*RESPONSE_COMPONENTS, "all"),
        default="longitudinal",
        help="directional response to plot (default: %(default)s)",
    )
    return parser


@cli(prepare_args, description)
def main(args):
    """Read one atomic BEC tensor and save its directional response surface."""
    if args.resolution < 3:
        raise ValueError("--resolution must be at least 3.")
    if args.max_abs is not None and args.max_abs <= 0.0:
        raise ValueError("--max-abs must be positive.")
    atoms = read_input_structures(args.input, input_format="extxyz")
    if args.all_atoms:
        print(f"Extracting all Born effective charges from '{args.keyword}' ... ", end="")
        born_charges = born_charges_from_atoms(atoms, args.keyword)
        print("done")
        labels = [
            f"atom {index}: {symbol}" for index, symbol in enumerate(atoms.get_chemical_symbols())
        ]
        print(f"Writing all-atom directional-response grid to {args.output} ... ", end="")
        responses = plot_born_charge_structure_comparison(
            born_charges,
            args.output,
            labels=labels,
            resolution=args.resolution,
            unit=args.unit,
            max_abs=args.max_abs,
        )
    else:
        print(
            f"Extracting atom {args.atom} Born effective charge from '{args.keyword}' ... ", end=""
        )
        born_charge = born_charge_from_atoms(atoms, args.keyword, args.atom)
        print("done")
        print(f"Writing {args.response} directional-response surface to {args.output} ... ", end="")
        responses = _plot_selected_response(args, born_charge)
    print("done")
    for component, response in responses.items():
        print(
            f"{component.capitalize()} response range: "
            f"[{response.min():.6e}, {response.max():.6e}] {args.unit}."
        )


def _plot_selected_response(args, born_charge):
    """Plot a selected one-atom BEC response and return it by component name."""
    if args.response == "all":
        responses = plot_born_charge_response_comparison(
            born_charge,
            args.output,
            resolution=args.resolution,
            unit=args.unit,
            max_abs=args.max_abs,
        )
    else:
        responses = {
            args.response: plot_born_charge_surface(
                born_charge,
                args.output,
                resolution=args.resolution,
                unit=args.unit,
                max_abs=args.max_abs,
                component=args.response,
            )
        }
    return responses


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
