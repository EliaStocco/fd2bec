"""Plot the signed longitudinal directional response of a piezoelectric tensor."""

import argparse

from fd2bec.cli import cli
from fd2bec.piezoelectric_plot import (
    RESPONSE_COMPONENTS,
    plot_piezoelectric_response_comparison,
    plot_piezoelectric_surface,
    read_piezoelectric_voigt,
)

description = (
    "Create directional piezoelectric response surfaces from a 3 x 6 Voigt matrix in the "
    "order xx, yy, zz, yz, xz, xy.\nPlot the signed longitudinal response, the total induced "
    "polarization magnitude, its transverse magnitude, or all three side by side."
)


def prepare_args(descr):
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i", "--input", **argv, required=True, help="3 x 6 piezoelectric text or CSV file"
    )
    parser.add_argument(
        "-o",
        "--output",
        **argv,
        default="piezoelectric-surface.png",
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
        default="e/Angstrom^2",
        help="label for the input coefficient unit (default: %(default)s)",
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
    """Read the tensor and save its directional response surface."""
    if args.resolution < 3:
        raise ValueError("--resolution must be at least 3.")
    if args.max_abs is not None and args.max_abs <= 0.0:
        raise ValueError("--max-abs must be positive.")
    print(f"Reading 3 x 6 piezoelectric tensor from {args.input} ... ", end="")
    tensor = read_piezoelectric_voigt(args.input)
    print("done")
    print(f"Writing {args.response} directional-response surface to {args.output} ... ", end="")
    if args.response == "all":
        responses = plot_piezoelectric_response_comparison(
            tensor, args.output, resolution=args.resolution, unit=args.unit, max_abs=args.max_abs
        )
    else:
        responses = {
            args.response: plot_piezoelectric_surface(
                tensor,
                args.output,
                resolution=args.resolution,
                unit=args.unit,
                max_abs=args.max_abs,
                component=args.response,
            )
        }
    print("done")
    for component, response in responses.items():
        print(
            f"{component.capitalize()} response range: "
            f"[{response.min():.6e}, {response.max():.6e}] {args.unit}."
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
