"""Show the symmetry-allowed components of a tensor."""

# Tested by pytest: tests/test_tensor_symmetries.py

import argparse

import numpy as np

from fd2bec.atomic import AtomicStructure
from fd2bec.cli import cli, count_with_percentage, positive_int, read_input_structures
from fd2bec.cli.parser import add_shared_argument
from fd2bec.displacements import symmetry_inequivalent_displacements
from fd2bec.show import (
    print_independent_components,
    print_numeric_tensor,
    print_reference_structure,
)
from fd2bec.tensor import MAPPING
from fd2bec.tensor_components import (
    affine_parameter_values,
    physical_modes,
    rotate_modes,
    selected_tensor_basis,
    selected_tensor_precision,
    symbolic_affine_components,
    symbolic_components,
)
from fd2bec.tools import tensor_from_atoms

description = "Show the symmetry-allowed components of a tensor."
choices = list(MAPPING.keys())


def prepare_args(descr: str):
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    add_shared_argument(parser, "input_structure")
    parser.add_argument(
        "-n",
        "--name",
        **argv,
        type=str,
        required=True,
        help=f"name of the tensor, choices: {choices}",
        choices=choices,
    )
    parser.add_argument(
        "-k",
        "--keyword",
        **argv,
        type=str,
        default=None,
        help=(
            "optional ASE atoms.info or atoms.arrays key containing the numeric tensor; "
            "values are interpreted as Cartesian"
        ),
    )
    parser.add_argument(
        "--conventional_axes",
        action="store_true",
        help="rotate Cartesian tensor components into spglib's conventional axes",
    )
    parser.add_argument(
        "--basis",
        choices=("cartesian", "fractional"),
        default=None,
        help="component basis; defaults to fractional for positions and Cartesian otherwise",
    )
    parser.add_argument(
        "--precision",
        type=positive_int,
        default=None,
        help=(
            "significant digits used to display numeric tensor components; "
            "defaults to 4 for positions and unrestricted otherwise"
        ),
    )
    add_shared_argument(parser, "symprec")
    return parser


@cli(prepare_args, description)
def main(args: argparse.Namespace):
    reference = read_input_structures(args.input)

    print_reference_structure(reference)
    unit_cell = AtomicStructure.from_ase(reference, symprec=args.symprec)
    basis = selected_tensor_basis(args.name, args.basis)
    precision = selected_tensor_precision(args.name, getattr(args, "precision", None))
    if basis == "fractional" and not unit_cell.pbc:
        raise ValueError("Fractional tensor components require a periodic structure.")
    if basis == "fractional" and args.conventional_axes:
        raise ValueError("--conventional_axes is only supported with Cartesian components.")
    if args.name == "positions" and args.conventional_axes:
        raise ValueError("--conventional_axes is not supported for positions.")

    tensor_class = MAPPING[args.name]
    if args.name == "positions":
        data = unit_cell.frac_pos if basis == "fractional" else unit_cell.positions
        tensor = tensor_class(data=data, basis=basis)
    else:
        tensor = tensor_class.template(len(unit_cell), basis=basis)
    shape = tensor.core_shape()
    print(f"Constructed {tensor.definition['name']} tensor with shape {shape} in {basis} basis.")

    numeric_tensor = None
    tensor_location = None
    keyword = getattr(args, "keyword", None)
    if keyword is not None:
        print(f"Extracting numeric tensor {keyword!r} from the structure ... ", end="")
        numeric_tensor, tensor_location = tensor_from_atoms(
            reference,
            keyword,
            args.name,
            tensor_class,
            tensor,
            basis,
        )
        print("done")

    if args.name == "positions":
        print("\nComputing symmetry-allowed displacement modes ... ", end="")
        _, _, displacement_modes = unit_cell.get_displacement_symmetry_modes(tensor)
        print("done")
        displacement_modes = displacement_modes.reshape((-1, *shape))
        components, pivots = symbolic_affine_components(
            tensor.data,
            displacement_modes,
            axes=tensor.axes,
            fractional=basis == "fractional",
        )
        print(
            "n. symmetry-inequivalent component(s):",
            count_with_percentage(len(pivots), int(np.prod(shape))),
        )
        tensor.print_components(components)
        if numeric_tensor is not None:
            parameter_values = affine_parameter_values(
                tensor.data,
                numeric_tensor.data,
                displacement_modes,
                axes=tensor.axes,
                fractional=basis == "fractional",
            )
            print_numeric_tensor(
                numeric_tensor,
                keyword,
                tensor_location,
                pivots,
                components,
                frame_label="input",
                parameter_values=parameter_values,
                precision=precision,
            )
        return

    print("\nComputing symmetry-allowed components ... ", end="")
    _, _, component_modes = unit_cell.get_symmetry_modes(tensor=tensor)
    print("done")
    modes = physical_modes(component_modes, shape, affine=tensor.has_affine_axis)
    print(
        "n. symmetry-inequivalent component(s):",
        count_with_percentage(len(modes), int(np.prod(shape))),
    )
    finite_difference_displacements, all_finite_difference_displacements = (
        symmetry_inequivalent_displacements(unit_cell, tensor, component_modes=component_modes)
    )
    print(
        "n. finite-difference displacements required:",
        count_with_percentage(
            len(finite_difference_displacements) - 1,
            len(all_finite_difference_displacements) - 1,
        ),
    )

    frame_label = "input"
    if args.conventional_axes:
        coordinate_rotation = np.asarray(
            unit_cell._spglib_dataset.std_rotation_matrix,
            dtype=float,  # pylint: disable=protected-access
        )
        modes = rotate_modes(tensor, modes, coordinate_rotation)
        if numeric_tensor is not None:
            numeric_tensor = numeric_tensor.rotate(coordinate_rotation)
        frame_label = "conventional"
        print("Rotating Cartesian components into conventional crystallographic axes.")
        print("Cartesian coordinate rotation (conventional <- input):")
        print(np.array2string(coordinate_rotation, precision=8, suppress_small=True))

    symbolic, pivots = symbolic_components(
        modes,
        axes=tensor.axes,
        symmetric_axis_pairs=tensor.symmetric_axes,
    )
    print_independent_components(pivots, shape, tensor.axes)
    print(f"\nSymmetry-allowed tensor components ({frame_label} axes):")
    tensor.print_components(symbolic)
    if numeric_tensor is not None:
        print_numeric_tensor(
            numeric_tensor,
            keyword,
            tensor_location,
            pivots,
            symbolic,
            frame_label=frame_label,
            precision=precision,
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
