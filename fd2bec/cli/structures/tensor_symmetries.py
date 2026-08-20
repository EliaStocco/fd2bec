"""Show the symmetry-allowed components of a tensor."""

# Tested by pytest: tests/test_tensor_symmetries.py

import argparse
from pathlib import Path

import numpy as np

from fd2bec import ATOL, Basis
from fd2bec._tensor_base import Tensor
from fd2bec.atomic import AtomicStructure
from fd2bec.cli import cli
from fd2bec.displacements import symmetry_inequivalent_displacements
from fd2bec.io import read
from fd2bec.show import print_reference_structure
from fd2bec.tensor import MAPPING
from fd2bec.tensor_components import (
    add_affine_reference,
    common_symbolic_components,
    parameter_name,
    print_independent_components,
    symbolic_affine_components,
    symbolic_components,
)

description = "Show the symmetry-allowed components of a tensor."
choices = list(MAPPING.keys())


def prepare_args(descr: str):
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        nargs="+",
        required=True,
        help="path(s) to input structure(s) (e.g. supercell.extxyz)",
    )
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
    return parser


def _rotated_modes(tensor: Tensor, modes: np.ndarray, coordinate_rotation: np.ndarray):
    """Rotate each tensor mode into a new Cartesian coordinate frame."""
    if not len(modes):
        return modes.copy()
    return np.asarray(
        [tensor.copy_with(data=mode).rotate(coordinate_rotation).data for mode in modes]
    )


def _selected_basis(name: str, requested_basis: Basis | None):
    return requested_basis or ("fractional" if name == "positions" else "cartesian")


def _physical_modes(
    component_modes: np.ndarray,
    shape: tuple[int, ...],
    *,
    affine: bool,
    atol: float = ATOL,
):
    """Reshape modes and discard homogeneous-only affine coordinates."""
    modes = component_modes.reshape((component_modes.shape[0], *shape))
    if not len(modes):
        return modes
    norms = np.linalg.norm(modes.reshape((len(modes), -1)), axis=1)
    if affine:
        return modes[norms > atol]
    if not np.allclose(norms, 1, atol=atol):
        raise ValueError("Symmetry modes must be normalized.")
    return modes


def _common_tensor_symmetries(args: argparse.Namespace, input_paths: list[str]):
    """Print several tensor spaces in one shared component parameterization."""
    if args.conventional_axes:
        raise ValueError(
            "--conventional_axes cannot define one shared frame for multiple structures; "
            "align the input structures in the desired common Cartesian frame instead."
        )

    basis = _selected_basis(args.name, args.basis)
    tensors = []
    modes_collection = []
    reference_symbols = None
    reference_shape = None

    for input_path in input_paths:
        print(f"Reading input structure from {input_path} ... ", end="")
        reference = read(input_path, index=0)
        print("done")
        unit_cell = AtomicStructure.from_ase(reference)
        symbols = tuple(unit_cell.symbols)
        if reference_symbols is None:
            reference_symbols = symbols
        elif symbols != reference_symbols:
            raise ValueError(
                "All structures must contain the same atoms in the same order. "
                f"Atom ordering differs in '{input_path}'."
            )
        if basis == "fractional" and not unit_cell.pbc:
            raise ValueError("Fractional tensor components require periodic structures.")

        tensor_class = MAPPING[args.name]
        if args.name == "positions":
            data = unit_cell.frac_pos if basis == "fractional" else unit_cell.positions
            tensor = tensor_class(data=data, basis=basis)
            _, _, component_modes = unit_cell.get_displacement_symmetry_modes(tensor)
            modes = component_modes.reshape((-1, *tensor.core_shape()))
        else:
            tensor = tensor_class.template(len(unit_cell), basis=basis)
            _, _, component_modes = unit_cell.get_symmetry_modes(tensor=tensor)
            modes = _physical_modes(
                component_modes,
                tensor.core_shape(),
                affine=tensor.has_affine_axis,
            )

        if reference_shape is None:
            reference_shape = tensor.core_shape()
        elif tensor.core_shape() != reference_shape:
            raise ValueError("All structures must produce tensors with the same shape.")

        tensors.append(tensor)
        modes_collection.append(modes)

    print(
        f"\nComputing a common parameterization for {len(input_paths)} "
        f"{tensors[0].definition['name']} tensors in {basis} basis ... ",
        end="",
    )
    symbolic_collection, common_pivots, parameter_indices = common_symbolic_components(
        modes_collection,
        axes=tensors[0].axes,
        symmetric_axis_pairs=tensors[0].symmetric_axes,
    )
    print("done")
    print(
        "The common space is the span of the symmetry-allowed spaces; "
        "symbols denote the same tensor entries in every structure."
    )
    print("The atom ordering and input coordinate axes are assumed to correspond.")
    print("n. common parameter(s): ", len(common_pivots))
    print_independent_components(common_pivots, reference_shape, tensors[0].axes)

    for input_path, tensor, modes, symbolic, indices in zip(
        input_paths,
        tensors,
        modes_collection,
        symbolic_collection,
        parameter_indices,
    ):
        if args.name == "positions":
            symbolic = add_affine_reference(
                tensor.data,
                symbolic,
                fractional=basis == "fractional",
            )
        names = ", ".join(parameter_name(index) for index in indices) or "none"
        print(f"\nStructure: {Path(input_path).stem} ({input_path})")
        print(f"n. symmetry-inequivalent component(s): {len(modes)}")
        print(f"independent common symbol(s): {names}")
        tensor.print_components(symbolic)


@cli(prepare_args, description)
def main(args: argparse.Namespace):
    input_paths = [args.input] if isinstance(args.input, str) else list(args.input)
    if len(input_paths) > 1:
        return _common_tensor_symmetries(args, input_paths)
    args.input = input_paths[0]

    print(f"Reading input structure from {args.input} ... ", end="")
    reference = read(args.input, index=0)
    print("done")

    print_reference_structure(reference)
    unit_cell = AtomicStructure.from_ase(reference)
    basis = _selected_basis(args.name, args.basis)
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
        print("n. symmetry-inequivalent component(s): ", len(pivots))
        tensor.print_components(components)
        return

    print("\nComputing symmetry-allowed components ... ", end="")
    _, _, component_modes = unit_cell.get_symmetry_modes(tensor=tensor)
    print("done")
    modes = _physical_modes(component_modes, shape, affine=tensor.has_affine_axis)
    print("n. symmetry-inequivalent component(s): ", len(modes))
    finite_difference_displacements, _ = symmetry_inequivalent_displacements(
        unit_cell, tensor, component_modes=component_modes
    )
    print("n. finite-difference displacements required: ", len(finite_difference_displacements) - 1)

    frame_label = "input"
    if args.conventional_axes:
        coordinate_rotation = np.asarray(
            unit_cell._spglib_dataset.std_rotation_matrix,
            dtype=float,  # pylint: disable=protected-access
        )
        modes = _rotated_modes(tensor, modes, coordinate_rotation)
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


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
