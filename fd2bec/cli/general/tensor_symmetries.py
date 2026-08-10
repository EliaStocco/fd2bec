"""Display symmetry-allowed tensor components in readable matrix form."""

import argparse
from itertools import product

import numpy as np

from fd2bec.atomic import AtomicStructure
from fd2bec.cli import cli
from fd2bec.io import read
from fd2bec.show import print_reference_structure
from fd2bec.tensor import MAPPING, Tensor

description = "Show the symmetry-allowed components of a tensor."
VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
CARTESIAN_LABELS = ("x", "y", "z")
choices = list(MAPPING.keys())


def prepare_args(descr):
    parser = argparse.ArgumentParser(description=descr)
    argv = {"metavar": "\b"}
    parser.add_argument(
        "-i",
        "--input",
        **argv,
        type=str,
        required=True,
        help="path to input structure (e.g. supercell.extxyz)",
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
        "--conventional-axes",
        action="store_true",
        help="rotate Cartesian tensor components into spglib's conventional axes",
    )
    return parser


def _parameter_name(index: int) -> str:
    return chr(ord("a") + index) if index < 26 else f"a{index + 1}"


def _independent_columns(basis: np.ndarray, atol: float) -> np.ndarray:
    """Remove dependent columns while preserving their deterministic order."""
    basis = np.asarray(basis, dtype=float)
    selected = []
    rank = 0
    for index in range(basis.shape[1]):
        trial = basis[:, selected + [index]]
        trial_rank = np.linalg.matrix_rank(trial, tol=atol)
        if trial_rank > rank:
            selected.append(index)
            rank = trial_rank
    return basis[:, selected]


def _symmetric_pairs(axes, shape):
    """Find physical strain pairs supported by a tensor definition.

    A Voigt reduction is only offered for explicitly named strain pairs.  This
    avoids silently applying Voigt notation to, for example, a Born-charge
    tensor whose two Cartesian indices have different physical roles.
    """
    pairs = []
    used = set()
    for index in range(len(axes) - 1):
        left, right = axes[index], axes[index + 1]
        if index in used or index + 1 in used:
            continue
        if left.get("type") != "cartesian" or right.get("type") != "cartesian":
            continue
        if left.get("role") != right.get("role"):
            continue
        if not (left.get("name", "").startswith("strain") and right.get("name", "").startswith("strain")):
            continue
        if shape[index] != 3 or shape[index + 1] != 3:
            continue
        pairs.append((index, index + 1))
        used.update((index, index + 1))
    return pairs


def _symmetric_basis(basis, shape, pairs, atol=1e-10):
    if not pairs:
        return basis
    modes = np.asarray(basis, dtype=float).reshape((*shape, basis.shape[1]))
    for left, right in pairs:
        modes = 0.5 * (modes + np.swapaxes(modes, left, right))
    return _independent_columns(modes.reshape((int(np.prod(shape)), -1)), atol)


def symbolic_components(basis: np.ndarray, shape, atol: float = 1e-10):
    """Return symbolic component expressions and independent component indices."""
    basis = np.asarray(basis, dtype=float)
    shape = tuple(shape)
    dimension = int(np.prod(shape)) if shape else 1
    if basis.ndim != 2 or basis.shape[0] != dimension:
        raise ValueError(f"Basis must have shape ({dimension}, number_of_modes).")
    if basis.shape[1] == 0:
        return np.full(shape, "0", dtype=object), []

    pivots = []
    rank = 0
    for index in range(dimension):
        trial = basis[pivots + [index], :]
        trial_rank = np.linalg.matrix_rank(trial, tol=atol)
        if trial_rank > rank:
            pivots.append(index)
            rank = trial_rank
        if rank == basis.shape[1]:
            break

    if rank != basis.shape[1]:
        basis = _independent_columns(basis, atol)
        return symbolic_components(basis, shape, atol=atol)

    coefficients = basis @ np.linalg.inv(basis[pivots, :])
    expressions = []
    for row in coefficients:
        terms = []
        for index, coefficient in enumerate(row):
            if abs(coefficient) <= atol:
                continue
            name = _parameter_name(index)
            if np.isclose(abs(coefficient), 1.0, atol=atol):
                term = name
            else:
                term = f"{abs(coefficient):.4g}{name}"
            if not terms:
                terms.append(f"-{term}" if coefficient < 0 else term)
            else:
                terms.append((" - " if coefficient < 0 else " + ") + term)
        expressions.append("".join(terms) if terms else "0")
    return np.asarray(expressions, dtype=object).reshape(shape), pivots


def voigt_components(symbolic, axes, pairs):
    """Compress symmetric strain pairs into one Voigt axis per pair."""
    shape = symbolic.shape
    pair_axes = {axis for pair in pairs for axis in pair}
    remaining = [axis for axis in range(len(shape)) if axis not in pair_axes]
    output_shape = tuple(shape[index] for index in remaining) + (6,) * len(pairs)
    result = np.empty(output_shape, dtype=object)

    for prefix in product(*(range(shape[index]) for index in remaining)):
        for voigt_index in product(range(6), repeat=len(pairs)):
            full = [None] * len(shape)
            for axis, value in zip(remaining, prefix):
                full[axis] = value
            for (left, right), component in zip(pairs, voigt_index):
                full[left], full[right] = VOIGT_PAIRS[component]
            result[prefix + voigt_index] = symbolic[tuple(full)]
    voigt_axes = [axes[index] for index in remaining] + [
        {"name": "voigt", "type": "voigt"} for _ in pairs
    ]
    return result, voigt_axes


def _axis_labels(axis, size):
    if axis.get("type") == "cartesian":
        return list(CARTESIAN_LABELS[:size])
    if axis.get("type") == "voigt":
        return list(VOIGT_LABELS)
    return [str(index) for index in range(size)]


def _coordinate_label(axis, coordinate):
    if axis.get("type") == "cartesian":
        return CARTESIAN_LABELS[coordinate]
    return str(coordinate)


def _prefix_label(prefixes, axes):
    """Format one or more equal tensor blocks compactly."""
    if len(prefixes) == 1:
        return ", ".join(
            f"{axis.get('name', f'axis_{index}')}={_coordinate_label(axis, value)}"
            for index, (axis, value) in enumerate(zip(axes, prefixes[0]))
        )

    varying = [
        index
        for index in range(len(axes))
        if len({prefix[index] for prefix in prefixes}) > 1
    ]
    if len(varying) == 1 and axes[varying[0]].get("type") == "atomic":
        varying_index = varying[0]
        values = ", ".join(str(prefix[varying_index]) for prefix in prefixes)
        reference = prefixes[0]
        fixed = [
            f"{axis.get('name', f'axis_{index}')}={_coordinate_label(axis, reference[index])}"
            for index, axis in enumerate(axes)
            if index != varying_index
        ]
        atomic = f"{axes[varying_index].get('name', 'atom')}={{{values}}}"
        return ", ".join([atomic, *fixed])

    indices = "; ".join(
        "("
        + ", ".join(_coordinate_label(axis, value) for axis, value in zip(axes, prefix))
        + ")"
        for prefix in prefixes
    )
    return f"indices={{{indices}}}"


def print_symbolic_tensor(symbolic, axes, title=None):
    """Print an object array as labeled vectors or a stack of matrices."""
    symbolic = np.asarray(symbolic, dtype=object)
    if title:
        print(title)
    if symbolic.ndim == 0:
        print(f"  {symbolic.item()}")
        return
    if symbolic.ndim == 1:
        labels = _axis_labels(axes[0], symbolic.shape[0])
        width = max(1, max(len(str(value)) for value in symbolic.flat))
        print("  [ " + "  ".join(f"{str(value):>{width}}" for value in symbolic) + " ]")
        print("    " + "  ".join(f"{label:>{width}}" for label in labels))
        return

    row_labels = _axis_labels(axes[-2], symbolic.shape[-2])
    column_labels = _axis_labels(axes[-1], symbolic.shape[-1])
    width = max(1, max(len(str(value)) for value in symbolic.flat))
    prefix_shape = symbolic.shape[:-2]
    prefixes = list(product(*(range(size) for size in prefix_shape)))
    prefix_axes = axes[: len(prefix_shape)]
    if any(axis.get("type") == "atomic" for axis in prefix_axes):
        groups = {}
        for prefix in prefixes:
            key = tuple(str(value) for value in symbolic[prefix].flat)
            groups.setdefault(key, []).append(prefix)
        prefix_groups = groups.values()
    else:
        prefix_groups = ([prefix] for prefix in prefixes)

    for group in prefix_groups:
        if prefix_shape:
            print(f"  [{_prefix_label(list(group), prefix_axes)}]")
        prefix = next(iter(group))
        header = " " * 8 + "  ".join(f"{label:>{width}}" for label in column_labels)
        print(header)
        for label, row in zip(row_labels, symbolic[prefix]):
            values = "  ".join(f"{str(value):>{width}}" for value in row)
            print(f"  {label:>3}  {values}")


def _component_label(index, shape, axes):
    coordinates = np.unravel_index(index, shape)
    labels = []
    for axis, coordinate in zip(axes, coordinates):
        if axis.get("type") == "cartesian":
            value = CARTESIAN_LABELS[coordinate]
        else:
            value = str(coordinate)
        labels.append(f"{axis.get('name', 'axis')}={value}")
    return ", ".join(labels)


def print_independent_components(pivots, shape, axes):
    """Print the tensor entries selected as independent parameters."""
    if not pivots:
        print("No symmetry-inequivalent components.")
        return
    print("\nSymmetry-inequivalent components:")
    for index, pivot in enumerate(pivots):
        print(f"  {_parameter_name(index)} = {_component_label(pivot, shape, axes)}")


@cli(prepare_args, description)
def main(args):
    print(f"Reading input structure from {args.input} ... ", end="")
    reference = read(args.input, index=0)
    print("done")

    print_reference_structure(reference)
    unit_cell = AtomicStructure.from_ase(reference)
    if args.name not in MAPPING:
        raise ValueError(f"{args.name} not supported.")

    tensor = MAPPING[args.name].template(len(unit_cell))
    shape = tensor.data.shape[-len(tensor.axes) :] if tensor.axes else ()
    print(f"Constructed {tensor.definition['name']} tensor with shape {shape}.")
    print("\nComputing symmetry-allowed components ... ", end="")
    _, _, theta_real = unit_cell.get_symmetrizer(tensor=tensor)
    print("done")

    frame_label = "input"
    if args.conventional_axes:
        coordinate_rotation = np.asarray(
            unit_cell._spglib_dataset.std_rotation_matrix, dtype=float
        )  # pylint: disable=protected-access
        theta_real = (
            np.asarray(
                [
                    Tensor(definition=tensor.definition, data=mode)
                    .rotate(coordinate_rotation)
                    .data
                    for mode in theta_real
                ]
            )
            if len(theta_real)
            else np.asarray(theta_real).copy()
        )
        frame_label = "conventional"
        print("Rotating Cartesian components into conventional crystallographic axes.")
        print("Cartesian coordinate rotation (conventional <- input):")
        print(np.array2string(coordinate_rotation, precision=8, suppress_small=True))

    basis = theta_real.T
    pairs = _symmetric_pairs(tensor.axes, shape)
    display_basis = _symmetric_basis(basis, shape, pairs)
    symbolic, pivots = symbolic_components(display_basis, shape)
    print(f"Found {display_basis.shape[1]} independent component(s).")
    print_independent_components(pivots, shape, tensor.axes)
    print(f"\nSymmetry-allowed tensor components ({frame_label} axes):")
    print_symbolic_tensor(symbolic, tensor.axes)

    if pairs:
        voigt, voigt_axes = voigt_components(symbolic, tensor.axes, pairs)
        print("\nVoigt notation (" + ", ".join(VOIGT_LABELS) + "):")
        print_symbolic_tensor(voigt, voigt_axes)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
