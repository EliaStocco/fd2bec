"""Symbolic tensor-component construction and axis-aware display helpers."""

from itertools import product

import numpy as np


VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
CARTESIAN_LABELS = ("x", "y", "z")


def parameter_name(index: int) -> str:
    """Return a readable name for an independent tensor parameter."""
    return chr(ord("a") + index) if index < 26 else f"a{index + 1}"


def _independent_columns(basis: np.ndarray, atol: float) -> np.ndarray:
    """Keep a deterministic linearly independent subset of basis columns."""
    selected = []
    rank = 0
    for index in range(basis.shape[1]):
        trial = basis[:, selected + [index]]
        trial_rank = np.linalg.matrix_rank(trial, tol=atol)
        if trial_rank > rank:
            selected.append(index)
            rank = trial_rank
    return basis[:, selected]


def symmetric_pairs(axes, shape):
    """Return explicitly named Cartesian strain-axis pairs eligible for Voigt notation."""
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
        if not (
            left.get("name", "").startswith("strain") and right.get("name", "").startswith("strain")
        ):
            continue
        if shape[index] != 3 or shape[index + 1] != 3:
            continue
        pairs.append((index, index + 1))
        used.update((index, index + 1))
    return pairs


def _symmetric_basis(basis, shape, pairs, atol=1e-10):
    """Apply strain-pair symmetry to a column-oriented component basis."""
    if not pairs:
        return basis
    modes = np.asarray(basis, dtype=float).reshape((*shape, basis.shape[1]))
    for left, right in pairs:
        modes = 0.5 * (modes + np.swapaxes(modes, left, right))
    return _independent_columns(modes.reshape((int(np.prod(shape)), -1)), atol=atol)


def _independent_rows(basis: np.ndarray, atol: float) -> list[int]:
    """Select component rows that can serve as independent parameters."""
    pivots = []
    rank = 0
    for index in range(basis.shape[0]):
        trial = basis[pivots + [index], :]
        trial_rank = np.linalg.matrix_rank(trial, tol=atol)
        if trial_rank > rank:
            pivots.append(index)
            rank = trial_rank
        if rank == basis.shape[1]:
            break
    return pivots


def _format_expression(coefficients, atol: float) -> str:
    """Format one component as a linear combination of parameter names."""
    terms = []
    for index, coefficient in enumerate(coefficients):
        if abs(coefficient) <= atol:
            continue
        name = parameter_name(index)
        term = name if np.isclose(abs(coefficient), 1.0, atol=atol) else f"{abs(coefficient):.4g}{name}"
        if not terms:
            terms.append(f"-{term}" if coefficient < 0 else term)
        else:
            terms.append((" - " if coefficient < 0 else " + ") + term)
    return "".join(terms) if terms else "0"


def symbolic_components(theta_real: np.ndarray, axes=None, atol: float = 1e-10):
    """Express tensor components in terms of independent symmetry-mode parameters.

    Parameters
    ----------
    theta_real
        Real-space symmetry modes with shape ``(number_of_modes, *tensor_shape)``.
        Each first-axis entry is one allowed mode.
    axes
        Optional tensor-axis definitions. When supplied, named strain pairs are
        symmetrized before independent parameters are selected.

    Returns
    -------
    symbolic, pivots
        An object array with shape ``tensor_shape`` and the flattened component
        indices chosen as the independent parameters.
    """
    modes = np.asarray(theta_real, dtype=float)
    if modes.ndim == 0:
        raise ValueError("theta_real must have shape (number_of_modes, *tensor_shape).")
    shape = modes.shape[1:]
    if axes is not None and len(axes) != len(shape):
        raise ValueError("The number of axes must match theta_real tensor dimensions.")

    dimension = int(np.prod(shape, dtype=int)) if shape else 1
    basis = modes.reshape((modes.shape[0], dimension)).T
    if axes is not None:
        basis = _symmetric_basis(basis, shape, symmetric_pairs(axes, shape), atol=atol)
    if basis.shape[1] == 0:
        return np.full(shape, "0", dtype=object), []

    pivots = _independent_rows(basis, atol)
    if len(pivots) != basis.shape[1]:
        basis = _independent_columns(basis, atol)
        pivots = _independent_rows(basis, atol)
    coefficients = basis @ np.linalg.inv(basis[pivots, :])
    symbolic = [_format_expression(row, atol) for row in coefficients]
    return np.asarray(symbolic, dtype=object).reshape(shape), pivots


def voigt_components(symbolic, axes, pairs):
    """Compress each symmetric strain pair into one Voigt axis."""
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
    return CARTESIAN_LABELS[coordinate] if axis.get("type") == "cartesian" else str(coordinate)


def _prefix_label(prefixes, axes):
    """Format one or more equal tensor blocks compactly."""
    if len(prefixes) == 1:
        return ", ".join(
            f"{axis.get('name', f'axis_{index}')}={_coordinate_label(axis, value)}"
            for index, (axis, value) in enumerate(zip(axes, prefixes[0]))
        )
    varying = [index for index in range(len(axes)) if len({prefix[index] for prefix in prefixes}) > 1]
    if len(varying) == 1 and axes[varying[0]].get("type") == "atomic":
        varying_index = varying[0]
        values = ", ".join(str(prefix[varying_index]) for prefix in prefixes)
        fixed = [
            f"{axis.get('name', f'axis_{index}')}={_coordinate_label(axis, prefixes[0][index])}"
            for index, axis in enumerate(axes)
            if index != varying_index
        ]
        return ", ".join([f"{axes[varying_index].get('name', 'atom')}={{{values}}}", *fixed])
    indices = "; ".join(
        "(" + ", ".join(_coordinate_label(axis, value) for axis, value in zip(axes, prefix)) + ")"
        for prefix in prefixes
    )
    return f"indices={{{indices}}}"


def print_components(components, axes, title=None):
    """Print numeric or symbolic tensor components using their axis definitions."""
    components = np.asarray(components, dtype=object)
    if components.ndim != len(axes):
        raise ValueError("The component array dimensions must match the tensor axes.")
    if title:
        print(title)
    if components.ndim == 0:
        print(f"  {components.item()}")
        return
    if components.ndim == 1:
        labels = _axis_labels(axes[0], components.shape[0])
        width = max(1, max(len(str(value)) for value in components.flat))
        print("  [ " + "  ".join(f"{str(value):>{width}}" for value in components) + " ]")
        print("    " + "  ".join(f"{label:>{width}}" for label in labels))
        return

    row_labels = _axis_labels(axes[-2], components.shape[-2])
    column_labels = _axis_labels(axes[-1], components.shape[-1])
    width = max(1, max(len(str(value)) for value in components.flat))
    prefixes = list(product(*(range(size) for size in components.shape[:-2])))
    prefix_axes = axes[: components.ndim - 2]
    groups = {}
    for prefix in prefixes:
        key = tuple(str(value) for value in components[prefix].flat)
        groups.setdefault(key, []).append(prefix)
    prefix_groups = groups.values() if any(axis.get("type") == "atomic" for axis in prefix_axes) else ([prefix] for prefix in prefixes)

    for group in prefix_groups:
        group = list(group)
        if prefix_axes:
            print(f"  [{_prefix_label(group, prefix_axes)}]")
        block = components[group[0]]
        print(" " * 8 + "  ".join(f"{label:>{width}}" for label in column_labels))
        for label, row in zip(row_labels, block):
            print(f"  {label:>3}  " + "  ".join(f"{str(value):>{width}}" for value in row))


def _component_label(index, shape, axes):
    coordinates = np.unravel_index(index, shape)
    labels = []
    for axis, coordinate in zip(axes, coordinates):
        value = CARTESIAN_LABELS[coordinate] if axis.get("type") == "cartesian" else str(coordinate)
        labels.append(f"{axis.get('name', 'axis')}={value}")
    return ", ".join(labels)


def print_independent_components(pivots, shape, axes):
    """Print the tensor entries selected as independent parameters."""
    if not pivots:
        print("No symmetry-inequivalent components.")
        return
    print("\nSymmetry-inequivalent components:")
    for index, pivot in enumerate(pivots):
        print(f"  {parameter_name(index)} = {_component_label(pivot, shape, axes)}")
