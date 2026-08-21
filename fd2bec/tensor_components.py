"""Symbolic tensor-component construction and axis-aware display helpers."""

from itertools import product

import numpy as np

VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
CARTESIAN_LABELS = ("x", "y", "z")


def parameter_name(index: int) -> str:
    """Return a readable name for an independent tensor parameter."""
    letter = chr(ord("a") + index % 26)
    suffix = index // 26
    return letter if suffix == 0 else f"{letter}{suffix}"


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


def symmetric_pairs(axes, shape, declared_pairs=None):
    """Return Cartesian symmetric-axis pairs eligible for Voigt notation.

    Explicit definition metadata takes precedence. The axis-name convention is
    retained as a compatibility fallback for callers that only provide axes.
    """
    if declared_pairs is not None:
        return [
            (left, right)
            for left, right in declared_pairs
            if shape[left] == 3 and shape[right] == 3
        ]
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
        term = (
            name
            if np.isclose(abs(coefficient), 1.0, atol=atol)
            else f"{abs(coefficient):.4g}{name}"
        )
        if not terms:
            terms.append(f"-{term}" if coefficient < 0 else term)
        else:
            terms.append((" - " if coefficient < 0 else " + ") + term)
    return "".join(terms) if terms else "0"


def symbolic_components(
    component_modes: np.ndarray,
    axes=None,
    atol: float = 1e-10,
    symmetric_axis_pairs=None,
):
    """Express tensor components in terms of independent symmetry-mode parameters.

    Parameters
    ----------
    component_modes
        Real-space symmetry modes with shape ``(number_of_modes, *tensor_shape)``.
        Each first-axis entry is one allowed mode.
    axes
        Optional tensor-axis definitions. When supplied, named strain pairs are
        symmetrized before independent parameters are selected.
    symmetric_axis_pairs
        Optional explicitly declared symmetric-axis pairs. When supplied, these
        take precedence over axis-name inference.

    Returns
    -------
    symbolic, pivots
        An object array with shape ``tensor_shape`` and the flattened component
        indices chosen as the independent parameters.
    """
    modes = np.asarray(component_modes, dtype=float)
    if modes.ndim == 0:
        raise ValueError("component_modes must have shape (number_of_modes, *tensor_shape).")
    shape = modes.shape[1:]
    if axes is not None and len(axes) != len(shape):
        raise ValueError("The number of axes must match component_modes tensor dimensions.")

    coefficients, pivots = _symbolic_coefficients(
        modes,
        axes=axes,
        atol=atol,
        symmetric_axis_pairs=symmetric_axis_pairs,
    )
    if coefficients.size == 0:
        return np.full(shape, "0", dtype=object), []

    symbolic = [_format_expression(row, atol) for row in coefficients]
    return np.asarray(symbolic, dtype=object).reshape(shape), pivots


def _symbolic_coefficients(
    component_modes: np.ndarray,
    axes=None,
    atol: float = 1e-10,
    symmetric_axis_pairs=None,
):
    """Return component coefficients and pivots for a real-space mode basis."""
    modes = np.asarray(component_modes, dtype=float)
    shape = modes.shape[1:]
    dimension = int(np.prod(shape, dtype=int)) if shape else 1
    basis = modes.reshape((modes.shape[0], dimension)).T
    if axes is not None:
        pairs = symmetric_pairs(axes, shape, declared_pairs=symmetric_axis_pairs)
        basis = _symmetric_basis(basis, shape, pairs, atol=atol)
    if basis.shape[1] == 0:
        return np.empty((0, 0)), []

    pivots = _independent_rows(basis, atol)
    if len(pivots) != basis.shape[1]:
        basis = _independent_columns(basis, atol)
        pivots = _independent_rows(basis, atol)
    return basis @ np.linalg.inv(basis[pivots, :]), pivots


def _format_affine_expression(constant: float, coefficients: np.ndarray, atol: float) -> str:
    """Format a constant reference value plus a symbolic displacement."""
    displacement = _format_expression(coefficients, atol)
    if displacement == "0":
        return str(constant)
    if np.isclose(constant, 0.0, atol=atol):
        return displacement
    if displacement.startswith("-"):
        return f"{constant} - {displacement[1:]}"
    return f"{constant} + {displacement}"


def symbolic_affine_components(
    reference_components: np.ndarray,
    component_modes: np.ndarray,
    axes=None,
    *,
    fractional: bool = False,
    atol: float = 1e-8,
):
    """Show affine reference components together with symbolic linear modes.

    Components with no allowed linear displacement retain their reference
    value. In a fractional basis, variable components are written as the
    nearest ``0`` or ``1/2`` reference coordinate plus their displacement;
    the nearest periodic image is selected modulo one. This makes small
    distortions of a high-symmetry structure legible (for example,
    ``0.982 -> -c``). Cartesian components retain the previous, purely
    symbolic presentation.
    """
    reference_components = np.asarray(reference_components, dtype=float)
    modes = np.asarray(component_modes, dtype=float)
    if modes.shape[1:] != reference_components.shape:
        raise ValueError("Reference components and component modes must have the same shape.")

    coefficients, pivots = _symbolic_coefficients(modes, axes=axes, atol=atol)
    if coefficients.size == 0:
        symbolic = np.full(reference_components.shape, "0", dtype=object)
    else:
        symbolic = np.asarray(
            [_format_expression(row, atol) for row in coefficients], dtype=object
        ).reshape(reference_components.shape)

    result = symbolic.copy()
    fixed_components = symbolic == "0"
    for index in zip(*np.where(fixed_components)):
        value = reference_components[index]
        if fractional:
            value %= 1.0
            if np.isclose(value, 0.0, atol=atol) or np.isclose(value, 1.0, atol=atol):
                value = 0.0
            elif np.isclose(value, 0.5, atol=atol):
                value = 0.5
        result[index] = str(value)

    if not fractional or not len(pivots):
        return result, pivots

    values = reference_components.reshape(-1) % 1.0
    # Fractional coordinates differ by an integer lattice vector.  The two
    # closest high-symmetry representatives in this setting are therefore
    # 0 and 1/2, chosen independently modulo one.
    origins = (np.round(2.0 * values) / 2.0) % 1.0
    displacements = (values - origins + 0.5) % 1.0 - 0.5

    # Eigenvectors have arbitrary signs. Choose each parameter sign so that
    # the selected pivot's displacement has the same sign as the input
    # structure; all symmetry-related entries follow automatically.
    oriented_coefficients = coefficients.copy()
    for parameter, pivot in enumerate(pivots):
        if displacements[pivot] < -atol:
            oriented_coefficients[:, parameter] *= -1.0

    for index, row in enumerate(oriented_coefficients):
        if not fixed_components.reshape(-1)[index]:
            result.reshape(-1)[index] = _format_affine_expression(origins[index], row, atol)
    return result, pivots


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


def flattened_nuclear_position_matrix(components, axes):
    """Return force constants as an atom-major ``3N x 3N`` matrix.

    Force constants are stored with the two atomic axes first and the two
    Cartesian axes last.  Interleaving each atom with its ``x``, ``y``, and
    ``z`` components gives the conventional matrix indexing used for nuclear
    displacements.
    """
    components = np.asarray(components, dtype=object)
    axis_types = [axis.get("type") for axis in axes]
    if components.ndim != 4 or axis_types != ["atomic", "atomic", "cartesian", "cartesian"]:
        raise ValueError("Force constants must have atomic, atomic, Cartesian, Cartesian axes.")
    natoms, other_natoms, first_size, second_size = components.shape
    if natoms != other_natoms or (first_size, second_size) != (3, 3):
        raise ValueError("Force constants must have shape (N, N, 3, 3).")

    labels = [f"{atom}{coordinate}" for atom in range(natoms) for coordinate in CARTESIAN_LABELS]
    matrix = components.transpose(0, 2, 1, 3).reshape((3 * natoms, 3 * natoms))
    matrix_axes = [
        {"name": "nuclear coordinate", "labels": labels},
        {"name": "nuclear coordinate", "labels": labels},
    ]
    return matrix, matrix_axes


def _axis_labels(axis, size):
    labels = axis.get("labels")
    if labels is not None:
        if len(labels) != size:
            raise ValueError("Explicit axis labels must match the component axis length.")
        return list(labels)
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
    varying = [
        index for index in range(len(axes)) if len({prefix[index] for prefix in prefixes}) > 1
    ]
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
        width = max(
            1,
            *(len(str(value)) for value in components.flat),
            *(len(label) for label in labels),
        )
        # Keep one-dimensional tensors consistent with the tabular display
        # used below: identify columns before showing their values.  This is
        # especially useful for a stress tensor represented in Voigt notation.
        print("    " + "  ".join(f"{label:>{width}}" for label in labels))
        print("  [ " + "  ".join(f"{str(value):>{width}}" for value in components) + " ]")
        return

    row_labels = _axis_labels(axes[-2], components.shape[-2])
    column_labels = _axis_labels(axes[-1], components.shape[-1])
    column_width = max(
        1,
        *(len(str(value)) for value in components.flat),
        *(len(label) for label in column_labels),
    )
    row_width = max(len(label) for label in row_labels)
    row_prefix = "  " + " " * row_width + "  "
    prefixes = list(product(*(range(size) for size in components.shape[:-2])))
    prefix_axes = axes[: components.ndim - 2]
    groups = {}
    for prefix in prefixes:
        key = tuple(str(value) for value in components[prefix].flat)
        groups.setdefault(key, []).append(prefix)
    prefix_groups = (
        groups.values()
        if any(axis.get("type") == "atomic" for axis in prefix_axes)
        else ([prefix] for prefix in prefixes)
    )

    for group in prefix_groups:
        group = list(group)
        if prefix_axes:
            print(f"  [{_prefix_label(group, prefix_axes)}]")
        block = components[group[0]]
        print(row_prefix + "  ".join(f"{label:>{column_width}}" for label in column_labels))
        for label, row in zip(row_labels, block):
            print(
                f"  {label:>{row_width}}  "
                + "  ".join(f"{str(value):>{column_width}}" for value in row)
            )


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
        # print("No symmetry-inequivalent components.")
        return
    print("\nSymmetry-inequivalent components:")
    for index, pivot in enumerate(pivots):
        print(f"  {parameter_name(index)} = {_component_label(pivot, shape, axes)}")
