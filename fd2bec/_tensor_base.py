"""Runtime tensor instances.

The mathematical metadata lives in ordinary dictionaries in
``fd2bec._tensor_definition``.  This module intentionally keeps runtime state
(``data``, ``basis`` and ``cell``) separate from those definitions.
"""

import json
import warnings
from copy import copy, deepcopy
from typing import List, Tuple

import numpy as np

from fd2bec import Basis

from ._tensor_definition import validate_definition

SCHEMA_VERSION = 1


class Tensor:
    """A tensor value attached to a JSON-compatible tensor definition."""

    tensor_definition = None

    def __init__(
        self,
        definition=None,
        data=None,
        cell=None,
        basis: Basis = "cartesian",
        *,
        axes=None,
        is_atomic=None,
        is_affine=None,
    ):
        if isinstance(definition, list) and axes is None:
            axes = definition
            definition = None
        if definition is None:
            definition = self.tensor_definition
        if definition is None and axes is not None:
            definition = _legacy_definition(axes, is_atomic=is_atomic, is_affine=is_affine)
        if definition is None:
            raise ValueError("Tensor requires a definition.")
        self.definition = validate_definition(definition)
        if basis not in ("cartesian", "fractional"):
            raise ValueError("basis must be 'cartesian' or 'fractional'.")
        self.basis = basis
        self.cell = _cell_array(cell)
        self.data = None if data is None else np.asarray(data)
        self._validate_data()

    def __repr__(self):
        return (
            f"{type(self).__name__}(definition={self.definition['name']!r}, "
            f"shape={self.shape}, basis={self.basis!r})"
        )

    def print_components(self, components=None, *, title=None, voigt=False):
        """Print this tensor's numeric or symbolic components with axis labels.

        Set ``voigt=True`` to also print any explicitly named symmetric strain
        axes in Voigt notation.
        """
        if components is None:
            if self.data is None:
                raise ValueError("Tensor components require data.")
            components = self.data
        from .tensor_components import (
            VOIGT_LABELS,
            print_components,
            symmetric_pairs,
            voigt_components,
        )

        print_components(components, self.axes, title=title)
        if voigt:
            pairs = symmetric_pairs(self.axes, np.shape(components))
            if pairs:
                components, axes = voigt_components(np.asarray(components), self.axes, pairs)
                print("\nVoigt notation (" + ", ".join(VOIGT_LABELS) + "):")
                print_components(components, axes)

    def _validate_data(self):
        if self.data is None:
            return
        axes = self.axes
        if not axes:
            return
        if self.data.ndim < len(axes):
            if self._reshape_if_compatible():
                return
            raise ValueError(
                f"Data for {self.definition['name']!r} needs at least {len(axes)} "
                f"dimensions for its explicit axes, got shape {self.data.shape}."
            )
        sizes = self.data.shape[-len(axes) :]
        atomic_sizes = [size for axis, size in zip(axes, sizes) if axis["type"] == "atomic"]
        if atomic_sizes and len(set(atomic_sizes)) != 1:
            if self._reshape_if_compatible():
                return
            raise ValueError("All atomic dimensions must have the same natoms size.")
        for axis, size in zip(axes, sizes):
            expected = atomic_sizes[0] if axis["type"] == "atomic" else 3
            if size != expected:
                if self._reshape_if_compatible():
                    return
                kind = "natoms" if axis["type"] == "atomic" else "3"
                raise ValueError(
                    f"Axis {axis['name']!r} is {axis['type']} and requires size {kind}; got {size}."
                )

    def _reshape_if_compatible(self) -> bool:
        """Reshape data to its explicit shape when its size makes that possible."""
        axes = self.axes
        cartesian_axes = sum(axis["type"] == "cartesian" for axis in axes)
        atomic_axes = sum(axis["type"] == "atomic" for axis in axes)
        cartesian_size = 3**cartesian_axes

        if self.data.size % cartesian_size:
            return False
        atomic_size = self.data.size // cartesian_size
        if atomic_axes:
            natoms = round(atomic_size ** (1 / atomic_axes))
            if natoms**atomic_axes != atomic_size:
                return False
            shape = tuple(
                natoms if axis["type"] == "atomic" else 3 for axis in axes
            )
        elif atomic_size == 1:
            shape = (3,) * cartesian_axes
        else:
            return False

        original_shape = self.data.shape
        self.data = self.data.reshape(shape)
        warnings.warn(
            f"Data for {self.definition['name']!r} with shape {original_shape} was reshaped "
            f"to its explicit tensor shape {shape}.",
            UserWarning,
            stacklevel=3,
        )
        return True

    @property
    def axes(self):
        """Explicit axis definitions in trailing storage order."""
        return self.definition["axes"]

    @property
    def variances(self) -> List[bool]:
        """Legacy compact view of Cartesian variances (False means contra)."""
        return [
            axis["variance"] == "covariant" for axis in self.axes if axis["type"] == "cartesian"
        ]

    @property
    def is_atomic(self) -> bool:
        """Deprecated compatibility view; use axis ``type`` instead."""
        return any(axis["type"] == "atomic" for axis in self.axes)

    @property
    def is_affine(self) -> bool:
        """Deprecated compatibility view; use axis ``affine`` instead."""
        return any(axis.get("affine", False) for axis in self.axes)

    @property
    def atomic_axes(self):
        return [index for index, axis in enumerate(self.axes) if axis["type"] == "atomic"]

    @property
    def cartesian_axes(self):
        return [index for index, axis in enumerate(self.axes) if axis["type"] == "cartesian"]

    @property
    def input_axes(self):
        return [index for index, axis in enumerate(self.axes) if axis.get("role") == "input"]

    @property
    def output_axes(self):
        return [index for index, axis in enumerate(self.axes) if axis.get("role") == "output"]

    @property
    def input_shape(self):
        if self.data is None:
            return tuple(3 for index in self.input_axes if self.axes[index]["type"] == "cartesian")
        core_shape = self.data.shape[-len(self.axes) :] if self.axes else ()
        return tuple(core_shape[index] for index in self.input_axes)

    @property
    def output_shape(self):
        if self.data is None:
            return tuple(3 for index in self.output_axes if self.axes[index]["type"] == "cartesian")
        core_shape = self.data.shape[-len(self.axes) :] if self.axes else ()
        return tuple(core_shape[index] for index in self.output_axes)

    @property
    def shape(self):
        return None if self.data is None else self.data.shape

    @property
    def rank(self) -> Tuple[int, int]:
        return axes_to_pq(self.variances)

    def core_shape(self, natoms=None) -> tuple:
        """Return the shape represented by the explicit axes."""
        if any(axis["type"] == "atomic" for axis in self.axes):
            if natoms is None:
                if self.data is not None:
                    natoms = self.data.shape[-len(self.axes) + self.atomic_axes[0]]
                else:
                    raise ValueError("natoms is required for a tensor with atomic axes.")
        return tuple(natoms if axis["type"] == "atomic" else 3 for axis in self.axes)

    @classmethod
    def template(cls, natoms: int = None) -> "Tensor":
        definition = getattr(cls, "tensor_definition", None)
        if definition is None:
            raise ValueError("Tensor.template requires a class tensor definition.")
        axes = definition["axes"]
        if any(axis["type"] == "atomic" for axis in axes) and natoms is None:
            raise ValueError("natoms is required for an atomic tensor template.")
        shape = tuple(natoms if axis["type"] == "atomic" else 3 for axis in axes)
        return cls(data=np.full(shape, np.nan))

    def _replace(self, **changes):
        result = copy(self)
        for key, value in changes.items():
            if key == "definition":
                value = validate_definition(value)
            elif key == "cell":
                value = _cell_array(value)
            elif key == "data" and value is not None:
                value = np.asarray(value)
            setattr(result, key, value)
        result._validate_data()
        return result

    copy_with = _replace

    def _apply_axis(self, arr: np.ndarray, axis: int, matrix: np.ndarray) -> np.ndarray:
        arr = np.moveaxis(arr, axis, -1)
        arr = np.einsum("...j,ij->...i", arr, matrix, optimize=True)
        return np.moveaxis(arr, -1, axis)

    def _transform(self, matrices: list[np.ndarray]) -> "Tensor":
        if self.data is None:
            raise ValueError("Tensor transformations require data.")
        if len(matrices) != len(self.axes):
            raise ValueError("One transformation matrix is required per explicit axis.")
        result = self.data
        number_axes = len(self.axes)
        for index, matrix in enumerate(matrices):
            result = self._apply_axis(result, -(number_axes - index), matrix)
        return self._replace(data=result)

    def _build_operator(self, matrices: list[np.ndarray]) -> np.ndarray:
        """Build the row-major Kronecker operator for storage-order axes."""
        if len(matrices) != len(self.axes):
            raise ValueError("One transformation matrix is required per explicit axis.")
        if not matrices:
            return np.ones((1, 1))
        result = np.asarray(matrices[0])
        for matrix in matrices[1:]:
            result = np.kron(result, matrix)
        return result

    def _apply_matrices(self, matrices, *, method="recursive", basis=None) -> "Tensor":
        if self.data is None:
            raise ValueError("Tensor transformations require data.")
        if len(matrices) != len(self.axes):
            raise ValueError(f"Expected {len(self.axes)} matrices, got {len(matrices)}")
        if method == "recursive":
            result = self._transform(matrices)
        elif method == "flat":
            core_ndim = len(self.axes)
            core_shape = self.data.shape[-core_ndim:] if core_ndim else ()
            batch_shape = self.data.shape[:-core_ndim] if core_ndim else self.data.shape
            operator = self._build_operator(matrices)
            flat = self.data.reshape(batch_shape + (-1,))
            transformed = np.einsum("...j,ij->...i", flat, operator, optimize=True)
            result = self._replace(data=transformed.reshape(batch_shape + core_shape))
        else:
            raise ValueError(f"method must be 'recursive' or 'flat', got {method!r}")
        return result._replace(basis=self.basis if basis is None else basis)

    def rotate(self, R: np.ndarray, method: str = "recursive") -> "Tensor":
        R = np.asarray(R)
        if R.shape != (3, 3):
            raise ValueError("Rotation must be (3,3)")
        matrices = [
            np.eye(self.core_shape()[index]) if axis["type"] == "atomic" else R
            for index, axis in enumerate(self.axes)
        ]
        return self._apply_matrices(matrices, method=method, basis=self.basis)

    def transform(self, A, Ainv=None, to="fractional", method: str = "recursive") -> "Tensor":
        if self.basis == to:
            return self
        A = np.asarray(A)
        if A.shape != (3, 3):
            raise ValueError("The basis matrix must be (3,3).")
        if Ainv is None:
            Ainv = np.linalg.inv(A)
        Ainv = np.asarray(Ainv)
        matrices = []
        for axis in self.axes:
            if axis["type"] == "atomic":
                matrices.append(np.eye(self.core_shape()[len(matrices)]))
                continue
            covariant = axis["variance"] == "covariant"
            if self.basis == "cartesian" and to == "fractional":
                matrices.append(A if covariant else Ainv.T)
            elif self.basis == "fractional" and to == "cartesian":
                matrices.append(Ainv if covariant else A.T)
            else:
                raise ValueError(f"Unsupported transformation: {self.basis} -> {to}")
        return self._apply_matrices(matrices, method=method, basis=to)

    def to(self, basis: Basis, **kwargs):
        if self.cell is None:
            raise ValueError("A cell is required for Cartesian/fractional conversion.")
        return self.transform(self.cell, None, basis, **kwargs)

    def rotation_operator(self, R: np.ndarray) -> np.ndarray:
        """Return the operator on Cartesian components, preserving atomic axes."""
        R = np.asarray(R)
        if R.shape != (3, 3):
            raise ValueError("Rotation must be (3,3)")
        matrices = [R for axis in self.axes if axis["type"] == "cartesian"]
        if not matrices:
            return np.ones((1, 1))
        result = matrices[0]
        for matrix in matrices[1:]:
            result = np.kron(result, matrix)
        return result

    def full_operator(self, matrices: list[np.ndarray]) -> np.ndarray:
        """Build an operator including atomic dimensions (used by symmetry)."""
        return self._build_operator(matrices)

    def __mul__(self, scalar):
        if not np.isscalar(scalar):
            raise TypeError("Tensor multiplication only supports scalar values.")
        return self._replace(data=None if self.data is None else self.data * scalar)

    __rmul__ = __mul__

    def flatten(self, full: bool = False) -> np.ndarray:
        """Flatten explicit dimensions while retaining leading batch dimensions.

        The compatibility form (``full=False``) retains all atomic dimensions,
        which keeps legacy ``(natoms, components)`` BEC/force workflows working.
        ``full=True`` flattens every explicit dimension for symmetry operators.
        """
        if self.data is None:
            raise ValueError("Tensor flattening requires data.")
        number_axes = len(self.axes)
        if not number_axes:
            return self.data
        batch_shape = self.data.shape[:-number_axes]
        core_shape = self.data.shape[-number_axes:]
        atomic_count = len(self.atomic_axes)
        if full:
            return self.data.reshape(batch_shape + (-1,))
        if atomic_count:
            preserved = core_shape[:atomic_count]
            cartesian_size = int(np.prod(core_shape[atomic_count:]))
            return self.data.reshape(batch_shape + preserved + (cartesian_size,))
        return self.data.reshape(batch_shape + (-1,))

    def contract(self, R: np.ndarray) -> "Tensor":
        arr = self.flatten()
        arr = contract(R, arr).reshape(self.shape)
        return self._replace(data=arr)

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "definition": deepcopy(self.definition),
            "basis": self.basis,
            "cell": None if self.cell is None else self.cell.tolist(),
            "data": None if self.data is None else self.data.tolist(),
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, payload: dict) -> "Tensor":
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported or missing tensor schema_version {SCHEMA_VERSION}.")
        return cls(
            definition=payload["definition"],
            data=None if payload.get("data") is None else np.asarray(payload["data"]),
            basis=payload.get("basis", "cartesian"),
            cell=payload.get("cell"),
        )

    @classmethod
    def from_json(cls, value: str) -> "Tensor":
        return cls.from_dict(json.loads(value))

    def __array__(self, dtype=None):
        if dtype:
            return self.data.astype(dtype)
        return self.data


def _cell_array(cell):
    if cell is None:
        return None
    if hasattr(cell, "array"):
        cell = cell.array
    cell = np.asarray(cell)
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape (3,3).")
    return cell


def _legacy_definition(axes, *, is_atomic=None, is_affine=None):
    """Convert the old Boolean-axis constructor to a temporary definition."""
    if not all(isinstance(axis, bool) for axis in axes):
        raise ValueError("Legacy axes must be a list of booleans.")
    result = []
    if is_atomic:
        result.append({"name": "atom", "type": "atomic"})
    for index, covariant in enumerate(axes):
        axis = {
            "name": f"axis_{index}",
            "type": "cartesian",
            "variance": "covariant" if covariant else "contravariant",
        }
        if is_affine:
            axis["affine"] = True
        result.append(axis)
    return {"name": "legacy_tensor", "axes": result}


class SpecialDict(dict):
    """Compatibility helper retained for older public constructors."""

    def __setitem__(self, key, value):
        if key in self and self[key] != value:
            raise ValueError(
                f"Key '{key}' already exists with a different value:\n"
                f"  existing: {self[key]}\n  new:      {value}"
            )
        super().__setitem__(key, value)


def axes_to_pq(axes: list[bool]) -> tuple[int, int]:
    if axes and isinstance(axes[0], dict):
        axes = [axis["variance"] == "covariant" for axis in axes if axis["type"] == "cartesian"]
    if not all(isinstance(axis, bool) for axis in axes):
        raise ValueError("axes must be a list of booleans or axis dictionaries")
    return axes.count(False), axes.count(True)


def pq_to_axes(p: int, q: int) -> list[bool]:
    if p < 0 or q < 0:
        raise ValueError("p and q must be non-negative integers")
    return [False] * p + [True] * q


def contract(R: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...j->...i", R, x)
