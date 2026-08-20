"""Runtime tensor instances.

The mathematical metadata lives in ordinary dictionaries in
``fd2bec._tensor_definition``.  This module intentionally keeps runtime state
(``data``, ``basis`` and ``cell``) separate from those definitions.
"""

import json
import warnings
from copy import copy, deepcopy
from typing import Tuple

import numpy as np

from fd2bec import ATOL, Basis

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
    ):
        if definition is None:
            definition = self.tensor_definition
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

    def print_components(self, components=None):
        """Print this tensor's numeric or symbolic components with axis labels.

        Explicitly named symmetric strain axes are displayed in Voigt notation.
        """
        if components is None:
            if self.data is None:
                raise ValueError("Tensor components require data.")
            components = self.data
        from .tensor_components import (
            flattened_nuclear_position_matrix,
            print_components,
            symmetric_pairs,
            voigt_components,
        )

        if self.definition["name"] == "force_constants":
            components, axes = flattened_nuclear_position_matrix(components, self.axes)
            print("Flattened nuclear-coordinate matrix (atom-major):")
            print_components(components, axes)
            return
        pairs = symmetric_pairs(self.axes, np.shape(components), declared_pairs=self.symmetric_axes)
        if pairs:
            components, axes = voigt_components(np.asarray(components), self.axes, pairs)
            print("Voigt notation:")
            print_components(components, axes)
        else:
            print_components(components, self.axes)

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
            shape = tuple(natoms if axis["type"] == "atomic" else 3 for axis in axes)
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
    def symmetric_axes(self) -> list[tuple[int, int]]:
        """Pairs of explicit axes related by intrinsic permutation symmetry."""
        return [tuple(pair) for pair in self.definition.get("symmetric_axes", [])]

    @property
    def atomic_axes(self):
        return [index for index, axis in enumerate(self.axes) if axis["type"] == "atomic"]

    @property
    def cartesian_axes(self):
        return [index for index, axis in enumerate(self.axes) if axis["type"] == "cartesian"]

    @property
    def has_affine_axis(self) -> bool:
        """Whether any explicit axis transforms affinely."""
        return any(axis.get("affine", False) for axis in self.axes)

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
        cartesian_axes = [axis for axis in self.axes if axis["type"] == "cartesian"]
        return (
            sum(axis["variance"] == "contravariant" for axis in cartesian_axes),
            sum(axis["variance"] == "covariant" for axis in cartesian_axes),
        )

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
    def template(cls, natoms: int = None, *, basis: Basis = "cartesian") -> "Tensor":
        definition = getattr(cls, "tensor_definition", None)
        if definition is None:
            raise ValueError("Tensor.template requires a class tensor definition.")
        axes = definition["axes"]
        if any(axis["type"] == "atomic" for axis in axes) and natoms is None:
            raise ValueError("natoms is required for an atomic tensor template.")
        shape = tuple(natoms if axis["type"] == "atomic" else 3 for axis in axes)
        return cls(data=np.full(shape, np.nan), basis=basis)

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

    def flatten(self) -> np.ndarray:
        """Flatten Cartesian components while retaining atomic and batch axes."""
        if self.data is None:
            raise ValueError("Tensor flattening requires data.")
        number_axes = len(self.axes)
        if not number_axes:
            return self.data
        batch_shape = self.data.shape[:-number_axes]
        core_shape = self.data.shape[-number_axes:]
        atomic_shape = core_shape[: len(self.atomic_axes)]
        return self.data.reshape(batch_shape + atomic_shape + (-1,))

    def flatten_full(self) -> np.ndarray:
        """Flatten all explicit dimensions while retaining leading batch dimensions."""
        if self.data is None:
            raise ValueError("Tensor flattening requires data.")
        number_axes = len(self.axes)
        if not number_axes:
            return self.data
        batch_shape = self.data.shape[:-number_axes]
        return self.data.reshape(batch_shape + (-1,))

    def apply_intrinsic_symmetry(self, vector: np.ndarray) -> np.ndarray:
        """Project flattened components onto declared symmetric-axis subspaces."""
        vector = np.asarray(vector)
        shape = self.core_shape()
        dimension = int(np.prod(shape, dtype=int)) if shape else 1
        if vector.shape[-1:] != (dimension,):
            raise ValueError(
                f"Intrinsic symmetry for {self.definition['name']!r} requires a trailing "
                f"dimension of {dimension}, got shape {vector.shape}."
            )
        components = vector.reshape(vector.shape[:-1] + shape)
        batch_ndim = vector.ndim - 1
        for left, right in self.symmetric_axes:
            components = 0.5 * (
                components + np.swapaxes(components, batch_ndim + left, batch_ndim + right)
            )
        return components.reshape(vector.shape)

    def intrinsic_symmetry_projection(self) -> np.ndarray:
        """Return the projector enforcing declared symmetric-axis relationships."""
        dimension = self.flatten_full().shape[-1]
        return self.apply_intrinsic_symmetry(np.eye(dimension)).T

    def symmetrize(self, projection: np.ndarray, *, atol: float = ATOL) -> "Tensor":
        """Apply a totally symmetric projection to this tensor.

        ``projection`` must act on :meth:`flatten_full` components. It can be
        a dense matrix or a matrix-free operator exposing ``shape`` and
        matrix multiplication. For an ordinary tensor with ``D`` components
        it therefore has shape ``(D, D)``. For a tensor with an affine axis,
        such as positions, it must be the homogeneous ``(D + 1, D + 1)``
        projection acting on ``[components, 1]``.

        For tensors associated with an :class:`~fd2bec.atomic.AtomicStructure`,
        compute this matrix with::

            projection = structure.get_symmetry_projection(tensor)

        That method averages the tensor representation of the structure's
        symmetry operations.  The supplied matrix must be a projection: its
        result is required to be invariant under a second application.
        """
        vector = self.flatten_full()
        if not self.axes:
            # A scalar has one implicit component; any data dimensions are
            # leading batch dimensions.
            vector = np.expand_dims(vector, axis=-1)
        dimension = vector.shape[-1] + int(self.has_affine_axis)
        if projection.shape != (dimension, dimension):
            raise ValueError(
                f"Projection for {self.definition['name']!r} must have shape "
                f"({dimension}, {dimension}), got {projection.shape}."
            )
        matrix_free = hasattr(projection, "matvec")
        if not matrix_free:
            projection = np.asarray(projection)

        if self.has_affine_axis:
            from .mathematics import append_one

            vector = append_one(vector, axis=-1)

        if matrix_free:
            flat_vectors = vector.reshape((-1, dimension))
            symmetrized = np.stack([projection @ item for item in flat_vectors]).reshape(
                vector.shape
            )
        else:
            symmetrized = np.einsum("ij,...j->...i", projection, vector, optimize=True)
        if self.has_affine_axis:
            if not np.allclose(symmetrized[..., -1], 1.0, atol=atol):
                raise ValueError("Affine projection must preserve the homogeneous coordinate.")
            result = symmetrized[..., :-1]
            check_vector = append_one(result, axis=-1)
        else:
            result = symmetrized
            check_vector = result

        if matrix_free:
            flat_vectors = check_vector.reshape((-1, dimension))
            repeated = np.stack([projection @ item for item in flat_vectors]).reshape(
                check_vector.shape
            )
        else:
            repeated = np.einsum("ij,...j->...i", projection, check_vector, optimize=True)
        if not np.allclose(repeated, check_vector, atol=atol):
            raise ValueError("Projection result is not invariant under repeated application.")
        return self.copy_with(data=result.reshape(self.shape))

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


def contract(R: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...j->...i", R, x)
