# pylint: disable=invalid-name
import warnings
from dataclasses import InitVar, dataclass, field, replace
from functools import cached_property
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import spglib
from ase import Atoms
from ase.cell import Cell
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from scipy.sparse.linalg import LinearOperator

from fd2bec import ATOL, SYMPREC, Basis
from fd2bec.mathematics import affine2homogeneous, append_one, find_mapping, wrap
from fd2bec.tensor import Displacement, Position, Rotation, Tensor, Translation
from fd2bec.tools import numbers2symbols, symbols2numbers

# Above this number of explicit tensor components, storing one dense operator
# per symmetry operation is needlessly expensive.  Symmetry modes are then
# constructed by applying the group average directly to vectors.
MATRIX_FREE_SYMMETRY_DIMENSION = 1024
MAX_DENSE_SYMMETRY_OPERATOR_BYTES = 1024**3


@dataclass
class AtomicStructure:
    """
    Immutable representation of an atomic structure.

    Attributes
    ----------
    symbols : tuple[str, ...]
        Chemical symbols of atoms in order.
    cell : ase.Cell
        lattice vectors
    positions : np.ndarray
        cartesian atomic positions with shape (N, 3).

    Notes
    -----
    - The class is fully immutable:
        - `symbols` is stored as a tuple
        - NumPy arrays are copied and marked read-only
    - Derived properties (`species`, `pos_dict`) are cached for efficiency.
    """

    symbols: List[str]
    cell: Cell = field(default=None)
    pbc: bool = field(default=None)
    frac_pos: np.ndarray = field(default=None)
    positions: np.ndarray = field(default=None)
    symprec: Dict[str, Any] = field(default=SYMPREC)

    # init-only (NOT stored)
    check: InitVar[bool] = True

    def __post_init__(self, check):
        """
        Normalize and validate structural data.

        Ensures `cell` is an ase.Cell, derives missing coordinates
        (`positions` or `frac_pos`) from the other, and checks their
        consistency if both are provided. Arrays are marked read-only
        to enforce immutability.

        If `check` is True, compares against the conventional cell and
        emits a warning if the structure is not in standard form.
        """
        if self.cell is None:
            assert self.pbc is None or not self.pbc, (
                "Please provide 'cell' for periodic structures."
            )
            self.pbc = False
        else:
            assert self.pbc is None or self.pbc, (
                "'pbc' has to be None or True if you specify a 'cell'."
            )
            self.pbc = True

        if not self.pbc:
            self.cell = np.full((3, 3), np.nan)

        if isinstance(self.cell, np.ndarray):
            self.cell = Cell(self.cell)

        if self.frac_pos is None:
            assert isinstance(self.positions, np.ndarray)
            self.frac_pos = self.cell.scaled_positions(self.positions)

        elif self.positions is None:
            assert isinstance(self.frac_pos, np.ndarray)
            self.positions = self.cell.cartesian_positions(self.frac_pos)

        elif self.frac_pos is None and self.positions is None:
            raise ValueError("You need to provide either 'positions' or 'frac_pos'.")

        elif self.frac_pos is not None and self.positions is not None:
            test = self.cell.cartesian_positions(self.frac_pos)
            if not np.allclose(self.positions, test, atol=ATOL):
                raise ValueError(
                    "The provided 'positions' and 'frac_pos' are not coherent with each other."
                )
        else:
            raise ValueError("This is a coding error")

        self.positions.setflags(write=False)
        self.frac_pos.setflags(write=False)
        self.cell.array.setflags(write=False)

        # if check:
        #     if self.pbc:
        #         conventional = self.conventional
        #         if not self.is_equal_to(conventional):
        #             warn("You are not using a conventional unit cell.")

    def clone(self, frac_pos=None, positions=None, **kwargs) -> "AtomicStructure":
        """
        Clone a 'AtomicStructure' by replacing the provided attributes.
        In this way one can choose either to initialize via 'positions' and '__post_init__' will retrieve 'frac_pos'
        or viceversa, and let  '__post_init__' retrieve 'pbc'.
        """
        if "pbc" not in kwargs:
            kwargs["pbc"] = self.pbc
        if not kwargs["pbc"]:
            assert "cell" not in kwargs
            kwargs["cell"] = None
        return replace(self, frac_pos=frac_pos, positions=positions, **kwargs)

    @classmethod
    def from_ase(
        cls, atoms: Atoms, keyword: str = "positions", symprec: float = SYMPREC
    ) -> "AtomicStructure":
        """
        Create an AtomicStructure from an ASE Atoms object.

        Parameters
        ----------
        atoms : ase.Atoms
            Input atomic structure.

        Returns
        -------
        AtomicStructure
            Immutable representation of the structure.
        """
        positions = atoms.arrays[keyword]
        pbc = atoms.get_pbc()
        if not (np.all(pbc) or not np.any(pbc)):
            raise ValueError("mixed PBC: inconsistent boundary conditions")
        pbc = bool(np.all(pbc))
        return cls(
            symbols=atoms.get_chemical_symbols(),
            positions=positions.copy(),
            cell=atoms.cell.copy() if pbc else None,
            pbc=pbc,
            symprec=symprec,
        )

    # @validate_types
    def to(self, basis: Basis, tensor: Tensor, **kwargs) -> Tensor:
        cell = tensor.cell if tensor.cell is not None else self.cell.array
        return tensor.transform(cell, None, basis, **kwargs)

    def __eq__(self, other: "AtomicStructure") -> bool:
        """Check if two AtomicStructure instances are equal."""
        return self.is_equal_to(other)

    def is_equal_to(
        self, other: "AtomicStructure", atol=ATOL, reason=False
    ) -> Union[bool, Tuple[bool, str]]:
        """
        Compare two structures for equality.

        Equality is defined as:
        - same species
        - same cell parameters (within tolerance)
        - same fractional positions per species (order-independent)

        Parameters
        ----------
        other : AtomicStructure

        Returns
        -------
        bool
        """
        if not isinstance(other, AtomicStructure):
            return NotImplemented

        if self.pbc != other.pbc:
            return (False, "different pbc") if reason else False

        if self.species != other.species:
            return (False, "different species") if reason else False

        if not np.allclose(self.cell, other.cell, equal_nan=True):
            return (False, "different cell") if reason else False

        try:
            mapping = self._get_atoms_mapping(
                other, atol=atol
            )  # will raise ValueError if not equal
        except ValueError:
            return (False, "mapping") if reason else False
        if self.pbc:
            diff = wrap(self.frac_pos[mapping] - other.frac_pos)
        else:
            diff = self.positions[mapping] - other.positions
        if not np.allclose(diff, 0, atol=atol):
            return False, "large difference" if reason else False

        return (True, "") if reason else True

    def __len__(self):
        """
        Number of atoms in the structure.

        Returns
        -------
        int
        """
        return len(self.symbols)

    @cached_property
    def space_group(self) -> int:
        """Space group number of the structure."""
        if self.pbc:
            return self._spglib_dataset.number
        else:
            return -1

    @cached_property
    def species(self) -> set[str]:
        """
        Unique chemical species present in the structure.

        Returns
        -------
        set[str]
        """
        return set(self.symbols)

    @cached_property
    def frac_pos_dict(self) -> dict[str, np.ndarray]:
        """
        Fractional positions grouped by chemical species.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping: species -> (n_atoms, 3) array of fractional positions.

        Notes
        -----
        - Returned arrays are copies and read-only.
        - Safe to use without risking mutation of internal state.
        """
        symbols_arr = np.asarray(self.symbols)
        result = {}

        for s in self.species:
            arr = self.frac_pos[symbols_arr == s].copy()
            arr.setflags(write=False)
            result[s] = arr

        return result

    @cached_property
    def pos_dict(self) -> dict[str, np.ndarray]:
        """
        Cartesian positions grouped by chemical species.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping: species -> (n_atoms, 3) array of fractional positions.

        Notes
        -----
        - Returned arrays are copies and read-only.
        - Safe to use without risking mutation of internal state.
        """
        symbols_arr = np.asarray(self.symbols)
        result = {}

        for s in self.species:
            arr = self.positions[symbols_arr == s].copy()
            arr.setflags(write=False)
            result[s] = arr

        return result

    def to_json(self) -> dict:
        """
        Convert the structure to a YAML-serializable dictionary.

        Returns
        -------
        dict
            Dictionary representation suitable for YAML dumping.
        """
        return {
            "symbols": list(self.symbols),
            "cell": self.cell.array.tolist(),
            "positions": self.positions.tolist(),
        }

    @cached_property
    def atomic_numbers(self):
        return symbols2numbers(self.symbols)

    @cached_property
    def _spglib_cell(self):
        assert np.allclose(self.positions, self.frac_pos @ self.cell.array)
        return (
            self.cell.array,
            self.frac_pos,
            self.atomic_numbers,
        )

    @cached_property
    def __pymatge_molecule(self) -> Molecule:
        return Molecule(self.symbols, self.positions)

    @cached_property
    def _spglib_dataset(self) -> spglib.SpglibDataset:
        """
        Convert the structure to a spglib-compatible cell representation.

        Returns
        -------
        tuple
            (cell, scaled_positions, atomic_numbers) for spglib.
        """
        return spglib.get_symmetry_dataset(self._spglib_cell, symprec=self.symprec)

    @cached_property
    def standardized(self):
        return spglib.standardize_cell(self._spglib_cell, symprec=self.symprec)

    @cached_property
    def conventional(self) -> "AtomicStructure":
        cell, frac_pos, numbers = self.standardized
        return type(self)(
            cell=Cell(cell), frac_pos=frac_pos, symbols=numbers2symbols(numbers), check=False
        )

    def _test_symmetry(self, basis: Basis = "cartesian", atol=ATOL):
        """
        Check symmetry: apply x' = R x + t (fractional coords) for all operations.
        """
        R, T = self.get_symmetry_operations(basis=basis)
        for _, (r, t) in enumerate(zip(R, T)):
            if basis == "fractional":
                new_pos = self.frac_pos @ r.T + t
                new_structure = self.clone(frac_pos=new_pos)
            else:
                new_pos = self.positions @ r.T + t
                new_structure = self.clone(positions=new_pos)

            if not self.is_equal_to(new_structure, atol=atol):
                ok, reason = self.is_equal_to(new_structure, atol=atol, reason=True)
                raise ValueError(
                    f"Symmetry operation does not preserve the structure. Reason: {reason}"
                )
            if self.pbc:
                if self.space_group != new_structure.space_group:
                    raise ValueError("Symmetry operation does not preserve the space group")
            mapping = self._get_atoms_mapping(new_structure)
            if self.pbc:
                diff = wrap(self.frac_pos[mapping] - new_structure.frac_pos)
            else:
                diff = self.positions[mapping] - new_structure.positions
            if not np.allclose(diff, 0, atol=atol):
                raise ValueError("Symmetry operation does not preserve atomic positions")

    def _get_atoms_mapping(self, other: "AtomicStructure", atol=ATOL, cell=None) -> np.ndarray:
        """
        Build an atom index mapping from `other` to `self`, computed per species
        using the provided `find_mapping` function.

        Returns
        -------
        mapping : np.ndarray
            mapping[i] = index in self corresponding to atom i in other
        """
        mapping = np.zeros(len(self), dtype=int)
        assert self.pbc == other.pbc, "Different pbc."

        for s in self.species:
            idx_self = np.where(np.array(self.symbols) == s)[0]
            idx_other = np.where(np.array(other.symbols) == s)[0]

            if not self.pbc:
                a = self.pos_dict[s]
                b = other.pos_dict[s]
            else:
                a = self.frac_pos_dict[s]
                b = other.frac_pos_dict[s]

            local_map, ok, dists = find_mapping(a, b, atol=atol, pbc=self.pbc, cell=cell)
            if not ok:
                raise ValueError(
                    f"Mapping failed for species {s}."
                    + f" Total distance: {np.linalg.norm(dists)}."
                    + f" All distances {dists.tolist()}"
                )

            mapping[idx_other] = idx_self[local_map]

        # try:
        if not np.all(np.sort(mapping) == np.arange(len(self))):
            raise ValueError(f"Invalid mapping: {mapping.tolist()}")
        # except:
        #     pass

        return mapping

    def get_atoms_mapping(self, other: "AtomicStructure", atol=ATOL) -> np.ndarray:
        """Return the correspondence from ``other`` atom indices to this structure.

        ``mapping[i]`` is the index of the atom in ``self`` that corresponds to
        atom ``i`` in ``other``.  Atoms are matched only to atoms of the same
        chemical species.  For periodic structures, positions are compared in
        fractional coordinates using the minimum-image convention; for
        molecules, Cartesian coordinates are used.

        Parameters
        ----------
        other
            Structure whose atom indices should be mapped onto this structure.
        atol
            Maximum allowed positional difference.  Its unit is fractional
            coordinates for periodic structures and Angstrom for molecules.

        Raises
        ------
        ValueError
            If the structures cannot be matched one-to-one within ``atol``.
        """
        if len(self) != len(other):
            raise ValueError(
                f"Structures have different numbers of atoms: {len(self)} and {len(other)}."
            )
        if self.pbc != other.pbc:
            raise ValueError("Cannot match periodic and non-periodic structures.")
        if sorted(self.symbols) != sorted(other.symbols):
            raise ValueError("Structures have different chemical compositions.")
        return self._get_atoms_mapping(other, atol=atol)

    def reordered_like(self, reference: "AtomicStructure", atol=ATOL) -> "AtomicStructure":
        """Return this structure with atom order matched to ``reference``.

        The returned structure contains this structure's symbols and positions,
        but its atom at index ``i`` corresponds to atom ``i`` in ``reference``.
        """
        # mapping[candidate_index] = reference_index; invert it to select the
        # candidate atom that belongs at every reference index.
        order = np.argsort(reference.get_atoms_mapping(self, atol=atol))
        return self.clone(
            symbols=[self.symbols[index] for index in order],
            positions=self.positions[order].copy(),
        )

    def __get_all_atoms_mapping(self) -> np.ndarray:
        R, T = self.get_symmetry_operations(basis="cartesian")
        mapping = [None] * len(R)
        for n, (r, t) in enumerate(zip(R, T)):
            new_pos = self.positions @ r.T + t
            new_structure = self.clone(positions=new_pos)
            mapping[n] = self._get_atoms_mapping(
                new_structure,
                atol=self.symprec,
                cell=self.cell.array if self.pbc else None,
            )
        return np.asarray(mapping)

    def get_tensor_symmetry_operations(self, tensor: Tensor):
        """
        Construct flattened symmetry operations acting on a vector representation.

        Each space-group operation (R, t) together with its atom mapping is converted into an
        affine transformation on a flattened state vector:

            x_flat -> R_flat @ x_flat + T_flat

        where x_flat stacks all components of the input representation. This construction
        combines rotation, translation, and permutation induced by symmetry.

        Atomic tensors include the permutation induced by each symmetry
        operation. Affine tensors additionally include a translation and are
        anchored at the supplied tensor value. A fully-NaN affine template is
        anchored at zero so it can still be used to count symmetry modes.

        Parameters
        ----------
        tensor : Tensor
            Tensor whose rank, basis, atomicity, and affine character determine
            the representation.

        Returns
        -------
        R_flat : np.ndarray
            Shape (Nops, dim, dim) linear symmetry operators in flattened form.

        T_flat : np.ndarray
            Shape (Nops, dim) translation vectors in flattened form.

        """
        affine = tensor.has_affine_axis
        axes = tensor.axes
        has_atomic = any(axis["type"] == "atomic" for axis in axes)

        x_flat = tensor.flatten_full()
        natoms = len(self)
        expected_shape = tensor.core_shape(natoms=natoms)
        expected_dim = int(np.prod(expected_shape, dtype=int)) if expected_shape else 1
        if x_flat.shape != (expected_dim,):
            raise ValueError(
                f"Expected one tensor with explicit shape {expected_shape}, "
                f"got flattened shape {x_flat.shape}."
            )

        if affine:
            if np.all(np.isnan(x_flat)):
                affine_point = np.zeros_like(x_flat)
            elif np.all(np.isfinite(x_flat)):
                affine_point = x_flat
            else:
                raise ValueError("Affine tensor data must be either finite or fully NaN.")

        R, T = self.get_symmetry_operations(basis=tensor.basis)
        if has_atomic:
            mappings = self.__get_all_atoms_mapping()
        else:
            mappings = [None] * len(R)

        nops = len(R)
        if R.shape != (nops, 3, 3):
            raise ValueError(f"Expected rotations with shape ({nops}, 3, 3), got {R.shape}.")
        if T.shape != (nops, 3):
            raise ValueError(f"Expected translations with shape ({nops}, 3), got {T.shape}.")
        if has_atomic and len(mappings) != nops:
            raise ValueError("Number of atomic mappings does not match symmetry operations.")

        atom_indices = np.arange(natoms)
        dense_operator_bytes = nops * expected_dim**2 * np.dtype(float).itemsize
        if dense_operator_bytes > MAX_DENSE_SYMMETRY_OPERATOR_BYTES:
            raise MemoryError(
                "Dense tensor symmetry operations would require "
                f"{dense_operator_bytes / 1024**3:.1f} GiB. "
                "Use get_symmetry_projection() or get_symmetry_modes(), which use a "
                "matrix-free projection for large tensors."
            )
        R_flat = np.zeros((nops, expected_dim, expected_dim))
        T_flat = np.zeros((nops, expected_dim))

        for n, (r, t, m) in enumerate(zip(R, T, mappings)):
            matrices = []
            if has_atomic:
                permutation = np.zeros((natoms, natoms))
                permutation[atom_indices, m] = 1

            for axis in axes:
                if axis["type"] == "atomic":
                    matrices.append(permutation)
                else:
                    matrices.append(r.T)

            R_flat[n] = tensor.full_operator(matrices)

            if affine:
                # An affine Cartesian axis receives the symmetry translation.
                # Broadcasting over all other explicit dimensions preserves the
                # ordinary position and global-vector cases and avoids the old
                # accidental overwrite of the computed translation.
                shift = np.zeros(expected_shape, dtype=float)
                for axis_index, axis in enumerate(axes):
                    if axis.get("affine", False) and axis["type"] == "cartesian":
                        reshape = [1] * len(axes)
                        reshape[axis_index] = 3
                        shift += np.broadcast_to(np.asarray(t).reshape(reshape), expected_shape)
                T_flat[n] = shift.reshape(-1)

        if affine:
            # Correct lattice-image translations so every affine operation
            # fixes the supplied tensor value exactly. The same expression is
            # valid for atomic positions and for a global affine dipole.
            transformed = R_flat @ affine_point + T_flat
            T_flat += affine_point - transformed

        return R_flat, T_flat

    def _combine_intrinsic_symmetry_projection(self, tensor: Tensor, projection):
        """Intersect a structural symmetry projector with intrinsic tensor symmetry."""
        if not tensor.symmetric_axes:
            return projection

        component_dimension = tensor.flatten_full().size

        def apply_intrinsic(vector):
            vector = np.asarray(vector, dtype=float)
            if tensor.has_affine_axis:
                components = tensor.apply_intrinsic_symmetry(vector[:-1])
                return np.concatenate((components, vector[-1:]))
            return tensor.apply_intrinsic_symmetry(vector)

        if hasattr(projection, "matvec"):
            return LinearOperator(
                projection.shape,
                matvec=lambda vector: apply_intrinsic(projection @ vector),
                dtype=float,
            )

        intrinsic = tensor.intrinsic_symmetry_projection()
        if tensor.has_affine_axis:
            homogeneous = np.eye(component_dimension + 1)
            homogeneous[:component_dimension, :component_dimension] = intrinsic
            intrinsic = homogeneous
        return intrinsic @ projection

    def _matrix_free_symmetry_projection(self, tensor: Tensor) -> tuple[LinearOperator, int]:
        """Return a matrix-free group projection and the dimension of its image.

        This representation applies every space-group operation directly to a
        tensor-shaped vector.  It avoids materializing an array with shape
        ``(number_of_operations, dimension, dimension)``.
        """
        if tensor.has_affine_axis:
            raise ValueError("Matrix-free symmetry projections do not support affine tensors.")

        shape = tensor.core_shape(natoms=len(self))
        dimension = int(np.prod(shape, dtype=int)) if shape else 1
        rotations, _ = self.get_symmetry_operations(basis=tensor.basis)
        has_atomic_axis = any(axis["type"] == "atomic" for axis in tensor.axes)
        mappings = self.__get_all_atoms_mapping() if has_atomic_axis else [None] * len(rotations)
        atom_indices = np.arange(len(self))

        def apply_operation(
            vector: np.ndarray, rotation: np.ndarray, mapping: np.ndarray
        ) -> np.ndarray:
            components = np.asarray(vector, dtype=float).reshape(shape)
            for axis_index, axis in enumerate(tensor.axes):
                if axis["type"] == "atomic":
                    components = np.take(components, mapping, axis=axis_index)
                    continue
                components = np.moveaxis(components, axis_index, -1)
                components = np.einsum("...j,ij->...i", components, rotation.T, optimize=True)
                components = np.moveaxis(components, -1, axis_index)
            return components.reshape(-1)

        def matvec(vector: np.ndarray) -> np.ndarray:
            vector = np.asarray(vector, dtype=float)
            if vector.shape != (dimension,):
                raise ValueError(
                    f"Expected a vector with shape ({dimension},), got {vector.shape}."
                )
            result = np.zeros(dimension, dtype=float)
            for rotation, mapping in zip(rotations, mappings):
                result += apply_operation(vector, rotation, mapping)
            return result / len(rotations)

        # The trace of the group-average projection is the number of
        # symmetry-invariant components.  For a Kronecker-product operation,
        # its trace is the product of the traces on the individual axes.
        characters = []
        for rotation, mapping in zip(rotations, mappings):
            character = 1.0
            for axis in tensor.axes:
                if axis["type"] == "atomic":
                    character *= np.count_nonzero(atom_indices == mapping)
                else:
                    character *= np.trace(rotation)
            characters.append(character)
        image_dimension = float(np.mean(characters))
        number_of_modes = int(np.rint(image_dimension))
        if not np.isclose(image_dimension, number_of_modes, atol=ATOL):
            raise ValueError("The symmetry projection must have an integer trace.")

        projection = LinearOperator((dimension, dimension), matvec=matvec, dtype=float)
        projection = self._combine_intrinsic_symmetry_projection(tensor, projection)
        return projection, number_of_modes

    def _matrix_free_symmetry_modes(
        self, tensor: Tensor, tensor_components: np.ndarray, atol: float
    ) -> tuple[LinearOperator, np.ndarray, np.ndarray]:
        """Construct invariant modes without materializing dense group operators."""
        projection, maximum_number_of_modes = self._matrix_free_symmetry_projection(tensor)
        dimension = tensor_components.size
        if maximum_number_of_modes:
            # The projected random vectors span the invariant subspace with
            # probability one. A fixed seed makes the displayed basis stable.
            random_vectors = np.random.default_rng(0).standard_normal(
                (dimension, maximum_number_of_modes)
            )
            projected_vectors = np.column_stack(
                [projection @ random_vectors[:, index] for index in range(maximum_number_of_modes)]
            )
            mode_basis, singular_values, _ = np.linalg.svd(projected_vectors, full_matrices=False)
            number_of_modes = np.count_nonzero(singular_values > atol)
            if not tensor.symmetric_axes and number_of_modes != maximum_number_of_modes:
                raise ValueError("Could not construct a complete symmetry-mode basis.")
            mode_basis = mode_basis[:, :number_of_modes]
        else:
            number_of_modes = 0
            mode_basis = np.empty((dimension, 0))

        if np.any(np.isnan(tensor_components)):
            mode_coefficients = np.full(number_of_modes, np.nan)
        else:
            mode_coefficients = np.linalg.lstsq(mode_basis, tensor_components, rcond=None)[0]
        return projection, mode_coefficients, mode_basis.T

    @cached_property
    def affine_symmetry_operations(self):
        """
        Flattened affine symmetry operations for the atomic coordinates.
        """
        tensor = Position(data=self.frac_pos, basis="fractional")
        return self.get_tensor_symmetry_operations(tensor=tensor)

    @cached_property
    def homogeneous_symmetry_operations(self):
        """
        Flattened homogeneous symmetry operations for the atomic coordinates.
        """
        R_flat, T_flat = self.affine_symmetry_operations
        H = affine2homogeneous(R_flat, T_flat)
        return H

    def get_symmetry_operations(self, basis: Basis = "cartesian") -> Tuple[np.ndarray, np.ndarray]:
        """Return space/point group symmetry operatios (R, t) such that x' = R x + t."""

        if basis == "fractional":
            assert self.pbc, "'fractional' is supported only for periodic structures."
            return self._spglib_dataset.rotations, self._spglib_dataset.translations

        elif basis == "cartesian":
            if self.pbc:
                R = Rotation(
                    data=self._spglib_dataset.rotations, basis="fractional", cell=self.cell
                ).to("cartesian")
                T = Translation(
                    data=self._spglib_dataset.translations, basis="fractional", cell=self.cell
                ).to("cartesian")

                return R.data, T.data
            else:
                pga = PointGroupAnalyzer(
                    self.__pymatge_molecule,
                    tolerance=self.symprec,
                    eigen_tolerance=self.symprec,
                    matrix_tolerance=self.symprec,
                )

                point_group_operations = pga.get_symmetry_operations()
                rotations = np.asarray(
                    [operation.rotation_matrix for operation in point_group_operations]
                )
                translations = np.asarray(
                    [operation.translation_vector for operation in point_group_operations]
                )

                O = np.mean(self.positions, axis=0)
                effective_translations = (
                    translations
                    + O[None, :]
                    - np.asarray([O @ rotation.T for rotation in rotations])
                )

                return rotations, effective_translations

        else:
            raise NotImplementedError

    def get_symmetry_projection(self, tensor: Tensor) -> np.ndarray:
        """Return the projector onto structurally and intrinsically allowed components."""
        if (
            not tensor.has_affine_axis
            and tensor.flatten_full().size > MATRIX_FREE_SYMMETRY_DIMENSION
        ):
            return self._matrix_free_symmetry_projection(tensor)[0]
        operations, translations = self.get_tensor_symmetry_operations(tensor=tensor)
        if tensor.has_affine_axis:
            operations = affine2homogeneous(operations, translations)
        projection = np.mean(operations, axis=0)
        return self._combine_intrinsic_symmetry_projection(tensor, projection)

    def symmetrize(self, tensor: Tensor) -> Tensor:
        """Symmetrize ``tensor`` with this structure's symmetry projection."""
        projection = self.get_symmetry_projection(tensor=tensor)
        return tensor.symmetrize(projection)

    def get_displacement_symmetry_modes(
        self, positions: Position, atol: float = ATOL
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return linear displacement modes about the supplied affine positions."""
        if positions.data is None:
            raise ValueError("Displacement modes require reference position data.")
        displacement = Displacement(
            data=np.zeros_like(positions.data), basis=positions.basis, cell=positions.cell
        )
        return self.get_symmetry_modes(displacement, atol=atol)

    def get_symmetry_modes(
        self,
        tensor: Tensor,
        atol: float = ATOL,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the symmetry projection, mode coefficients, and flattened invariant modes.

        Positions are affine reference values, so this returns the linear
        displacement modes about the supplied positions instead.
        """
        if isinstance(tensor, Position):
            return self.get_displacement_symmetry_modes(tensor, atol=atol)

        tensor_components = tensor.flatten_full()
        if not tensor.has_affine_axis and tensor_components.size > MATRIX_FREE_SYMMETRY_DIMENSION:
            return self._matrix_free_symmetry_modes(tensor, tensor_components, atol)

        # ------------------------
        # Projection construction
        # ------------------------
        projection = self.get_symmetry_projection(tensor=tensor)

        # ------------------------
        # Vector construction
        # ------------------------
        if tensor.has_affine_axis:
            tensor_components = append_one(tensor_components)

        # ------------------------
        # Eigen-decomposition
        # ------------------------

        # Use an elementwise tolerance: the Frobenius norm accumulates harmless
        # round-off over every matrix entry and therefore depends on the tensor
        # dimension (and, for atomic tensors, the number of atoms).
        projection_asymmetry = projection - projection.T
        largest_asymmetry = np.max(np.abs(projection_asymmetry))
        symmetric_projection = largest_asymmetry < atol
        if not symmetric_projection and tensor.basis == "cartesian" and not tensor.has_affine_axis:
            warnings.warn(
                f"\n\tThe symmetry projection for {tensor} is not symmetric."
                "\tSymmetrizing it before constructing symmetry modes.",
                RuntimeWarning,
                stacklevel=2,
            )
            print(f"Largest component of projection - projection.T: {largest_asymmetry}")
            projection = (projection + projection.T) / 2
            symmetric_projection = True

        if symmetric_projection:
            # A symmetric projection has a stable orthonormal eigendecomposition.
            eigenvalues, eigenvectors = np.linalg.eigh(projection)
        else:
            eigenvalues, eigenvectors = np.linalg.eig(projection)

        if not np.allclose(eigenvalues.imag, 0, atol=atol):
            raise ValueError("Eigenvalues should be real")
        eigenvalues = eigenvalues.real

        if not np.all(
            (np.isclose(eigenvalues, 0, atol=atol)) | (np.isclose(eigenvalues, 1, atol=atol))
        ):
            raise ValueError("Eigenvalues should be 0 or 1.")

        invariant_indices = np.where(eigenvalues > 0.5)[0]
        if symmetric_projection:
            mode_basis = eigenvectors[:, invariant_indices]
        else:
            # A real, non-orthogonal projection can have a degenerate
            # invariant eigenspace.  ``eig`` is then free to return a complex
            # basis for that otherwise real space.  The left singular vectors
            # span the image of the projection and provide a real,
            # orthonormal basis instead.
            mode_basis = np.linalg.svd(projection, full_matrices=False)[0][
                :, : len(invariant_indices)
            ]

        if not np.allclose(mode_basis.imag, 0, atol=atol):
            raise ValueError("Symmetry-mode basis should be real.")
        mode_basis = np.real(mode_basis)

        # Express the supplied tensor in the invariant-mode basis.
        if not np.any(np.isnan(tensor_components)):
            mode_coefficients = np.linalg.lstsq(mode_basis, tensor_components, rcond=None)[0]
        else:
            mode_coefficients = np.full(mode_basis.shape[1], np.nan)

        # ------------------------
        # Real-space interpretation of modes
        # ------------------------
        if tensor.has_affine_axis:
            component_modes = mode_basis[:-1, :].T
        else:
            component_modes = mode_basis.T

        # ``component_modes`` is flattened to work for arbitrary tensor shapes.
        return projection, mode_coefficients, component_modes
