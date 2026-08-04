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

from fd2bec import ATOL, DEBUG, SYMPREC, Basis
from fd2bec.mathematics import affine2homogeneous, append_one, find_mapping, wrap
from fd2bec.tensor import Position, Rotation, Tensor, Translation
from fd2bec.tools import numbers2symbols, symbols2numbers


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

    def _get_atoms_mapping(self, other: "AtomicStructure", atol=ATOL) -> np.ndarray:
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

            local_map, ok, dists = find_mapping(a, b, atol=atol, pbc=self.pbc)
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
            mapping[n] = self._get_atoms_mapping(new_structure)
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
        affine = tensor.is_affine
        atomic = tensor.is_atomic
        rank = sum(tensor.rank)

        x_flat = tensor.flatten(full=True)

        natoms = len(self)
        tensor_dim = 3**rank
        expected_dim = natoms * tensor_dim if atomic else tensor_dim
        if x_flat.shape != (expected_dim,):
            kind = "atomic" if atomic else "global"
            raise ValueError(
                f"Expected one {kind} rank-{rank} tensor with flattened "
                f"shape ({expected_dim},), got {x_flat.shape}."
            )

        if affine:
            if np.all(np.isnan(x_flat)):
                affine_point = np.zeros_like(x_flat)
            elif np.all(np.isfinite(x_flat)):
                affine_point = x_flat
            else:
                raise ValueError("Affine tensor data must be either finite or fully NaN.")

        R, T = self.get_symmetry_operations(basis=tensor.basis)
        if atomic:
            mappings = self.__get_all_atoms_mapping()
        else:
            mappings = [None] * len(R)

        nops = len(R)
        if R.shape != (nops, 3, 3):
            raise ValueError(f"Expected rotations with shape ({nops}, 3, 3), got {R.shape}.")
        if T.shape != (nops, 3):
            raise ValueError(f"Expected translations with shape ({nops}, 3), got {T.shape}.")
        if atomic and len(mappings) != nops:
            raise ValueError("Number of atomic mappings does not match symmetry operations.")

        atom_indices = np.arange(natoms)
        R_flat = np.zeros((nops, expected_dim, expected_dim))
        T_flat = np.zeros((nops, expected_dim))

        for n, (r, t, m) in enumerate(zip(R, T, mappings)):
            if atomic:
                # Permutation matrix (maps reordered atoms)
                permutation = np.zeros((natoms, natoms))
                permutation[atom_indices, m] = 1

            R_tensor = tensor.rotation_operator(r.T)

            if atomic:
                R_tensor = np.kron(permutation, R_tensor)

            if affine:
                if atomic:
                    # Repeat the translation for every atom, then apply the
                    # same permutation as for the linear part.
                    t_flat = np.tile(t, natoms)
                    t_flat = (permutation @ t_flat.reshape(natoms, 3)).reshape(-1)
                else:
                    # A global affine vector has one translation only and no
                    # atomic indices to tile or permute.
                    t_flat = np.asarray(t).copy()
            else:
                t_flat = np.zeros(expected_dim)

            R_flat[n] = R_tensor
            T_flat[n] = t_flat

        if affine:
            # Correct lattice-image translations so every affine operation
            # fixes the supplied tensor value exactly. The same expression is
            # valid for atomic positions and for a global affine dipole.
            transformed = R_flat @ affine_point + T_flat
            T_flat += affine_point - transformed

        return R_flat, T_flat

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

                S = pga.get_symmetry_operations()
                R = np.asarray([s.rotation_matrix for s in S])
                T = np.asarray([s.translation_vector for s in S])

                O = np.mean(self.positions, axis=0)
                Teff = T + O[None, :] - np.asarray([O @ r.T for r in R])

                return R, Teff

        else:
            raise NotImplementedError

    def get_totally_symmetric_projection(self, tensor: Tensor):
        """Construct the projection operator onto the totally symmetric representation."""
        G, T = self.get_tensor_symmetry_operations(tensor=tensor)
        if tensor.is_affine:
            G = affine2homogeneous(G, T)
        P = np.mean(G, axis=0)
        return P

    def symmetrize(self, tensor: Tensor, debug=True) -> Tensor:
        P = self.get_totally_symmetric_projection(tensor=tensor)
        out = P @ tensor.flatten(full=True)
        assert np.allclose(out, P @ out, atol=ATOL), "error"
        out = np.reshape(out, tensor.shape)
        return type(tensor)(data=out)

    def get_symmetrizer(
        self,
        tensor: Tensor,
        debug: bool = DEBUG,
        atol: float = ATOL,
    ):

        # ------------------------
        # Projection construction
        # ------------------------
        P = self.get_totally_symmetric_projection(tensor=tensor)

        # ------------------------
        # Vector construction
        # ------------------------
        # shape = (3,) * sum(x.rank)
        # if x.is_atomic:
        #     shape = (len(self), *shape)
        # if x is None:
        #     x = np.zeros(shape)

        # assert x.shape == shape, f"Wrong shape, expected {shape} but got {x.shape}."

        x = tensor.flatten(full=True)
        if tensor.is_affine:
            x = append_one(x)

        # ------------------------
        # Eigen-decomposition
        # ------------------------

        if np.linalg.norm((P - P.T)) < atol:
            # P is symmetric, use eigh for better numerical stability
            w, v = np.linalg.eigh(P)
        else:
            warnings.warn("Projection operator is not symmetric.", UserWarning)
            w, v = np.linalg.eig(P)

        if debug and not np.allclose(w.imag, 0, atol=atol):
            raise ValueError("Eigenvalues should be real")
        w = w.real

        if debug and not np.all((np.isclose(w, 0, atol=atol)) | (np.isclose(w, 1, atol=atol))):
            raise ValueError("Eigenvalues should be 0 or 1.")

        mask = np.where(w > 0.5)[0]
        S = v[:, mask]
        if debug and not np.allclose(S.imag, 0):
            raise ValueError("Eigenvectors should be real")
        S = np.real(S)

        # Solve for theta
        if not np.any(np.isnan(x.data)):
            theta = np.linalg.lstsq(S, x.data, rcond=None)[0]  # if x is not None else None
        else:
            theta = np.full(S.shape[1], np.nan)

        # ------------------------
        # Real-space interpretation of modes
        # ------------------------
        if tensor.is_affine:
            theta_real = S[:-1, :].T  # .reshape((len(theta), -1, 3))
        else:
            theta_real = S.T  # .reshape((len(theta), -1, 3))

        return S, theta, theta_real  # , shape
