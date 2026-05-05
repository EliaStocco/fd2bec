# pylint: disable=invalid-name
import warnings
from dataclasses import InitVar, dataclass, field, replace
from functools import cached_property
from typing import Any, Dict, List, Tuple
from warnings import warn

import numpy as np
import spglib
from ase import Atoms
from ase.cell import Cell

from fd2bec import ATOL, DEBUG, SYMPREC, Basis
from fd2bec.mathematics import affine2homogeneous, append_one, find_mapping, invert_indices, wrap
from fd2bec.tensor import Position, Tensor
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
    - Derived properties (`species`, `frac_pos_dict`) are cached for efficiency.
    """

    symbols: List[str]
    cell: Cell
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

        if check:
            conventional = self.conventional
            if not self.is_equal_to(conventional):
                warn("You are not using a conventional unit cell.")

    def clone(self, frac_pos=None, positions=None, **kwargs) -> "AtomicStructure":
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
        return cls(
            symbols=atoms.get_chemical_symbols(),
            positions=positions.copy(),
            cell=atoms.cell.copy(),
            symprec=symprec,
        )

    def to(self, basis: Basis, tensor: Tensor, **kwargs) -> Tensor:
        cell = tensor.cell if tensor.cell is not None else self.cell.array
        return tensor.transform(cell, None, basis, **kwargs)

    def __eq__(self, other: "AtomicStructure") -> bool:
        """Check if two AtomicStructure instances are equal."""
        return self.is_equal_to(other)

    def is_equal_to(self, other: "AtomicStructure", atol=ATOL) -> bool:
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

        if self.species != other.species:
            return False

        if not np.allclose(self.cell, other.cell):
            return False

        try:
            mapping = self._get_atoms_mapping(
                other, atol=atol
            )  # will raise ValueError if not equal
        except ValueError:
            return False
        diff = wrap(self.frac_pos[mapping] - other.frac_pos)
        if not np.allclose(diff, 0, atol=atol):
            return False

        return True

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
        return self.spglib_dataset.number

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
    def spglib_cell(self):
        assert np.allclose(self.positions, self.frac_pos @ self.cell.array)
        return (
            self.cell.array,
            self.frac_pos,
            self.atomic_numbers,
        )

    @cached_property
    def spglib_dataset(self) -> spglib.SpglibDataset:
        """
        Convert the structure to a spglib-compatible cell representation.

        Returns
        -------
        tuple
            (cell, scaled_positions, atomic_numbers) for spglib.
        """
        return spglib.get_symmetry_dataset(self.spglib_cell, symprec=self.symprec)

    @cached_property
    def standardize(self):
        return spglib.standardize_cell(self.spglib_cell, symprec=self.symprec)

    @cached_property
    def conventional(self) -> "AtomicStructure":
        cell, frac_pos, numbers = self.standardize
        return type(self)(
            cell=Cell(cell), frac_pos=frac_pos, symbols=numbers2symbols(numbers), check=False
        )

    def get_space_group_operatios(self,basis="fractional") -> Tuple[np.ndarray, np.ndarray]:
        """Return spglib symmetry (R, t) where x' = R x + t in fractional coords."""
        if basis == "fractional":
            return self.spglib_dataset.rotations, self.spglib_dataset.translations
        else:
            raise NotImplemented

    def _test_symmetry_pbc_fractional(self, atol=ATOL) -> bool:
        """
        Check symmetry: apply x' = R x + t (fractional coords) for all operations.
        """
        R, T = self.get_space_group_operatios()
        for _, (r, t) in enumerate(zip(R, T)):
            new_pos = self.frac_pos @ r.T + t
            new_structure = self.clone(frac_pos=new_pos)
            if not self.is_equal_to(new_structure, atol=atol):
                self.is_equal_to(new_structure, atol=atol)
                raise ValueError("Symmetry operation does not preserve the structure.")
            if self.space_group != new_structure.space_group:
                raise ValueError("Symmetry operation does not preserve the space group")
            mapping = self._get_atoms_mapping(new_structure)
            diff = wrap(self.frac_pos[mapping] - new_structure.frac_pos)
            if not np.allclose(diff, 0, atol=atol):
                raise ValueError("Symmetry operation does not preserve atomic positions")
        return True

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

        for s in self.species:
            idx_self = np.where(np.array(self.symbols) == s)[0]
            idx_other = np.where(np.array(other.symbols) == s)[0]

            a = self.frac_pos_dict[s]
            b = other.frac_pos_dict[s]

            local_map, ok, dists = find_mapping(a, b, atol=atol * len(a))
            if not ok:
                raise ValueError(
                    f"Mapping failed for species {s}."
                    + f" Total distance: {np.linalg.norm(dists)}."
                    + " All distances {dists.tolist()}"
                )

            mapping[idx_other] = idx_self[local_map]

        assert np.all(np.sort(mapping) == np.arange(len(self))), (
            "Invalid mapping: not a permutation"
        )

        return mapping

    def __get_all_atoms_mapping(self, debug=DEBUG):
        """
        Compute inverse atom index mappings for all symmetry operations.

        For each space-group operation (R, t), the function applies the transformation
        to the fractional coordinates, builds the transformed structure, and determines
        how atom indices map back to the original structure.

        Returns an array inv_map such that for each operation k:
            inv_map[k, i] gives the index in the transformed structure corresponding to
            atom i in the original structure.

        Parameters
        ----------
        debug : bool, optional
            If True, performs consistency checks on the mappings.
        **kwargs :
            Passed to the spglib cell construction.

        Returns
        -------
        inv_map : ndarray of shape (Nops, Natoms)
            Inverse atom mappings for each symmetry operation.
        """
        spg = self.spglib_dataset
        R = spg.rotations
        T = spg.translations
        mappings = [None] * len(R)
        for n, (r, t) in enumerate(zip(R, T)):
            new_pos = self.frac_pos @ r + t
            new_structure = self.clone(frac_pos=new_pos)
            mappings[n] = self._get_atoms_mapping(new_structure)
        mappings = np.asarray(mappings)
        inv_map = invert_indices(mappings, axis=1)

        if debug:
            for r, t, m, im in zip(R, T, mappings, inv_map):
                new_pos = self.frac_pos @ r + t
                if not np.allclose(wrap(new_pos - self.frac_pos[m]), 0, atol=SYMPREC):
                    raise ValueError("Error in computing atom mapping for symmetry operation.")
                if not np.allclose(wrap(new_pos[im] - self.frac_pos), 0, atol=SYMPREC):
                    raise ValueError("Error in computing atom mapping for symmetry operation.")
                new_pos = self.frac_pos[im] @ r + t
                if not np.allclose(wrap(new_pos - self.frac_pos), 0, atol=SYMPREC):
                    raise ValueError("Error in computing atom mapping for symmetry operation.")

        return inv_map

    def get_symmetry_operations(self, tensor: Tensor):
        """
        Construct flattened symmetry operations acting on a vector representation.

        Each space-group operation (R, t) together with its atom mapping is converted into an
        affine transformation on a flattened state vector:

            x_flat -> R_flat @ x_flat + T_flat

        where x_flat stacks all components of the input representation. This construction
        combines rotation, translation, and permutation induced by symmetry.

        Parameters
        ----------
        affine : bool, optional
            If False, translation components are ignored (purely linear action).
        **kwargs :
            Passed to the spglib interface.

        Returns
        -------
        R_flat : np.ndarray
            Shape (Nops, dim, dim) linear symmetry operators in flattened form.

        T_flat : np.ndarray
            Shape (Nops, dim) translation vectors in flattened form.

        """
        # if rank != 1 and affine:
        affine = tensor.is_affine
        atomic = tensor.is_atomic
        rank = sum(tensor.rank)

        # if affine:
        #     x = self.frac_pos.copy()
        x_flat = tensor.flatten(full=True)

        spg = self.spglib_dataset
        R = spg.rotations.copy()
        T = spg.translations.copy()
        if atomic:
            mappings = self.__get_all_atoms_mapping()
        else:
            mappings = [None] * len(R)

        Natoms = len(self)
        Nops = len(R)
        ii = np.arange(Natoms)

        if atomic:
            dim = Natoms * (3**rank)
        else:
            dim = 3**rank
        R_flat = np.zeros((Nops, dim, dim))
        T_flat = np.zeros((Nops, dim))

        P = None
        for n, (r, t, m) in enumerate(zip(R, T, mappings)):
            if not affine:
                t[...] = 0.0

            if atomic:
                # Permutation matrix (maps reordered atoms)
                P = np.zeros((Natoms, Natoms))
                P[ii, m] = 1
            # else:
            #     P = np.ones(1)

            # # Flattened rotation (row-vector convention → use r.T)
            # R_cart = r.T
            # for _ in range(rank - 1):
            #     R_cart = np.kron(R_cart, r.T)
            R_cart = tensor.rotation_operator(r.T)

            if atomic:
                R_cart = np.kron(P, R_cart)

            if affine:
                # Flattened translation (must be permuted)
                t_flat = np.tile(t, Natoms)
                t_flat = (P @ t_flat.reshape(Natoms, 3)).reshape(-1)
            else:
                t_flat = np.zeros(dim)

            R_flat[n] = R_cart
            T_flat[n] = t_flat

        if affine:
            x_new = R_flat @ x_flat + T_flat
            diff = x_new - x_flat
            T_flat -= diff

        return R_flat, T_flat

    @cached_property
    def affine_symmetry_operations(self):
        """
        Flattened affine symmetry operations for the atomic coordinates.
        """
        tensor = Position(data=self.frac_pos, basis="fractional")
        return self.get_symmetry_operations(tensor=tensor)

    @cached_property
    def homogeneous_symmetry_operations(self):
        """
        Flattened homogeneous symmetry operations for the atomic coordinates.
        """
        R_flat, T_flat = self.affine_symmetry_operations
        H = affine2homogeneous(R_flat, T_flat)
        return H

    def get_totally_symmetric_projection(self, tensor: Tensor):
        """Construct the projection operator onto the totally symmetric representation."""
        G, T = self.get_symmetry_operations(tensor=tensor)
        if tensor.is_affine:
            G = affine2homogeneous(G, T)
        P = np.mean(G, axis=0)
        return P

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
        theta = np.linalg.lstsq(S, x, rcond=None)[0]  # if x is not None else None

        # ------------------------
        # Real-space interpretation of modes
        # ------------------------
        if tensor.is_affine:
            theta_real = S[:-1, :].T  # .reshape((len(theta), -1, 3))
        else:
            theta_real = S.T  # .reshape((len(theta), -1, 3))

        return S, theta, theta_real  # , shape
