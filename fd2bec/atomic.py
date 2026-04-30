# pylint: disable=invalid-name
import warnings
from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np
import spglib
from ase import Atoms
from ase.cell import Cell
from ase.data import atomic_numbers
from ase.geometry import cellpar_to_cell

from fd2bec import ATOL, DEBUG, SYMPREC, Basis, validate_types
from fd2bec.mathematics import affine2homogeneous, append_one, find_mapping, invert_indices, wrap
from fd2bec.tensor import Position, Tensor


@dataclass(frozen=True)
class AtomicStructure:
    """
    Immutable representation of an atomic structure.

    Attributes
    ----------
    symbols : tuple[str, ...]
        Chemical symbols of atoms in order.
    cellpar : np.ndarray
        Cell parameters (a, b, c, alpha, beta, gamma).
    frac_pos : np.ndarray
        Fractional (scaled) atomic positions with shape (N, 3).

    Notes
    -----
    - The class is fully immutable:
        - `symbols` is stored as a tuple
        - NumPy arrays are copied and marked read-only
    - Derived properties (`species`, `frac_pos_dict`) are cached for efficiency.
    """

    symbols: tuple[str, ...]
    cellpar: np.ndarray
    frac_pos: np.ndarray

    def __post_init__(self):
        """
        Enforce immutability by:
        - Converting symbols to tuple
        - Copying NumPy arrays
        - Marking arrays as read-only
        """
        object.__setattr__(self, "symbols", tuple(self.symbols))

        cellpar = np.array(self.cellpar, copy=True)
        frac_pos = np.array(self.frac_pos, copy=True)

        cellpar.setflags(write=False)
        frac_pos.setflags(write=False)

        object.__setattr__(self, "cellpar", cellpar)
        object.__setattr__(self, "frac_pos", frac_pos)

    # def duplicate(self, **kwargs) -> "AtomicStructure":
    #     """
    #     Create a new AtomicStructure with some attributes modified.

    #     Parameters
    #     ----------
    #     **kwargs
    #         Any of the attributes (symbols, cellpar, frac_pos) can be overridden.

    #     Returns
    #     -------
    #     AtomicStructure
    #         New instance with updated attributes.
    #     """
    #     return AtomicStructure(
    #         symbols=kwargs.get("symbols", self.symbols),
    #         cellpar=kwargs.get("cellpar", self.cellpar),
    #         frac_pos=kwargs.get("frac_pos", self.frac_pos),
    #     )

    @classmethod
    def from_ase(cls, atoms: Atoms, keyword: str = "positions") -> "AtomicStructure":
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
        frac_pos = atoms.cell.scaled_positions(atoms.arrays[keyword])
        return cls(
            symbols=tuple(atoms.get_chemical_symbols()),
            frac_pos=frac_pos,
            cellpar=atoms.cell.cellpar(),
        )

    @cached_property
    def cell(self) -> Cell:
        return Cell.fromcellpar(self.cellpar)

    @validate_types
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

        if not np.allclose(self.cellpar, other.cellpar):
            return False

        try:
            mapping = self.__get_atoms_mapping(
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
        return self.to_spglib_cell().number

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
    def positions(self) -> np.ndarray:
        return self.cell.cartesian_positions(self.frac_pos)

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
            "cellpar": self.cellpar.tolist(),
            "frac_pos": self.frac_pos.tolist(),
        }

    def to_spglib_cell(self, symprec=SYMPREC, **kwargs) -> spglib.SpglibDataset:
        """
        Convert the structure to a spglib-compatible cell representation.

        Returns
        -------
        tuple
            (cell, scaled_positions, atomic_numbers) for spglib.
        """
        cell = (
            np.transpose(cellpar_to_cell(self.cellpar)),
            self.frac_pos,
            [atomic_numbers[s] for s in self.symbols],
        )
        return spglib.get_symmetry_dataset(cell, symprec=symprec, **kwargs)

    def _test_symmetry(self, symprec=SYMPREC, atol=ATOL, **kwargs) -> bool:
        spg = self.to_spglib_cell(symprec=symprec, **kwargs)
        R = spg.rotations
        T = spg.translations
        for _, (r, t) in enumerate(zip(R, T)):
            new_pos = self.frac_pos @ r + t
            new_structure = replace(self, frac_pos=new_pos)
            if not self.is_equal_to(new_structure, atol=atol):
                self.is_equal_to(new_structure, atol=atol)
                raise ValueError("Symmetry operation does not preserve the structure")
            if self.space_group != new_structure.space_group:
                raise ValueError("Symmetry operation does not preserve the space group")
            mapping = self.__get_atoms_mapping(new_structure)
            diff = wrap(self.frac_pos[mapping] - new_structure.frac_pos)
            if not np.allclose(diff, 0, atol=atol):
                raise ValueError("Symmetry operation does not preserve atomic positions")
        return True

    def __get_atoms_mapping(self, other: "AtomicStructure", atol=ATOL) -> np.ndarray:
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

    def __get_all_atoms_mapping(self, debug=DEBUG, **kwargs):
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
        spg = self.to_spglib_cell(**kwargs)
        R = spg.rotations
        T = spg.translations
        mappings = [None] * len(R)
        for n, (r, t) in enumerate(zip(R, T)):
            new_pos = self.frac_pos @ r + t
            new_structure = replace(self, frac_pos=new_pos)
            mappings[n] = self.__get_atoms_mapping(new_structure)
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

    def get_symmetry_operations(self, tensor: Tensor, **kwargs):
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

        spg = self.to_spglib_cell(**kwargs)
        R = spg.rotations.copy()
        T = spg.translations.copy()
        if atomic:
            mappings = self.__get_all_atoms_mapping(**kwargs)
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

    def get_affine_symmetry_operations(self, **kwargs):
        """
        Flattened affine symmetry operations for the atomic coordinates.
        """
        # assert kwargs.pop("rank", 1) == 1, "error"
        # assert kwargs.pop("atomic", True), "error"
        # assert kwargs.pop("affine", True), "error"
        tensor = Position(data=self.frac_pos, basis="fractional")
        return self.get_symmetry_operations(tensor=tensor, **kwargs)

    def get_homogeneous_symmetry_operations(self, **kwargs):
        """
        Flattened homogeneous symmetry operations for the atomic coordinates.
        """
        R_flat, T_flat = self.get_affine_symmetry_operations(**kwargs)
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
