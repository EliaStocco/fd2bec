import numpy as np
import spglib
from ase import Atoms
from dataclasses import dataclass
from functools import cached_property
from fd2bec import SYMPREC, DEBUG, ATOL
from fd2bec.mathematics import wrap, find_mapping, invert_indices, affine2homogeneous
from ase.data import atomic_numbers
from ase.geometry import cellpar_to_cell

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
    - Derived properties (`species`, `pos`) are cached for efficiency.
    """

    symbols: tuple[str, ...]
    cellpar: np.ndarray
    frac_pos: np.ndarray
    
    def duplicate(self, **kwargs) -> "AtomicStructure":
        """
        Create a new AtomicStructure with some attributes modified.

        Parameters
        ----------
        **kwargs
            Any of the attributes (symbols, cellpar, frac_pos) can be overridden.

        Returns
        -------
        AtomicStructure
            New instance with updated attributes.
        """
        return AtomicStructure(
            symbols=kwargs.get("symbols", self.symbols),
            cellpar=kwargs.get("cellpar", self.cellpar),
            frac_pos=kwargs.get("frac_pos", self.frac_pos),
        )

    @classmethod
    def from_ase(cls, atoms: Atoms) -> "AtomicStructure":
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
        return cls(
            symbols=tuple(atoms.get_chemical_symbols()),
            frac_pos=atoms.get_scaled_positions(),
            cellpar=atoms.cell.cellpar(),
        )

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

    def __eq__(self, other: "AtomicStructure") -> bool:
        """Check if two AtomicStructure instances are equal."""
        return self.is_equal_to(other)
    
    def is_equal_to(self, other: "AtomicStructure", atol=SYMPREC)-> bool:
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
            mapping = self.__get_atoms_mapping(other,atol=atol)  # will raise ValueError if not equal
        except ValueError as e:
            return False
        diff = wrap(self.frac_pos[mapping] - other.frac_pos)
        if not np.allclose(diff,0,atol=atol):
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
        """Space group number of the structure.
        """
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
    def pos(self) -> dict[str, np.ndarray]:
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
    
    def to_spglib_cell(self,**kwargs) -> spglib.SpglibDataset:
        """
        Convert the structure to a spglib-compatible cell representation.

        Returns
        -------
        tuple
            (cell, scaled_positions, atomic_numbers) for spglib.
        """
        cell = cellpar_to_cell(self.cellpar), self.frac_pos, [atomic_numbers[s] for s in self.symbols]
        return spglib.get_symmetry_dataset(cell, **kwargs)
    
    def _test_symmetry(self,atol=SYMPREC,**kwargs)->bool:
        spg = self.to_spglib_cell(**kwargs)
        R = spg.rotations
        T = spg.translations
        for r,t in zip(R,T):
            new_pos = self.frac_pos @ r + t
            new_structure = self.duplicate(frac_pos=new_pos)
            if not self.is_equal_to(new_structure,atol=atol):
                self.is_equal_to(new_structure,atol=atol)
                raise ValueError("Symmetry operation does not preserve the structure")
            if self.space_group != new_structure.space_group:
                raise ValueError("Symmetry operation does not preserve the space group")
            mapping = self.__get_atoms_mapping(new_structure)
            diff = wrap(self.frac_pos[mapping] - new_structure.frac_pos)
            if not np.allclose(diff,0,atol=atol):
                raise ValueError("Symmetry operation does not preserve atomic positions")
        return True
    
    def __get_atoms_mapping(self, other: "AtomicStructure",atol=ATOL) -> np.ndarray:
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

            a = self.pos[s]
            b = other.pos[s]

            local_map, ok = find_mapping(a, b, atol=atol)
            if not ok:
                raise ValueError(f"Mapping failed for species {s}")

            mapping[idx_other] = idx_self[local_map]

        assert np.all(np.sort(mapping) == np.arange(len(self))), \
            "Invalid mapping: not a permutation"

        return mapping
    
    def __get_all_atoms_mapping(self,debug=DEBUG,**kwargs):
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
        mappings = []
        for r,t in zip(R,T):
            new_pos = self.frac_pos @ r + t
            new_structure = self.duplicate(frac_pos=new_pos)
            mapping = self.__get_atoms_mapping(new_structure)
            mappings.append(mapping)
        mappings = np.asarray(mappings)
        inv_map = invert_indices(mappings, axis=1)
        
        if debug:
            for r,t,m,im in zip(R,T,mappings,inv_map):
                new_pos = self.frac_pos @ r + t
                if not np.allclose(wrap(new_pos - self.frac_pos[m]), 0,atol=SYMPREC):
                    raise ValueError("Error in computing atom mapping for symmetry operation.")
                if not np.allclose(wrap(new_pos[im] - self.frac_pos), 0,atol=SYMPREC):
                    raise ValueError("Error in computing atom mapping for symmetry operation.")
                new_pos = self.frac_pos[im] @ r + t
                if not np.allclose(wrap(new_pos - self.frac_pos), 0,atol=SYMPREC):
                    raise ValueError("Error in computing atom mapping for symmetry operation.")
                
        return inv_map
    
    def get_affine_symmetry_operations(self, atol=SYMPREC, debug=DEBUG, **kwargs):
        """
        Construct flattened symmetry operations acting on the full atomic coordinate vector.

        Each symmetry operation (R, t), together with its induced atom mapping m, is converted
        into an affine transformation acting on the flattened fractional coordinates:

            x_flat -> R_flat @ x_flat + T_flat

        where x_flat is a vector of shape (3 * Natoms,) obtained by concatenating all atomic
        fractional positions. The flattened operators consistently combine:
        - rotation in fractional coordinates,
        - translation,
        - permutation of atoms induced by the symmetry operation.

        Parameters
        ----------
        atol : float, optional
            Numerical tolerance used for validation checks (only if debug=True).
        debug : bool, optional
            If True, perform consistency checks to verify correctness of the flattened operators.
        **kwargs :
            Additional arguments passed to the spglib interface.

        Returns
        -------
        R_flat : np.ndarray
            Array of shape (Nops, 3*Natoms, 3*Natoms) containing flattened linear operators.

        T_flat : np.ndarray
            Array of shape (Nops, 3*Natoms) containing flattened translation vectors.

        Raises
        ------
        ValueError
            If debug=True and any constructed operation fails to reproduce the symmetry action
            within the specified tolerance.
        """
        spg = self.to_spglib_cell(**kwargs)
        R = spg.rotations
        T = spg.translations
        mappings = self.__get_all_atoms_mapping(**kwargs)

        Natoms = len(self)
        Nops = len(R)
        ii = np.arange(Natoms)

        R_flat = np.zeros((Nops, 3 * Natoms, 3 * Natoms))
        T_flat = np.zeros((Nops, 3 * Natoms))

        pos = self.frac_pos.copy()
        pos_flat = pos.flatten()

        for n, (r, t, m) in enumerate(zip(R, T, mappings)):

            # Permutation matrix (maps reordered atoms)
            P = np.zeros((Natoms, Natoms))
            P[ii, m] = 1

            # Flattened rotation (row-vector convention → use r.T)
            r_flat = np.kron(P, r.T)

            # Flattened translation (must be permuted)
            t_flat = np.tile(t, Natoms)
            t_flat = (P @ t_flat.reshape(Natoms, 3)).reshape(-1)

            if debug:
                # Validate against direct application
                a = (r_flat @ pos_flat + t_flat)
                b = (pos @ r + t)[m].flatten()
                c = (pos[m] @ r + t).flatten()

                if not np.allclose(wrap(a - b), 0):
                    raise ValueError("Error in flattening symmetry operation.")
                
                if not np.allclose(wrap(b - c), 0):
                    raise ValueError("Just a test.")

            # # This check is redundant since in the next debug block we are going to do the same thing.
            # if debug:
            #     new_pos = np.asarray(r_flat @ pos_flat + t_flat).reshape((Natoms,3))
            #     if not np.allclose(wrap(new_pos - pos), 0, atol=atol):
            #         raise ValueError("Error in applying flattened symmetry operation.")               
            
            R_flat[n] = r_flat
            T_flat[n] = t_flat
            
        new_pos = R_flat @ pos_flat + T_flat
        diff = new_pos - pos_flat
        T_flat -= diff
        
        if debug:
            # positions are the same modulo 1
            if not np.allclose(wrap(diff), 0, atol=atol):
                raise ValueError("Error in applying flattened symmetry operation.")    
            # positions are the same with the translation correction
            new_pos = R_flat @ pos_flat + T_flat
            if not np.allclose(new_pos, pos_flat, atol=atol):
                raise ValueError("Error in applying flattened symmetry operation.")   
            
        return R_flat, T_flat

    def get_homogeneous_symmetry_operations(self,**kwargs):
        """
        Construct homogeneous symmetry operations corresponding to the flattened affine operations.

        Returns
        -------
        H_ops : np.ndarray
            Array of shape (Nops, 3*Natoms+1, 3*Natoms+1) containing homogeneous transformation matrices.
        """
        R_flat, T_flat = self.get_affine_symmetry_operations(**kwargs)
        H = affine2homogeneous(R_flat, T_flat)
        return H

    def get_symmetrizer(
        self,
        use_translations: bool = True,
        method: str = "null_space",
        atol=ATOL,
        debug=DEBUG,
        **kwargs
    ):
        """
        Compute a symmetry-adapted basis for atomic coordinates.

        This method returns a matrix S whose columns span the symmetric subspace
        of atomic configurations such that any symmetry-invariant configuration
        can be written as:

            x = S @ theta

        where:
        - x is the flattened atomic coordinate vector (dimension 3N or 3N+1)
        - theta contains the independent (symmetry-allowed) degrees of freedom

        Parameters
        ----------
        use_translations : bool, optional
            If True, include translational components using homogeneous coordinates.
            If False, use purely affine (linear) symmetry operations.

        method : str, optional
            Method used to construct the symmetric subspace:

            - "null_space":
                Construct the constraint matrix A by stacking (G_i - I) for all symmetry
                operations and compute its null space.
                WARNING: This approach can be very slow and memory intensive for large
                systems, since A can become extremely large.

            - "eigen":
                Construct the averaging operator P = (1/N) sum_i G_i and compute its
                eigen-decomposition. The symmetric subspace corresponds to eigenvectors
                with eigenvalue 1. This method is significantly more efficient and
                recommended for large systems.

        atol : float, optional
            Numerical tolerance used for eigenvalue selection and validation checks.

        debug : bool, optional
            If True, perform consistency checks to verify correctness of the result.

        Returns
        -------
        S : np.ndarray
            Matrix of shape (dim, k) whose columns form a basis of the symmetric subspace.

        theta : np.ndarray
            Reduced coordinates such that x ≈ S @ theta.

        theta_real : np.ndarray
            Real-space interpretation of the symmetry-adapted modes, with shape:
                (k, Natoms, 3)
            Each entry corresponds to the displacement pattern associated with one
            independent degree of freedom.
        """

        import numpy as np

        if use_translations:
            G = self.get_homogeneous_symmetry_operations(atol=atol, debug=debug, **kwargs)
            x = np.append(self.frac_pos.copy(), 1.0)
        else:
            G, _ = self.get_affine_symmetry_operations(atol=atol, debug=debug, **kwargs)
            x = self.frac_pos.copy()

        _, dim, _ = G.shape

        # ------------------------
        # Method 1: Null space
        # ------------------------
        if method == "null_space":
            import warnings
            warnings.warn(
                "Using 'null_space' method: this can be very slow and memory intensive "
                "for large systems. Consider using method='eigen' instead.",
                RuntimeWarning
            )

            from scipy.linalg import null_space

            A_blocks = []
            I = np.eye(dim)

            for g in G:
                A_blocks.append(g - I)

            A = np.vstack(A_blocks)

            S = null_space(A, rcond=atol)

        # ------------------------
        # Method 2: Eigen-decomposition
        # ------------------------
        elif method == "eigen":
            P = np.mean(G, axis=0)

            w, v = np.linalg.eig(P)

            mask = np.isclose(w, 1.0, atol=atol)
            S = v[:, mask]

        else:
            raise ValueError("method must be either 'null_space' or 'eigen'")

        # Solve for theta
        theta = np.linalg.lstsq(S, x, rcond=None)[0]

        # ------------------------
        # Debug checks
        # ------------------------
        if debug:
            test = S @ theta
            if use_translations:
                assert np.allclose(test[-1], 1.0, atol=atol), \
                    f"Last component should be one but it is {test[-1]}."
            diff = test - x
            assert np.allclose(diff, 0, atol=atol), \
                "There is a problem here."

        # ------------------------
        # Real-space interpretation of modes
        # ------------------------
        if use_translations:
            theta_real = S[:-1, :].T.reshape((len(theta), -1, 3))
        else:
            theta_real = S.T.reshape((len(theta), -1, 3))

        return S, theta, theta_real