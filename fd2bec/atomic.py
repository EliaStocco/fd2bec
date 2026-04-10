import numpy as np
import spglib
import warnings
from ase import Atoms
from ase.cell import Cell
from dataclasses import dataclass
from functools import cached_property
from fd2bec import SYMPREC, DEBUG, ATOL
from fd2bec.mathematics import wrap, find_mapping, invert_indices, affine2homogeneous, append_one
from fd2bec.symmetry import SymmetryRepresentationBuilder
from ase.data import atomic_numbers
from ase.geometry import cellpar_to_cell

BUILDER = "internal"

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
    def from_ase(cls, atoms: Atoms, keyword:str='positions') -> "AtomicStructure":
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
    def cell(self)->Cell:
        return Cell.fromcellpar(self.cellpar)
    
    def get_fractional(self,arr:np.ndarray)->np.ndarray:
        # arr = arr.reshape(len(self),3,-1)
        return self.cell.scaled_positions(arr)
        
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
    
    def __get_symmetry_operations(
        self, 
        use_translations=True, 
        atol=SYMPREC, 
        debug=DEBUG, 
        x: np.ndarray = None, 
        rank: int = 1,
        **kwargs
        ):
        """
        Construct flattened symmetry operations acting on a vector representation.

        Each space-group operation (R, t) together with its atom mapping is converted into an
        affine transformation on a flattened state vector:

            x_flat -> R_flat @ x_flat + T_flat

        where x_flat stacks all components of the input representation. This construction
        combines rotation, translation, and permutation induced by symmetry.

        Parameters
        ----------
        use_translations : bool, optional
            If False, translation components are ignored (purely linear action).
        atol : float, optional
            Numerical tolerance used in debug validation.
        debug : bool, optional
            If True, verifies correctness of the flattened operators using x.
        x : np.ndarray, optional
            Reference state used only for debug validation (can represent positions or other
            compatible vector fields).
        **kwargs :
            Passed to the spglib interface.

        Returns
        -------
        R_flat : np.ndarray
            Shape (Nops, dim, dim) linear symmetry operators in flattened form.

        T_flat : np.ndarray
            Shape (Nops, dim) translation vectors in flattened form.

        Raises
        ------
        ValueError
            If debug=True and the flattened operations fail validation within tolerance.
        """
        if rank not in [1,2]:
            raise ValueError("Only rank 1 and 2 have been implemented.")
            
        if rank != 1 and use_translations:
            raise ValueError("Translations only apply to rank-1 objects (positions).")
        
        # if x is not None and use_translations:
        #     warnings.warn(
        #         "When 'use_translations' == True the variable 'x' will be ignored and automatically set to the fractional coordinates.",
        #         RuntimeWarning
        #     )
        if use_translations and debug and x is None:
            x = self.frac_pos.copy()
        if x is not None:
            if x.ndim < 2:
                raise ValueError("Please provide a non-flattened 'x' array.")
            x_flat = x.flatten()
        if debug and x is None:
            raise ValueError("To use 'debug' = True you need to provide 'x'.")

        spg = self.to_spglib_cell(**kwargs)
        R = spg.rotations.copy()
        T = spg.translations.copy()
        mappings = self.__get_all_atoms_mapping(**kwargs)

        Natoms = len(self)
        Nops = len(R)
        ii = np.arange(Natoms)

        dim = Natoms * (3 ** rank)
        R_flat = np.zeros((Nops, dim, dim))
        T_flat = np.zeros((Nops, dim))
        
        for n, (r, t, m) in enumerate(zip(R, T, mappings)):
            
            if not use_translations:
                t[...] = 0.

            # Permutation matrix (maps reordered atoms)
            P = np.zeros((Natoms, Natoms))
            P[ii, m] = 1

            # Flattened rotation (row-vector convention → use r.T)
            R_cart = r.T
            for _ in range(rank - 1):
                R_cart = np.kron(R_cart, r.T)

            r_flat = np.kron(P, R_cart)

            # Flattened translation (must be permuted)
            t_flat = np.tile(t, Natoms)
            t_flat = (P @ t_flat.reshape(Natoms, 3)).reshape(-1)

            if debug:
                if rank == 1:
                    a = (r_flat @ x_flat + t_flat)
                    b = (x @ r + t)[m].flatten()

                elif rank == 2:
                    a = r_flat @ x_flat
                    b = (x[m] @ r @ r.T).flatten()  # conceptual check
                    
                if not np.allclose(wrap(a - b), 0):
                    raise ValueError("Error in flattening symmetry operation.")
                
            # # This check is redundant since in the next debug block we are going to do the same thing.
            # if debug:
            #     x_new = np.asarray(r_flat @ x_flat + t_flat).reshape((Natoms,3))
            #     if not np.allclose(wrap(x_new - x), 0, atol=atol):
            #         raise ValueError("Error in applying flattened symmetry operation.")               
            
            R_flat[n] = r_flat
            T_flat[n] = t_flat
            
        if use_translations or debug:
            x_new = R_flat @ x_flat + T_flat
            diff = x_new - x_flat
            T_flat -= diff
        
        if debug:
            # positions are the same modulo 1
            if not np.allclose(wrap(diff), 0, atol=atol):
                raise ValueError("Error in applying flattened symmetry operation.")    
            # positions are the same with the translation correction
            if rank == 1:
                x_new = R_flat @ x_flat + T_flat
                ref = x_flat

            elif rank == 2:
                x_new = R_flat @ x_flat
                ref = x_flat

            if not np.allclose(x_new, ref, atol=atol):
                raise ValueError(f"Error in rank-{rank} symmetry operation.")
            
        return R_flat, T_flat

    def get_affine_symmetry_operations(self,debug=DEBUG,**kwargs):
        """
        Flattened affine symmetry operations for the atomic coordinates.
        """
        if BUILDER == "internal":
            R_flat, T_flat = self.__get_symmetry_operations(use_translations=True,debug=debug,**kwargs)
        else:
            builder = SymmetryRepresentationBuilder(natoms=len(self))
            spg = self.to_spglib_cell(**kwargs)
            R = spg.rotations.copy()
            T = spg.translations.copy()
            mapping = self.__get_all_atoms_mapping(**kwargs)
            R_flat = np.asarray([builder.build_R_flat(mapping, r, rank=1) for r in R])
            T_flat = np.asarray([builder.build_T_flat(mapping, t) for t in T])
        return R_flat, T_flat
    
    def get_homogeneous_symmetry_operations(self,**kwargs):
        """
        Flattened homogeneous symmetry operations for the atomic coordinates.
        """
        R_flat, T_flat = self.get_affine_symmetry_operations(**kwargs)
        H = affine2homogeneous(R_flat, T_flat)
        return H
    
    def get_symmetry_operations(self,rank=1,debug=DEBUG,**kwargs):
        """
        Flattened symmetry operations for the atomic tensors.
        """
        if BUILDER == "internal":
            return self.__get_symmetry_operations(use_translations=False,debug=debug,**kwargs)[0]
        else:
            builder = SymmetryRepresentationBuilder(natoms=len(self))
            spg = self.to_spglib_cell(**kwargs)
            R = spg.rotations.copy()
            # T = spg.translations.copy()
            mapping = self.__get_all_atoms_mapping(**kwargs)
            R_flat = builder.build_R_flat(mapping, R, rank=rank)
            return R_flat
            

    def get_symmetrizer(
        self,
        what: str,
        method: str = "eigen",
        x: np.ndarray = None,
        atol=ATOL,
        debug=DEBUG,
        **kwargs
    ):
        """
        Compute a symmetry-adapted basis for atomic or tensorial configurations.

        This method constructs a matrix S whose columns span the symmetry-invariant
        subspace of a representation space. Any symmetry-invariant configuration can
        be written as:

            x = S @ theta

        where:
        - x is the flattened configuration vector (e.g. 3N or 3N+1 in homogeneous form)
        - theta contains the independent symmetry-allowed degrees of freedom

        The symmetric subspace is defined as the eigenspace of eigenvalue 1 of the
        group-averaging operator:

            P = (1/|G|) ∑_g G_g

        Parameters
        ----------
        what : str
            Type of object the symmetry acts on:

            - "positions":
                Symmetry acts on atomic positions in fractional coordinates using
                homogeneous representations (includes translation component).

            - "vector":
                Symmetry acts on vector-like quantities using standard linear
                representation matrices.

            - "tensor":
                Not implemented.

        method : str, optional
            Method used to construct the symmetric subspace:

            - "null_space":
                Builds the constraint matrix A = stack_g (G_g - I) and computes its
                null space using SVD.
                WARNING: This method can be very slow and memory intensive for large systems.

            - "eigen":
                Builds the projection (averaging) operator:

                    P = (1/N) ∑_g G_g

                and computes its eigendecomposition. The symmetric subspace corresponds
                to eigenvectors associated with eigenvalue 1.

                Numerically, eigenvalues are validated to be close to {0, 1} within
                tolerance atol.

                This method is recommended for large systems.

        x : np.ndarray, optional
            Configuration vector to project onto the symmetric subspace. If None,
            theta is not computed.

        atol : float, optional
            Numerical tolerance used for eigenvalue filtering and validation checks.

        debug : bool, optional
            If True, performs consistency checks on reconstructed configurations.

        Returns
        -------
        S : np.ndarray
            Basis matrix of shape (dim, k), whose columns span the symmetry-invariant
            subspace.

        theta : np.ndarray
            Reduced symmetry-adapted coordinates such that x ≈ S @ theta.

        theta_real : np.ndarray
            Real-space interpretation of symmetry-adapted modes with shape:
                (k, Natoms, 3)
            Each mode corresponds to a symmetry-allowed displacement pattern.

        Notes
        -----
        In eigen-method mode, the matrix P is a projection operator. Therefore:
        - eigenvalues should be ~0 or ~1
        - eigenvectors with eigenvalue ~1 span the invariant subspace
        """
        choices = ['positions','vector','tensor']
        if what not in choices:
            raise ValueError(f"'what' can only be one of {choices} but got '{what}'.")
        
        if what == 'positions':
            if x is None:
                x = self.frac_pos.copy()
            G = self.get_homogeneous_symmetry_operations(atol=atol, debug=debug,x=x, **kwargs)
            # if x is not None:
            #     warnings.warn(
            #         "When 'what' == 'positions' the variable 'x' will be ignored and automatically set to the fractional coordinates.",
            #         RuntimeWarning
            #     )
            x = append_one(x.flatten())
        elif what == 'vector':
            if debug and x is None:
                warnings.warn(
                    "To use 'debug' = True you need to provide 'x'.",
                    RuntimeWarning
                )
            G = self.get_symmetry_operations(atol=atol,debug=debug and x is not None,x=x,**kwargs)
        elif what == 'tensor':
            raise ValueError("'what' = 'tensor' has not been implemented yet.")
        
        x = x.flatten()

        _, dim, _ = G.shape

        # ------------------------
        # Method 1: Null space
        # ------------------------
        if method == "null_space":
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
            
            if not np.allclose(w.imag,0,atol=atol):
                raise ValueError("Eigenvalues should be real")
            w = w.real
            
            if not np.all((np.isclose(w, 0,atol=atol)) | (np.isclose(w, 1, atol=atol))):
                raise ValueError("Eigenvalues should be 0 or 1.")
        
            mask = np.where(w > 0.5)[0]
            S = v[:, mask]
            if not np.allclose(S.imag,0):
                raise ValueError("Eigenvectors should be real")
            S = S.real

        else:
            raise ValueError("method must be either 'null_space' or 'eigen'")

        # Solve for theta
        theta = np.linalg.lstsq(S, x, rcond=None)[0] if x is not None else None

        # ------------------------
        # Debug checks
        # ------------------------
        if debug:
            test = S @ theta
            if what == 'positions':
                assert np.allclose(test[-1], 1.0, atol=atol), \
                    f"Last component should be one but it is {test[-1]}."
            diff = test - x
            if not np.allclose(diff, 0, atol=atol):
                raise ValueError("There is a problem here.")

        # ------------------------
        # Real-space interpretation of modes
        # ------------------------
        if what == 'positions':
            theta_real = S[:-1, :].T.reshape((len(theta), -1, 3))
        elif theta is not None:
            theta_real = S.T.reshape((len(theta), -1, 3))

        return S, theta, theta_real