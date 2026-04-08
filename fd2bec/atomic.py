import numpy as np
import spglib
from ase import Atoms
from dataclasses import dataclass
from functools import cached_property
from fd2bec import SYMPREC
from fd2bec.mathematics import wrap, find_mapping
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
            mapping = self.get_atoms_mapping(other)  # will raise ValueError if not equal
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

    def get_atoms_mapping(self, other: "AtomicStructure") -> np.ndarray:
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

            local_map, ok = find_mapping(a, b)
            if not ok:
                raise ValueError(f"Mapping failed for species {s}")

            mapping[idx_other] = idx_self[local_map]

        assert np.all(np.sort(mapping) == np.arange(len(self))), \
            "Invalid mapping: not a permutation"

        return mapping
        
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
            if self != new_structure:
                self.is_equal_to(new_structure)
                raise ValueError("Symmetry operation does not preserve the structure")
            if self.space_group != new_structure.space_group:
                raise ValueError("Symmetry operation does not preserve the space group")
            mapping = self.get_atoms_mapping(new_structure)
            diff = wrap(self.frac_pos[mapping] - new_structure.frac_pos)
            if not np.allclose(diff,0,atol=atol):
                raise ValueError("Symmetry operation does not preserve atomic positions")
        return True
    
