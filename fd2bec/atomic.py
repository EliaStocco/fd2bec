import numpy as np
import spglib
from ase import Atoms
from dataclasses import dataclass
from functools import cached_property
from fd2bec.hungarian import equal_rows_hungarian   
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

        for s in self.species:
            a = self.pos[s]
            b = other.pos[s]
            if not equal_rows_hungarian(a, b):
                return False

        return True

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
    
def structures_equal(a1:Atoms, a2:Atoms):
    info1 = AtomicStructure.from_ase(a1)
    info2 = AtomicStructure.from_ase(a2)
    return info1 == info2