import numpy as np
from ase import Atoms
from dataclasses import dataclass
from fd2bec.hungarian import equal_rows_hungarian   

@dataclass
class AtomicStructure:
    species : set[str]
    cellpar : np.ndarray
    pos: dict[str,set[list[float]]]
    
    def __eq__(self:'AtomicStructure', other:'AtomicStructure')->bool:
        if not self.species == other.species:
            return False
        if not np.allclose(self.cellpar,other.cellpar):
            return False
        for s in self.species:
            a = self.pos[s]
            b = other.pos[s]
            if not equal_rows_hungarian(a,b):
                return False
        return True
    
    @classmethod
    def from_ase(cls,atoms:Atoms)->'AtomicStructure':
        symbols = atoms.get_chemical_symbols()
        species = set(symbols)
        frac_pos = atoms.get_scaled_positions()
        pos = { s:frac_pos[atoms.symbols == s,:] for s in species }
        return cls(species=species,pos=pos,cellpar=atoms.cell.cellpar())
    
def structures_equal(a1:Atoms, a2:Atoms):
    info1 = AtomicStructure.from_ase(a1)
    info2 = AtomicStructure.from_ase(a2)
    return info1 == info2