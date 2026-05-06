from pathlib import Path
import pytest
import numpy as np
from fd2bec import SYMPREC
from fd2bec.io import read
from fd2bec.atomic import AtomicStructure
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

DATA_DIR = Path(__file__).parent / "molecules/point_group_dataset"


# Collect all .xyz files
xyz_files = sorted(DATA_DIR.glob("*.xyz"))


@pytest.mark.parametrize("filepath", xyz_files)
def test_translations(filepath:Path):
    assert filepath.exists()
    atoms = read(filepath)
    atomic_structure = AtomicStructure.from_ase(atoms)
    
    pmg_mol = Molecule(
        atoms.get_chemical_symbols(),
        atoms.get_positions()
    )

    pga = PointGroupAnalyzer(pmg_mol,tolerance=SYMPREC,eigen_tolerance=SYMPREC,matrix_tolerance=SYMPREC)

    assert pga.sch_symbol is not None
    
    S = pga.get_symmetry_operations()
    R = np.asarray([s.rotation_matrix for s in S])
    T = np.asarray([s.translation_vector for s in S])
    x = atoms.get_positions()
    for _,(r,t) in enumerate(zip(R,T)):
        x_test = x @ r.T + t
        new_structure = atomic_structure.clone(positions=x_test)
        if not atomic_structure.is_equal_to(new_structure):
            atomic_structure.is_equal_to(new_structure)
            raise ValueError("Point group operation does not preserve the structure.")
        pass
    

if __name__ == "__main__":
    pytest.main([__file__])
