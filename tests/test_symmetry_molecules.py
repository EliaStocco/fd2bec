from pathlib import Path
import pytest
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

    pga = PointGroupAnalyzer(pmg_mol)

    assert pga.sch_symbol is not None
    

if __name__ == "__main__":
    pytest.main([__file__])
