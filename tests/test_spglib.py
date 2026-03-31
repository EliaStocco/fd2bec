# tests/test_structures.py
import numpy as np
from ase.io import read
import pytest

# Import the structures_path fixture from conftest
from fd2bec import SYMPREC
from fd2bec.conftest import structures_dir
from fd2bec.tools import ase2spglib_dataset, wrap
from fd2bec.atomic import AtomicStructure, structures_equal

def test_spacegroup():
    """
    Test that BaTiO3 structures have the correct number of atoms
    and all have space group 99.
    """
    for n in [1,2,4]:
        file_path = structures_dir / f"BaTiO3.{n}x{n}x{n}.extxyz"
        atoms = read(file_path, index=0)

        # Check number of atoms
        factor = np.power(n, 3)
        expected_atoms = factor * 5
        actual_atoms = atoms.get_global_number_of_atoms()
        assert actual_atoms == expected_atoms, f"{file_path} has {actual_atoms} atoms, expected {expected_atoms}"

        # Prepare cell and positions for spglib
        dataset = ase2spglib_dataset(atoms,symprec=SYMPREC)
        assert dataset.number == 99, f"{file_path} detected space group {dataset.number}, expected 99"
            
def test_number_operations():
    """
    Test that BaTiO3 structures have the correct number of atoms
    and all have space group 99.
    """
    first = True
    first_No = 0
    for n in [1,2,4]:
        file_path = structures_dir / f"BaTiO3.{n}x{n}x{n}.extxyz"
        atoms = read(file_path, index=0)

        # Check number of atoms
        factor = np.power(n, 3)

        # Prepare cell and positions for spglib
        dataset = ase2spglib_dataset(atoms,symprec=SYMPREC)
        
        if first:
            first_No = dataset.rotations.shape[0]
            first = False
        else:
            # Supercells have fewer translational symmetries then the unit cell.
            # The missing translations become rotations.
            # The number of symmetry elements is preserved.
            assert dataset.rotations.shape[0] == factor * first_No
            
def test_structures_spacegroup_positions():
    """
    Test that BaTiO3 structures have the correct number of atoms,
    correct space group, and that atomic positions are symmetric.
    """
    for n in [1, 2, 4]:
        file_path = structures_dir / f"BaTiO3.{n}x{n}x{n}.extxyz"
        atoms = read(file_path, index=0)
        N = atoms.get_global_number_of_atoms()

        # Get symmetry dataset from spglib
        dataset = ase2spglib_dataset(atoms, symprec=SYMPREC)
        new_atoms = atoms.copy()
        
        # Check symmetry for each atom under each symmetry operation
        frac_pos = atoms.get_scaled_positions()
        new_frac = frac_pos.copy()
        for R, t in zip(dataset.rotations, dataset.translations):
            new_frac = (new_frac @ R + t[None,:])
            new_atoms.set_scaled_positions(new_frac)
            if not structures_equal(atoms,new_atoms):
                raise ValueError("ops")
            
        diff = new_frac - frac_pos
        diff = wrap(diff)        
        assert np.allclose(diff,0,atol=1e-5*N), "Ops"
        

if __name__ == "__main__":
    pytest.main([__file__])