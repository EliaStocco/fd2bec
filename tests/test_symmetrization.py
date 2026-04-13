import pytest
import numpy as np
from ase.io import read
from fd2bec import ATOL
from fd2bec.conftest import structure
from fd2bec.atomic import AtomicStructure
from fd2bec.mathematics import append_one, remove_one


def test_symmetrizer(structure):
    """
    Test that the structures have the correct number of atoms,
    correct space group, and that atomic positions are symmetric.
    """
    n, file_path = structure

    atoms = read(file_path, index=0)
    Natoms = atoms.get_global_number_of_atoms()
    if Natoms > 200:
        return 
    
    atomic_structure = AtomicStructure.from_ase(atoms)
    assert atomic_structure._test_symmetry(), "Error in AtomicStructure._test_symmetry() method"
    pos = atomic_structure.frac_pos.flatten()
    
    #----------------------#
    # Positions
    #----------------------#
    # affine
    R,T = atomic_structure.get_affine_symmetry_operations(debug=True)
    new_pos = R @ pos + T 
    assert np.allclose(pos,new_pos,atol=ATOL*len(atomic_structure)), "Error with the affine symmetry operations"
    
    # homogeneous
    pos1 = append_one(pos)
    H = atomic_structure.get_homogeneous_symmetry_operations(debug=True)
    new_pos = H @ pos1
    assert np.allclose(pos1,new_pos,atol=ATOL*len(atomic_structure)), "Error with the homogeneous symmetry operations"
    
    # symmetrizer
    S, theta, theta_real = atomic_structure.get_symmetrizer(what='positions',debug=True)
    assert len(theta) == 5, "there is something wrong"
    assert np.allclose(remove_one(S @ theta), pos), "Error with the positions symmetrizer."
    
    #----------------------#
    # affine
    #----------------------#
    for name in ["positions"]: # "REF_atomic-oxn-dipole" 
        frac_forces = atomic_structure.to_fractional(atoms.arrays[name])
        assert np.allclose(frac_forces @ atomic_structure.cell, atoms.arrays[name],atol=ATOL*len(atomic_structure)), f"Error with fractional {name}"
        
        forces = frac_forces.flatten()
        pos1 = append_one(forces)
        H = atomic_structure.get_homogeneous_symmetry_operations(x=frac_forces,debug=True)
        new_pos = H @ pos1
        assert np.allclose(pos1, new_pos,atol=ATOL*len(atomic_structure)), f"Error with symmetrizer when using {name}."
        
        S, theta, theta_real = atomic_structure.get_symmetrizer(what='positions',x=frac_forces,debug=True)
        assert np.allclose(remove_one(S @ theta), forces,atol=ATOL*len(atomic_structure)), f"Error with symmetrizer when using {name}."
    

    #----------------------#
    # vectors
    #----------------------#   
    for name in ["REF_forces","REF_atomic_dipoles"]:
        frac_forces = atomic_structure.to_fractional(atoms.arrays[name])
        assert np.allclose(frac_forces @ atomic_structure.cell, atoms.arrays[name],atol=ATOL*len(atomic_structure)), f"Error with fractional {name}"
        
        forces = frac_forces.flatten()
        R = atomic_structure.get_symmetry_operations(x=frac_forces,debug=True)
        assert np.allclose(R @ forces, forces,atol=ATOL*len(atomic_structure)), f"Error with symmetrizer when using {name}."
        
        S, theta, theta_real = atomic_structure.get_symmetrizer(what='vector',x=frac_forces,debug=True)
        assert np.allclose(S @ theta, forces,atol=ATOL*len(atomic_structure)), f"Error with symmetrizer when using {name}."
    
    #----------------------#
    # Born Charges
    #----------------------#   
    bec = atoms.arrays["REF_BEC"].reshape((Natoms,3,3))
    frac_bec = atomic_structure.to_fractional(bec)
    
    bec = frac_bec.flatten()
    R = atomic_structure.get_symmetry_operations(x=frac_bec,debug=True)
    assert np.allclose(R @ bec, bec,atol=ATOL*len(atomic_structure)), "Error with the forces symmetrizer."
    
    S, theta, theta_real = atomic_structure.get_symmetrizer(what='vector',x=frac_bec,debug=True)
    assert np.allclose(S @ theta, bec,atol=ATOL*len(atomic_structure)), "Error with the forces symmetrizer."
    
    pass


if __name__ == "__main__":
    pytest.main([__file__])