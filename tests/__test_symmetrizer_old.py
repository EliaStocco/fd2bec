import numpy as np
import pytest
from fd2bec.io import read

from fd2bec import ATOL
from fd2bec.atomic import AtomicStructure
from fd2bec.mathematics import append_one, remove_one

# # from fd2bec.conftest import structure # noqa: F401
from fd2bec.tensor import AtomicVector, BornCharge, Force, Position


def test_symmetrizer(structure):
    """
    Test that the structures have the correct number of atoms,
    correct space group, and that atomic positions are symmetric.
    """
    n, file_path = structure

    atoms = read(file_path, index=0)
    Natoms = atoms.get_global_number_of_atoms()
    if Natoms > 200:
        pytest.skip("Too many atoms.")

    atomic_structure = AtomicStructure.from_ase(atoms)
    atomic_structure._test_symmetry()

    # ----------------------#
    # affine
    # ----------------------#
    # for name in ["positions"]:  # "REF_atomic-oxn-dipole"

    # params = {"rank": 1, "affine": True, "atomic": True}

    tensor = Position(data=atoms.arrays["positions"], cell=atoms.cell)
    frac_pos = atomic_structure.to(basis="fractional", tensor=tensor).data
    assert np.allclose(
        frac_pos @ atomic_structure.cell,
        atoms.arrays["positions"],
        atol=ATOL * len(atomic_structure),
    ), "Error with fractional 'positions'"

    flat_pos = frac_pos.flatten()
    pos1 = append_one(flat_pos)
    H = atomic_structure.homogeneous_symmetry_operations
    new_pos = H @ pos1
    assert np.allclose(
        pos1, new_pos, atol=ATOL * len(atomic_structure)
    ), "Error with symmetrizer when using 'positions'."

    x = tensor.to("fractional")
    S, theta, theta_real = atomic_structure.get_symmetrizer(tensor=x)
    assert np.allclose(
        remove_one(S @ theta), flat_pos, atol=ATOL * len(atomic_structure)
    ), "Error with symmetrizer when using 'positions'."

    # ----------------------#
    # vectors
    # ----------------------#
    for name, classname in [("REF_forces", Force), ("REF_atomic_dipoles", AtomicVector)]:

        array = atoms.arrays[name]
        # params = {"rank": 1, "affine": False, "atomic": True}
        tensor = classname(data=array)
        frac_vector = atomic_structure.to(basis="fractional", tensor=tensor)  # .data
        # assert np.allclose(
        #     frac_vector @ atomic_structure.cell,
        #     array,
        #     atol=ATOL * len(atomic_structure),
        # ), f"Error with fractional {name}"

        # flat_vector = frac_vector.flatten()
        # R = atomic_structure.get_tensor_symmetry_operations(**params)
        # assert np.allclose(
        #     R @ flat_vector, flat_vector, atol=ATOL * len(atomic_structure)
        # ), f"Error with symmetrizer when using {name}."

        R = atomic_structure.get_tensor_symmetry_operations(tensor=frac_vector)
        S, theta, theta_real = atomic_structure.get_symmetrizer(tensor=frac_vector)
        # assert np.allclose(
        #     S @ theta, flat_vector, atol=ATOL * len(atomic_structure)
        # ), f"Error with symmetrizer when using {name}."

    # ----------------------#
    # Born Charges
    # ----------------------#
    # for name in ["REF_BEC"]:
    # params = {"rank": 2, "affine": False, "atomic": True}

    tensor = BornCharge(data=atoms.arrays["REF_BEC"].reshape((-1, 3, 3)))
    frac_bec = atomic_structure.to(basis="fractional", tensor=tensor)
    tmp = atomic_structure.to(basis="cartesian", tensor=frac_bec).data
    assert np.allclose(
        tmp, tensor, atol=ATOL * len(atomic_structure)
    ), "Error with fractional 'REF_BEC'"

    # frac_bec = frac_bec.data.flatten()
    R = atomic_structure.get_tensor_symmetry_operations(tensor=frac_bec)
    S, theta, theta_real = atomic_structure.get_symmetrizer(tensor=frac_vector)

    # assert np.allclose(
    #     R @ flat_vector, flat_vector, atol=ATOL * len(atomic_structure)
    # ), "Error with symmetrizer when using 'REF_BEC'."

    # S, theta, theta_real = atomic_structure.get_symmetrizer(tensor=frac_bec, **params)
    # assert np.allclose(
    #     S @ theta, flat_vector, atol=ATOL * len(atomic_structure)
    # ), "Error with symmetrizer when using 'REF_BEC'."


if __name__ == "__main__":
    pytest.main([__file__])
