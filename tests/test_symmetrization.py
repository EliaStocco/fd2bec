import pytest
import numpy as np
from ase.io import read
from fd2bec import ATOL

# # from fd2bec.conftest import structure # noqa: F401
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
    assert atomic_structure._test_symmetry(), (
        "Error in AtomicStructure._test_symmetry() method"
    )

    # ----------------------#
    # affine
    # ----------------------#
    for name in ["positions"]:  # "REF_atomic-oxn-dipole"
        params = {"rank": 1, "affine": True, "atomic": True}

        frac_pos = atomic_structure.to_fractional(atoms.arrays[name])
        assert np.allclose(
            frac_pos @ atomic_structure.cell,
            atoms.arrays[name],
            atol=ATOL * len(atomic_structure),
        ), f"Error with fractional {name}"

        flat_pos = frac_pos.flatten()
        pos1 = append_one(flat_pos)
        H = atomic_structure.get_homogeneous_symmetry_operations()
        new_pos = H @ pos1
        assert np.allclose(pos1, new_pos, atol=ATOL * len(atomic_structure)), (
            f"Error with symmetrizer when using {name}."
        )

        S, theta, theta_real, shape = atomic_structure.get_symmetrizer(
            x=frac_pos, **params
        )
        assert np.allclose(
            remove_one(S @ theta), flat_pos, atol=ATOL * len(atomic_structure)
        ), f"Error with symmetrizer when using {name}."

    # ----------------------#
    # vectors
    # ----------------------#
    for name in ["REF_forces", "REF_atomic_dipoles"]:
        params = {"rank": 1, "affine": False, "atomic": True}
        frac_vector = atomic_structure.to_fractional(atoms.arrays[name])
        assert np.allclose(
            frac_vector @ atomic_structure.cell,
            atoms.arrays[name],
            atol=ATOL * len(atomic_structure),
        ), f"Error with fractional {name}"

        flat_vector = frac_vector.flatten()
        R = atomic_structure.get_symmetry_operations(**params)
        assert np.allclose(
            R @ flat_vector, flat_vector, atol=ATOL * len(atomic_structure)
        ), f"Error with symmetrizer when using {name}."

        S, theta, theta_real, shape = atomic_structure.get_symmetrizer(
            x=frac_vector, **params
        )
        assert np.allclose(S @ theta, flat_vector, atol=ATOL * len(atomic_structure)), (
            f"Error with symmetrizer when using {name}."
        )

    # ----------------------#
    # Born Charges
    # ----------------------#
    for name in ["REF_BEC"]:
        params = {"rank": 2, "affine": False, "atomic": True}
        tensor = atoms.arrays[name].reshape((-1, 3, 3))
        frac_vector = atomic_structure.to_fractional(tensor, rank=2)
        tmp = atomic_structure.to_cartesian(frac_vector, rank=2)
        assert np.allclose(tmp, tensor, atol=ATOL * len(atomic_structure)), (
            f"Error with fractional {name}"
        )

        flat_vector = frac_vector.flatten()
        R = atomic_structure.get_symmetry_operations(**params)
        assert np.allclose(
            R @ flat_vector, flat_vector, atol=ATOL * len(atomic_structure)
        ), f"Error with symmetrizer when using {name}."

        S, theta, theta_real, shape = atomic_structure.get_symmetrizer(
            x=frac_vector, **params
        )
        assert np.allclose(S @ theta, flat_vector, atol=ATOL * len(atomic_structure)), (
            f"Error with symmetrizer when using {name}."
        )


if __name__ == "__main__":
    pytest.main([__file__])
