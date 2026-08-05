import numpy as np
import pytest
from ase import Atoms

from fd2bec.atomic import AtomicStructure
from fd2bec.cli.general.sort_structure import sort_atoms_like


def test_reordered_like_matches_reference_order_for_periodic_structure():
    reference = AtomicStructure(
        symbols=["Si", "O", "O"],
        cell=np.eye(3) * 5.0,
        frac_pos=np.array([[0.00, 0.00, 0.00], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]),
        check=False,
    )
    candidate = AtomicStructure(
        symbols=["O", "Si", "O"],
        cell=np.eye(3) * 5.0,
        frac_pos=np.array([[0.751, 0.75, 0.75], [0.999, 0.00, 0.00], [0.25, 0.25, 0.25]]),
        check=False,
    )

    ordered = candidate.reordered_like(reference, atol=0.01)

    assert ordered.symbols == reference.symbols
    assert reference.is_equal_to(ordered, atol=0.01)


def test_sort_atoms_like_reorders_ase_arrays_with_the_atoms():
    reference = Atoms(
        symbols=["H", "O", "H"],
        positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [-0.2, 0.7, 0.0]],
    )
    candidate = Atoms(
        symbols=["H", "H", "O"],
        positions=[[-0.199, 0.7, 0.0], [0.001, 0.0, 0.0], [0.8, 0.0, 0.0]],
    )
    candidate.set_array("label", np.array([30, 10, 20]))
    candidate.info["source"] = "candidate"

    ordered = sort_atoms_like(reference, candidate, atol=0.01)

    assert ordered.get_chemical_symbols() == reference.get_chemical_symbols()
    assert np.array_equal(ordered.arrays["label"], [10, 20, 30])
    assert ordered.info == {"source": "candidate"}


def test_sort_atoms_like_uses_nearest_periodic_image_of_reference():
    reference = Atoms(
        symbols=["O", "Si"],
        cell=np.eye(3) * 5.0,
        scaled_positions=[[0.0, 0.0, 0.0], [0.2, 0.3, 0.4]],
        pbc=True,
    )
    candidate = Atoms(
        symbols=["Si", "O"],
        cell=np.eye(3) * 5.0,
        scaled_positions=[[0.2, 0.3, 0.4], [0.0, 0.0, 0.9]],
        pbc=True,
    )

    ordered = sort_atoms_like(reference, candidate, atol=0.11)

    np.testing.assert_allclose(
        ordered.get_scaled_positions(wrap=False),
        [[0.0, 0.0, -0.1], [0.2, 0.3, 0.4]],
    )


def test_sort_atoms_like_requires_an_ase_standard_reference_cell():
    reference = Atoms(
        symbols=["Si"],
        cell=[[0.0, 3.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 5.0]],
        scaled_positions=[[0.0, 0.0, 0.0]],
        pbc=True,
    )
    candidate = reference.copy()

    with pytest.raises(ValueError, match="rotate_cell"):
        sort_atoms_like(reference, candidate)
