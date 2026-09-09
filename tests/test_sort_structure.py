import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from fd2bec.atomic import AtomicStructure
from fd2bec.cli.structures import sort_structure
from fd2bec.structure_alignment import sort_atoms_like


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


def test_sort_atoms_like_aligns_before_using_nearest_periodic_images():
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
        [[0.0, 0.0, 0.0], [0.2, 0.3, 0.5]],
    )


def test_sort_atoms_like_finds_anchor_after_translation_and_shuffle():
    reference_positions = np.array(
        [
            [0.10, 0.20, 0.30],
            [0.32, 0.43, 0.54],
            [0.71, 0.64, 0.82],
            [0.88, 0.13, 0.47],
        ]
    )
    reference = Atoms(
        symbols=["O", "Si", "O", "C"],
        cell=np.eye(3) * 5.0,
        scaled_positions=reference_positions,
        pbc=True,
    )
    order = np.array([2, 3, 0, 1])
    shift = np.array([0.27, -0.31, 0.19])
    candidate = Atoms(
        symbols=np.asarray(reference.get_chemical_symbols())[order],
        cell=reference.cell,
        scaled_positions=(reference_positions[order] + shift) % 1.0,
        pbc=True,
    )
    candidate.set_array("original_index", order)

    ordered = sort_atoms_like(reference, candidate, atol=1e-8)

    np.testing.assert_allclose(ordered.get_scaled_positions(wrap=False), reference_positions)
    np.testing.assert_array_equal(ordered.arrays["original_index"], np.arange(4))


def test_sort_atoms_like_aligns_translated_molecule_before_sorting():
    reference = Atoms(
        symbols=["H", "O", "H"],
        positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [-0.2, 0.7, 0.0]],
    )
    order = np.array([2, 0, 1])
    candidate = reference[order]
    candidate.translate([12.5, -8.0, 3.0])

    ordered = sort_atoms_like(reference, candidate, atol=1e-8)

    assert ordered.get_chemical_symbols() == reference.get_chemical_symbols()
    np.testing.assert_allclose(ordered.positions, reference.positions)


def test_sort_atoms_like_requires_an_ase_standard_reference_cell():
    reference = Atoms(
        symbols=["Si"],
        cell=[[0.0, 3.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 5.0]],
        scaled_positions=[[0.0, 0.0, 0.0]],
        pbc=True,
    )
    candidate = reference.copy()

    with pytest.raises(ValueError, match="rotate_cell"):
        sort_atoms_like(reference, candidate, atol=1e-8)


def test_sort_structure_cli_aligns_then_sorts(tmp_path):
    reference_path = tmp_path / "reference.extxyz"
    candidate_path = tmp_path / "candidate.extxyz"
    output_path = tmp_path / "sorted.extxyz"
    reference = Atoms(
        symbols=["Na", "Cl"],
        cell=np.eye(3) * 4.0,
        scaled_positions=[[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]],
        pbc=True,
    )
    candidate = Atoms(
        symbols=["Cl", "Na"],
        cell=reference.cell,
        scaled_positions=[[0.83, 0.51, 0.99], [0.33, 0.01, 0.49]],
        pbc=True,
    )
    write(reference_path, reference)
    write(candidate_path, candidate)
    args = sort_structure.prepare_args(sort_structure.description).parse_args(
        [
            "-r",
            str(reference_path),
            "-i",
            str(candidate_path),
            "-o",
            str(output_path),
            "--atol",
            "1e-8",
        ]
    )

    sort_structure.main.__wrapped__(args)

    ordered = read(output_path)
    assert ordered.get_chemical_symbols() == reference.get_chemical_symbols()
    np.testing.assert_allclose(
        ordered.get_scaled_positions(wrap=False),
        reference.get_scaled_positions(wrap=False),
        atol=1e-8,
    )
