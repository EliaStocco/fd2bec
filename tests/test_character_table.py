import numpy as np
import pytest
import spglib

from fd2bec.symmetry import gamma_character_table


def _c2v_rotations():
    return np.asarray(
        [
            np.eye(3, dtype=int),
            np.diag([-1, -1, 1]),
            np.diag([1, -1, 1]),
            np.diag([-1, 1, 1]),
        ]
    )


def test_gamma_character_table_has_orthogonal_irreducible_characters():
    table = gamma_character_table(_c2v_rotations())

    weights = np.asarray(table.class_sizes)
    gram = (table.characters * weights) @ table.characters.conj().T

    assert table.class_labels == ("E", "C2", "m(1)", "m(2)")
    assert table.class_sizes == (1, 1, 1, 1)
    assert table.class_representatives.shape == (4, 3, 3)
    assert table.dimensions == (1, 1, 1, 1)
    assert np.allclose(gram, 4 * np.eye(4))
    assert np.allclose(table.characters[0], 1)


def test_gamma_character_table_removes_duplicate_space_group_rotations():
    rotations = np.repeat(_c2v_rotations(), repeats=2, axis=0)

    table = gamma_character_table(rotations)

    assert len(table.characters) == 4
    assert sum(table.class_sizes) == 4


def test_gamma_character_table_supports_complex_irreps():
    threefold = np.asarray([[0, -1, 0], [1, -1, 0], [0, 0, 1]])
    rotations = np.asarray([np.eye(3, dtype=int), threefold, threefold @ threefold])

    table = gamma_character_table(rotations)

    assert table.dimensions == (1, 1, 1)
    assert np.any(np.abs(table.characters.imag) > 0.5)


@pytest.mark.parametrize(
    "hall_number",
    [
        1,
        2,
        3,
        18,
        57,
        108,
        125,
        227,
        349,
        355,
        357,
        366,
        376,
        388,
        400,
        430,
        435,
        438,
        446,
        454,
        462,
        468,
        469,
        471,
        477,
        481,
        485,
        489,
        494,
        503,
        511,
        517,
    ],
)
def test_gamma_character_table_covers_every_crystallographic_point_group(hall_number):
    symmetry = spglib.get_symmetry_from_database(hall_number)

    table = gamma_character_table(symmetry["rotations"])

    group_order = len(np.unique(symmetry["rotations"], axis=0))
    assert sum(table.class_sizes) == group_order
    assert sum(dimension**2 for dimension in table.dimensions) == group_order
