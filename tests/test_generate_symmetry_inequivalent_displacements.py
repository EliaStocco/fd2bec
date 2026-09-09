import numpy as np
import pytest
from ase import Atoms

from fd2bec.atomic import AtomicStructure
from fd2bec.displacements import (
    all_cartesian_displacements,
    all_cell_displacements,
    displacements2structures,
    random_cartesian_displacements,
    symmetry_inequivalent_displacements,
    target_tensor,
)
from fd2bec.piezoelectric import proper_piezoelectric_symmetry_basis
from fd2bec.tensor import BornCharges, ImproperPiezoelectricTensor


def test_all_registered_displacement_targets_build_from_definitions():
    expected = {
        "bec": ((4, 3, 3), (4, 3)),
        "piezo": ((3, 3, 3), (3, 3)),
        "elastic": ((3, 3, 3, 3), (3, 3)),
        "force_constants": ((4, 4, 3, 3), (4, 3)),
    }
    for name, (shape, input_shape) in expected.items():
        tensor = target_tensor(name, 4)
        assert tensor.shape == shape
        assert tensor.input_shape == input_shape


class StructureWithSymmetryModes:
    """Minimal stand-in returning predefined symmetry modes."""

    def __init__(self, modes):
        self.modes = modes

    def get_symmetry_modes(self, tensor):
        return None, np.zeros(len(self.modes)), self.modes


def test_atomic_tensor_modes_produce_atomic_displacements():
    tensor = BornCharges(data=np.zeros((2, 3, 3)))
    modes = np.zeros((1, 18))
    modes[0, 1] = 1.0

    unique, displacements = symmetry_inequivalent_displacements(
        StructureWithSymmetryModes(modes), tensor
    )

    expected = np.zeros(6)
    expected[1] = 1.0
    np.testing.assert_allclose(displacements, [np.zeros(6), expected, -expected])
    np.testing.assert_allclose(unique, displacements)


def test_precomputed_symmetry_modes_select_the_same_displacements():
    tensor = BornCharges(data=np.zeros((2, 3, 3)))
    modes = np.zeros((1, 18))
    modes[0, 1] = 1.0

    selected, candidates = symmetry_inequivalent_displacements(
        object(), tensor, component_modes=modes
    )

    expected = np.zeros(6)
    expected[1] = 1.0
    np.testing.assert_allclose(selected, [np.zeros(6), expected, -expected])
    np.testing.assert_allclose(candidates, selected)


def test_global_tensor_modes_produce_covariant_perturbations():
    tensor = ImproperPiezoelectricTensor(data=np.zeros((3, 3, 3)))
    modes = np.zeros((1, 27))
    modes[0, 3] = 1.0

    unique, perturbations = symmetry_inequivalent_displacements(
        StructureWithSymmetryModes(modes), tensor
    )

    expected = np.zeros(9)
    expected[3] = 1.0
    np.testing.assert_allclose(perturbations, [np.zeros(9), expected, -expected])
    np.testing.assert_allclose(unique, perturbations)


def test_all_cartesian_displacements_contains_reference_and_both_signs():
    displacements = all_cartesian_displacements(2)

    np.testing.assert_allclose(
        displacements,
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
    )


def test_all_cell_displacements_use_six_lower_triangular_components():
    displacements = all_cell_displacements()
    matrices = displacements.reshape((-1, 3, 3))

    assert displacements.shape == (13, 9)
    np.testing.assert_allclose(matrices, np.tril(matrices))
    assert np.count_nonzero(np.linalg.norm(displacements, axis=1)) == 12


def test_random_atomic_displacements_are_reproducible():
    first = random_cartesian_displacements(4, 6, atomic=True, seed=17)
    second = random_cartesian_displacements(4, 6, atomic=True, seed=17)

    assert first.shape == (4, 6)
    np.testing.assert_allclose(first, second)


def test_random_cell_displacements_are_lower_triangular():
    displacements = random_cartesian_displacements(4, 9, atomic=False, seed=17)
    matrices = displacements.reshape((-1, 3, 3))

    assert displacements.shape == (4, 9)
    np.testing.assert_allclose(matrices, np.tril(matrices))


PHASE_STRUCTURES = {
    "cubic": {
        "cell": np.diag([3.9775557239945183] * 3),
        "scaled_positions": [
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0],
        ],
    },
    "tetragonal": {
        "cell": np.diag([3.9630821981688404, 3.9630821981688404, 4.051482709681195]),
        "scaled_positions": [
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.015716720164655616],
            [0.0, 0.0, 0.4710470232144153],
            [0.0, 0.5, -0.018084606365200425],
            [0.5, 0.0, -0.018084606365200425],
        ],
    },
    "orthorhombic": {
        "cell": np.diag([3.956890824831104, 4.015338059204685, 4.015338059204685]),
        "scaled_positions": [
            [0.5, 0.5, 0.5],
            [0.0, -0.013176272886592092, -0.013176272886592092],
            [0.0, 0.012377573013078772, 0.5218384253342361],
            [0.0, 0.5218384253342361, 0.012377573013078772],
            [0.5, 0.014849066036499475, 0.014849066036499475],
        ],
    },
    "rhombohedral": {
        "cell": [
            [3.9968414651261877, 0.0, 0.0],
            [0.009112640060936053, 3.996831076883682, 0.0],
            [0.009112640060936053, 0.009091887233766279, 3.996820735876166],
        ],
        "scaled_positions": [
            [0.5, 0.5, 0.5],
            [0.011426470066678369, 0.011426470352913255, 0.011426471943077656],
            [-0.011333998425580718, -0.011333999795953143, 0.48209931526428623],
            [-0.011333998418420536, 0.48209931077150564, -0.01133399844365788],
            [0.48209931570821524, -0.011334002286522187, -0.01133399844365788],
        ],
    },
}


def _phase_structure(phase):
    values = PHASE_STRUCTURES[phase]
    return Atoms(
        "BaTiO3",
        cell=values["cell"],
        scaled_positions=values["scaled_positions"],
        pbc=True,
    )


def _proper_piezoelectric_design(displacements, cell, symmetry_basis):
    inverse_cell = np.linalg.inv(cell)
    blocks = []
    for displacement in displacements.reshape((-1, 3, 3)):
        gradient = (inverse_cell @ displacement).T
        strain = 0.5 * (gradient + gradient.T)
        block = np.zeros((3, 27))
        for component in range(3):
            block[component, 9 * component : 9 * component + 9] = strain.reshape(9)
        blocks.append(block @ symmetry_basis)
    return np.vstack(blocks)


@pytest.mark.parametrize(
    "phase, space_group, number_of_parameters, number_of_structures",
    [
        ("cubic", 221, 0, 1),
        ("tetragonal", 99, 3, 7),
        ("orthorhombic", 38, 5, 9),
        ("rhombohedral", 160, 4, 5),
    ],
)
def test_piezoelectric_displacements_span_every_symmetry_allowed_parameter(
    phase, space_group, number_of_parameters, number_of_structures
):
    """Regression test using the BaTiO3 phases exercised by MACE-POLAR."""
    reference = _phase_structure(phase)
    unit_cell = AtomicStructure.from_ase(reference)
    tensor = target_tensor("piezo", len(reference))

    if number_of_parameters:
        selected, candidates = symmetry_inequivalent_displacements(unit_cell, tensor)
    else:
        with pytest.warns(UserWarning, match="no symmetry-allowed components"):
            selected, candidates = symmetry_inequivalent_displacements(unit_cell, tensor)
    symmetry_basis = proper_piezoelectric_symmetry_basis(unit_cell)
    design = _proper_piezoelectric_design(selected, reference.cell.array, symmetry_basis)

    assert unit_cell.space_group == space_group
    assert symmetry_basis.shape == (27, number_of_parameters)
    assert selected.shape == (number_of_structures, 9)
    assert candidates.shape == (13, 9)
    assert np.linalg.matrix_rank(design, tol=1e-10) == number_of_parameters


def test_atomic_displacements_are_saved_on_displaced_structures():
    atoms = Atoms(
        symbols=["Si", "O"],
        cell=np.eye(3) * 5.0,
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        pbc=True,
    )
    displacement = np.array([[0.1, 0.0, 0.0, 0.0, -0.2, 0.0]])

    [displaced] = displacements2structures(atoms, displacement, atomic=True)

    np.testing.assert_allclose(
        displaced.get_positions(), atoms.get_positions() + displacement.reshape((2, 3))
    )
    np.testing.assert_allclose(displaced.arrays["displacements"], displacement.reshape((2, 3)))


def test_cell_displacements_preserve_fractional_positions():
    atoms = Atoms(
        symbols=["Si", "O"],
        cell=np.eye(3) * 5.0,
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        pbc=True,
    )
    displacement = np.zeros((1, 9))
    displacement[0, 0] = 0.1

    [displaced] = displacements2structures(atoms, displacement, atomic=False)

    np.testing.assert_allclose(
        displaced.get_scaled_positions(wrap=False), atoms.get_scaled_positions(wrap=False)
    )
    np.testing.assert_allclose(
        displaced.cell.array, atoms.cell.array + displacement.reshape((3, 3))
    )
    np.testing.assert_allclose(displaced.info["cell_displacement"], displacement.reshape((3, 3)))
