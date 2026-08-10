import numpy as np
from ase import Atoms

from fd2bec.cli.displacements.generate_displacements import (
    TENSOR_TARGETS,
    _target_tensor,
    all_cartesian_displacements,
    all_cell_displacements,
    atomic_structure2unique_displacements,
    displacements2structures,
    proper_piezoelectric_cell_displacements,
    random_cartesian_displacements,
)
from fd2bec.tensor import BornCharges, ImproperPiezoelectricTensor


def test_all_registered_displacement_targets_build_from_definitions():
    expected = {
        "bec": ((4, 3, 3), (4, 3)),
        "piezo": ((3, 3, 3), (3, 3)),
        "elastic": ((3, 3, 3, 3), (3, 3)),
        "force-constants": ((4, 4, 3, 3), (4, 3)),
    }
    for name, (shape, input_shape) in expected.items():
        tensor = _target_tensor(name, 4)
        assert name in TENSOR_TARGETS
        assert tensor.shape == shape
        assert tensor.input_shape == input_shape


class SymmetrizedStructure:
    """Minimal stand-in returning predefined symmetry modes."""

    def __init__(self, modes):
        self.modes = modes

    def get_symmetrizer(self, tensor):
        return None, np.zeros(len(self.modes)), self.modes


def test_atomic_tensor_modes_produce_atomic_displacements():
    tensor = BornCharges(data=np.zeros((2, 3, 3)))
    modes = np.zeros((1, 18))
    modes[0, 1] = 1.0

    unique, displacements = atomic_structure2unique_displacements(
        SymmetrizedStructure(modes), tensor
    )

    expected = np.zeros(6)
    expected[1] = 1.0
    np.testing.assert_allclose(displacements, [np.zeros(6), expected, -expected])
    np.testing.assert_allclose(unique, displacements)


def test_global_tensor_modes_produce_covariant_perturbations():
    tensor = ImproperPiezoelectricTensor(data=np.zeros((3, 3, 3)))
    modes = np.zeros((1, 27))
    modes[0, 3] = 1.0

    unique, perturbations = atomic_structure2unique_displacements(
        SymmetrizedStructure(modes), tensor
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
