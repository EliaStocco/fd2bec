import numpy as np

from fd2bec.cli import KEYWORDS
from fd2bec.cli.structures.check_relaxation_symmetry import prepare_args
from fd2bec.relaxation import project_relaxation_tensor
from fd2bec.tensor import Forces, Stress


def test_relaxation_symmetry_cli_uses_shared_data_field_defaults():
    parser = prepare_args("test")
    args = parser.parse_args(["-i", "structure.extxyz"])

    assert args.forces_key == KEYWORDS["forces"]
    assert args.stress_key == KEYWORDS["stress"]
    assert args.index == 0
    assert args.tolerance == 1e-5
    assert args.precision == 4


def test_symmetry_projection_splits_cubic_force_and_stress_components():
    from ase import Atoms

    from fd2bec.atomic import AtomicStructure

    atoms = Atoms("H", cell=np.eye(3), pbc=True)
    structure = AtomicStructure.from_ase(atoms)
    forces = Forces(data=np.asarray([[1.0, -2.0, 3.0]]), cell=atoms.cell)
    stress = Stress(
        data=np.asarray(
            [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 7.0]]
        ),
        cell=atoms.cell,
    )

    force_report = project_relaxation_tensor(structure, forces)
    stress_report = project_relaxation_tensor(structure, stress)

    np.testing.assert_allclose(force_report.symmetrized.data, 0.0)
    np.testing.assert_allclose(force_report.symmetry_breaking.data, forces.data)
    np.testing.assert_allclose(stress_report.symmetrized.data, 4.0 * np.eye(3))
    np.testing.assert_allclose(
        stress_report.symmetry_breaking.data, stress.data - 4.0 * np.eye(3)
    )
