import numpy as np
import pytest
from ase import Atoms

from fd2bec.cli import KEYWORDS
from fd2bec.cli.ml.mace_polar_dPdR import (
    build_displaced_structures,
    evaluate_dipoles,
)
from fd2bec.cli.ml.mace_polar_dPdS import evaluate_polarizations
from fd2bec.piezoelectric import (
    build_strained_structures,
    evaluate_piezoelectric_from_structures,
)


class FakePolarCalculator:
    def __init__(self):
        self.results = {}

    def get_potential_energy(self, atoms=None, force_consistent=False):
        del force_consistent
        positions = atoms.get_positions()
        self.results["dipole"] = positions.sum(axis=0)
        return float(np.square(positions).sum())


def water(pbc=False):
    return Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.7586, 0.0, 0.5043], [-0.7586, 0.0, 0.5043]],
        pbc=pbc,
    )


def test_build_and_evaluate_dipole_dataset():
    reference = water()
    structures = build_displaced_structures(reference, amplitude=0.01)
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return FakePolarCalculator()

    evaluated = evaluate_dipoles(
        reference,
        structures,
        model="checkpoint.model",
        device="cpu",
        default_dtype="float64",
        charge=-1,
        spin=2,
        calculator_factory=factory,
    )

    assert len(evaluated) == 2 * 3 * len(reference) + 1
    assert factory_calls == [
        {"model": "checkpoint.model", "device": "cpu", "default_dtype": "float64"}
    ]
    for atoms in evaluated:
        assert atoms.calc is None
        assert atoms.info["charge"] == -1
        assert atoms.info["spin"] == 2
        assert np.allclose(atoms.info["external_field"], 0)
        assert np.allclose(
            atoms.arrays[KEYWORDS["displacements"]],
            atoms.get_positions() - reference.get_positions(),
        )
        assert np.allclose(atoms.info[KEYWORDS["dipole"]], atoms.get_positions().sum(axis=0))


def test_input_electronic_state_is_preserved():
    reference = water()
    reference.info.update(charge=1, spin=3)
    structures = build_displaced_structures(reference, amplitude=0.01)

    evaluated = evaluate_dipoles(
        reference,
        structures[:1],
        model="checkpoint.model",
        calculator_factory=lambda **kwargs: FakePolarCalculator(),
    )

    assert evaluated[0].info["charge"] == 1
    assert evaluated[0].info["spin"] == 3


def test_periodic_structures_are_rejected():
    reference = water(pbc=True)
    with pytest.raises(ValueError, match="only meaningful for non-periodic"):
        evaluate_dipoles(
            reference,
            [reference.copy()],
            model="checkpoint.model",
            calculator_factory=lambda **kwargs: FakePolarCalculator(),
        )


def test_non_positive_displacement_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        build_displaced_structures(water(), amplitude=0)


def test_periodic_mace_polar_piezoelectric_adapter():
    reference = Atoms(
        "Si2",
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=np.diag([4.0, 5.0, 6.0]),
        pbc=True,
    )
    structures = build_strained_structures(reference, amplitude=1e-3)

    evaluated = evaluate_polarizations(
        reference,
        structures,
        model="checkpoint.model",
        calculator_factory=lambda **kwargs: FakePolarCalculator(),
    )

    assert len(evaluated) == 13
    for atoms in evaluated:
        expected_dipole = atoms.get_positions().sum(axis=0)
        np.testing.assert_allclose(atoms.info[KEYWORDS["dipole"]], expected_dipole)
        np.testing.assert_allclose(
            atoms.info[KEYWORDS["polarization"]], expected_dipole / atoms.get_volume()
        )

    result = evaluate_piezoelectric_from_structures(evaluated, reference)
    assert np.max(np.abs(result.proper_voigt)) < 1e-6
