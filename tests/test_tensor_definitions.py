import json

import numpy as np
import pytest

from fd2bec.tensor import (
    BORN_CHARGES,
    DIPOLE,
    ELASTIC_STIFFNESS,
    FORCE_CONSTANTS,
    POSITIONS,
    STRAIN,
    STRESS,
    VOLUME,
    BornCharges,
    ForceConstants,
    MAPPING,
    Position,
    Tensor,
    derivative,
    divide_by,
    evaluate_scalar,
)


def test_definitions_are_plain_json_and_derivatives_do_not_mutate_inputs():
    positions_before = json.loads(json.dumps(POSITIONS))
    derived = derivative(DIPOLE, POSITIONS, name="test_bec")

    json.dumps(derived)
    assert POSITIONS == positions_before
    assert [axis["type"] for axis in derived["axes"]] == ["atomic", "cartesian", "cartesian"]
    assert [axis["role"] for axis in derived["axes"]] == ["input", "output", "input"]
    assert derived["axes"][-1]["variance"] == "covariant"
    assert not any(axis.get("affine", False) for axis in derived["axes"])


def test_repeated_derivative_has_two_atomic_dimensions():
    assert [axis["type"] for axis in FORCE_CONSTANTS["axes"]] == [
        "atomic",
        "atomic",
        "cartesian",
        "cartesian",
    ]
    assert ForceConstants.template(3).shape == (3, 3, 3, 3)


def test_force_constants_use_one_canonical_registry_name():
    assert MAPPING["force_constants"] is ForceConstants
    assert not {"force-constant", "force-constants", "hessian"} & MAPPING.keys()


def test_template_accepts_an_explicit_basis():
    tensor = BornCharges.template(2, basis="fractional")

    assert tensor.basis == "fractional"


def test_tensor_instance_round_trip_and_role_shapes():
    tensor = BornCharges(data=np.arange(36.0).reshape(4, 3, 3), cell=np.eye(3))
    restored = Tensor.from_json(tensor.to_json())

    assert restored.definition == tensor.definition
    assert restored.basis == tensor.basis
    np.testing.assert_allclose(restored.data, tensor.data)
    np.testing.assert_allclose(restored.cell, tensor.cell)
    assert tensor.input_shape == (4, 3)
    assert tensor.output_shape == (3,)


def test_tensor_repr_is_compact_and_identifies_its_definition():
    tensor = BornCharges(data=np.zeros((4, 3, 3)))

    assert repr(tensor) == "BornCharges(definition='born_charges', shape=(4, 3, 3), basis='cartesian')"


def test_tensor_symmetrize_applies_a_projection_without_mutating_the_tensor():
    tensor = BornCharges(data=np.arange(9.0).reshape(1, 3, 3))
    projection = np.diag([1.0, 0.0, 1.0] * 3)

    symmetrized = tensor.symmetrize(projection)

    np.testing.assert_allclose(symmetrized.flatten_full(), projection @ tensor.flatten_full())
    np.testing.assert_allclose(tensor.data, np.arange(9.0).reshape(1, 3, 3))


def test_tensor_symmetrize_uses_homogeneous_projection_for_affine_tensor():
    tensor = Position(data=np.array([[4.0, 5.0, 6.0]]))
    projection = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    symmetrized = tensor.symmetrize(projection)

    np.testing.assert_allclose(symmetrized.data, [[1.0, 2.0, 3.0]])


def test_tensor_symmetrize_treats_a_scalar_as_one_component():
    tensor = Tensor(VOLUME, data=np.array(4.0))

    symmetrized = tensor.symmetrize(np.ones((1, 1)))

    assert symmetrized.data == 4.0


def test_scalar_volume_and_invalid_scalar_operand():
    assert np.allclose(evaluate_scalar(VOLUME, np.diag([2.0, 3.0, -4.0])),24.0)
    with pytest.raises(ValueError):
        divide_by(DIPOLE, STRAIN, name="invalid")


def test_invalid_axis_size_and_missing_atomic_template_size():
    with pytest.warns(UserWarning, match=r"reshaped.*\(2, 3, 3\)"):
        tensor = BornCharges(data=np.zeros((3, 3, 2)))

    assert tensor.shape == (2, 3, 3)
    with pytest.raises(ValueError):
        BornCharges.template()


def test_data_is_reshaped_when_its_size_matches_the_explicit_shape():
    with pytest.warns(UserWarning, match=r"reshaped.*\(4, 3, 3\)"):
        tensor = BornCharges(data=np.zeros(36))

    assert tensor.shape == (4, 3, 3)


def test_invalid_data_does_not_warn_when_no_explicit_shape_matches():
    with pytest.raises(ValueError, match="needs at least"):
        BornCharges(data=np.zeros(35))


def test_stress_and_elastic_definitions_have_explicit_roles():
    assert all(axis["role"] == "input" for axis in STRESS["axes"])
    assert [axis["role"] for axis in ELASTIC_STIFFNESS["axes"]] == [
        "output",
        "output",
        "input",
        "input",
    ]
