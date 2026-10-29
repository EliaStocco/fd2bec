import numpy as np
import pytest

from fd2bec.mathematics import zero_dipole_fractional_shifts


def test_zero_dipole_fractional_shifts_cancel_the_charge_weighted_coordinates():
    oxidation_numbers = np.array([2.0, -1.0, -1.0])
    fractional_positions = np.array(
        [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    shifts = zero_dipole_fractional_shifts(oxidation_numbers, fractional_positions)

    assert np.issubdtype(shifts.dtype, np.integer)
    np.testing.assert_array_equal(oxidation_numbers @ shifts, -(oxidation_numbers @ fractional_positions))
    np.testing.assert_allclose(oxidation_numbers @ (fractional_positions + shifts), 0.0, atol=1e-12)


def test_zero_dipole_fractional_shifts_require_a_nonzero_oxidation_number():
    with pytest.raises(ValueError, match="non-zero"):
        zero_dipole_fractional_shifts([0.0, 0.0], np.zeros((2, 3)))


def test_zero_dipole_fractional_shifts_reject_noninteger_quanta():
    with pytest.raises(ValueError, match="non-integer dipole quanta"):
        zero_dipole_fractional_shifts([1.0, -1.0], [[0.25, 0.0, 0.0], [0.0, 0.0, 0.0]])


def test_zero_dipole_fractional_shifts_require_quanta_divisible_by_charge_gcd():
    with pytest.raises(ValueError, match="gcd"):
        zero_dipole_fractional_shifts([4.0, 2.0, -2.0], [[0.25, 0.0, 0.0]] * 3)
