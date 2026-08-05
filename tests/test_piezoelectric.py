import numpy as np
import pytest
from ase import Atoms

from fd2bec.piezoelectric import (
    E_PER_ANGSTROM2_TO_C_PER_M2,
    VOIGT_PAIRS,
    apply_strains,
    build_strained_structures,
    canonical_piezoelectric_modes,
    evaluate_dipole_lattice_derivative,
    evaluate_piezoelectric_from_structures,
    evaluate_piezoelectric_tensors,
    evaluate_proper_piezoelectric_direct,
    generate_strains,
    piezoelectric_symbolic_matrix,
    piezoelectric_to_voigt,
    proper_piezoelectric_symmetry_basis,
    proper_piezoelectric_tensor,
    strain_to_voigt,
    voigt_to_piezoelectric,
    voigt_to_strain,
)


def periodic_reference():
    return Atoms(
        "Si2",
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=np.diag([4.0, 5.0, 6.0]),
        pbc=True,
    )


def synthetic_response(strains):
    reference_polarization = np.array([0.07, -0.04, 0.11])
    proper_voigt = np.array(
        [
            [0.20, -0.10, 0.03, 0.04, -0.02, 0.06],
            [-0.05, 0.12, 0.08, -0.03, 0.07, 0.01],
            [0.09, 0.02, -0.15, 0.05, 0.04, -0.08],
        ]
    )
    proper = voigt_to_piezoelectric(proper_voigt)
    correction = proper_piezoelectric_tensor(np.zeros((3, 3, 3)), reference_polarization)
    improper = proper - correction
    polarizations = reference_polarization + np.einsum("ijk,njk->ni", improper, strains)
    return reference_polarization, improper, proper, polarizations


def test_strain_voigt_round_trip():
    voigt = np.arange(6, dtype=float)
    strain = voigt_to_strain(voigt)

    np.testing.assert_allclose(strain_to_voigt(strain), voigt)


def test_piezoelectric_voigt_round_trip():
    voigt = np.arange(18, dtype=float).reshape(3, 6)
    tensor = voigt_to_piezoelectric(voigt)

    np.testing.assert_allclose(piezoelectric_to_voigt(tensor), voigt)
    np.testing.assert_allclose(tensor, tensor.swapaxes(1, 2))


def test_one_strained_structure_set_produces_both_tensors():
    strains = generate_strains(amplitude=1e-3)
    reference_polarization, improper, proper, polarizations = synthetic_response(strains)

    result = evaluate_piezoelectric_tensors(polarizations, strains)

    assert len(strains) == 13
    assert result.rank == 7
    assert result.residual_rms < 1e-14
    assert result.improper.rank == (1, 2)
    assert result.proper.rank == (1, 2)
    assert not result.improper.is_atomic
    assert not result.proper.is_atomic
    np.testing.assert_allclose(result.reference_polarization, reference_polarization, atol=1e-12)
    np.testing.assert_allclose(result.improper.data, improper, atol=1e-12)
    np.testing.assert_allclose(result.proper.data, proper, atol=1e-12)


def test_direct_proper_fit_matches_vanderbilt_without_symmetry():
    strains = generate_strains(amplitude=1e-3)
    reference_polarization, _, proper, polarizations = synthetic_response(strains)
    symmetric_modes = []
    for component in range(3):
        for j, k in VOIGT_PAIRS:
            mode = np.zeros((3, 3, 3))
            mode[component, j, k] = 1
            mode[component, k, j] = 1
            symmetric_modes.append(mode.reshape(-1))
    symmetry_basis = np.asarray(symmetric_modes).T

    direct, fitted_reference, rank, residual = evaluate_proper_piezoelectric_direct(
        polarizations, strains, symmetry_basis
    )

    assert rank == 21
    assert residual < 1e-14
    np.testing.assert_allclose(fitted_reference, reference_polarization, atol=1e-12)
    np.testing.assert_allclose(direct.data, proper, atol=1e-12)


def test_dipole_lattice_linear_system_recovers_improper_and_rotations():
    reference = periodic_reference()
    structures = build_strained_structures(reference, amplitude=1e-3)
    strains = np.asarray([atoms.info["strain"] for atoms in structures])
    reference_polarization, improper, proper, polarizations = synthetic_response(strains)
    cells = np.asarray([atoms.cell.array for atoms in structures])
    dipoles = polarizations * np.abs(np.linalg.det(cells))[:, None]

    fit = evaluate_dipole_lattice_derivative(dipoles, cells, reference.cell.array)

    np.testing.assert_allclose(
        fit.reference_dipole,
        reference_polarization * reference.get_volume(),
        atol=1e-5,
    )
    np.testing.assert_allclose(fit.result.improper.data, improper, atol=1e-7)
    np.testing.assert_allclose(fit.result.proper.data, proper, atol=1e-7)

    rotation = np.array([[0.0, -2e-4, 3e-4], [2e-4, 0.0, -1e-4], [-3e-4, 1e-4, 0.0]])
    rotated_cell_change = reference.cell.array @ rotation.T
    predicted = np.einsum("iab,ab->i", fit.dipole_lattice_derivative, rotated_cell_change)
    np.testing.assert_allclose(predicted, rotation @ fit.reference_dipole, atol=1e-12)


def test_proper_symmetry_basis_removes_forbidden_cubic_modes():
    from ase.build import bulk

    from fd2bec.atomic import AtomicStructure

    centrosymmetric = AtomicStructure.from_ase(bulk("Si", "diamond", a=5.43))
    basis = proper_piezoelectric_symmetry_basis(centrosymmetric)

    assert basis.shape == (27, 0)
    np.testing.assert_array_equal(piezoelectric_symbolic_matrix(basis), np.full((3, 6), "0"))


def test_symbolic_pattern_labels_unrestricted_components_in_voigt_order():
    modes = []
    for component in range(3):
        for j, k in VOIGT_PAIRS:
            mode = np.zeros((3, 3, 3))
            mode[component, j, k] = 1.0
            mode[component, k, j] = 1.0
            modes.append(mode.reshape(-1))

    pattern = piezoelectric_symbolic_matrix(np.asarray(modes).T)

    assert pattern.shape == (3, 6)
    assert pattern[0].tolist() == ["a", "b", "c", "d", "e", "f"]
    assert len(set(pattern.reshape(-1))) == 18


def test_canonical_modes_are_identity_on_their_anchor_components():
    modes = []
    for component in range(3):
        for j, k in VOIGT_PAIRS:
            mode = np.zeros((3, 3, 3))
            mode[component, j, k] = 1.0
            mode[component, k, j] = 1.0
            modes.append(mode.reshape(-1))

    canonical, selected = canonical_piezoelectric_modes(np.asarray(modes).T)

    np.testing.assert_allclose(canonical.reshape((18, 18))[:, selected], np.eye(18))


def test_proper_correction_removes_pure_geometric_response():
    polarization = np.array([0.2, -0.3, 0.4])
    correction = proper_piezoelectric_tensor(np.zeros((3, 3, 3)), polarization)
    geometric_improper = -correction

    proper = proper_piezoelectric_tensor(geometric_improper, polarization)

    np.testing.assert_allclose(proper, 0)


def test_build_strained_structures_preserves_fractional_positions():
    reference = periodic_reference()
    structures = build_strained_structures(reference, amplitude=1e-3)

    assert len(structures) == 13
    for atoms in structures:
        np.testing.assert_allclose(atoms.get_scaled_positions(), reference.get_scaled_positions())
        assert np.asarray(atoms.info["strain"]).shape == (3, 3)


def test_apply_strains_rejects_nonperiodic_structure():
    with pytest.raises(ValueError, match="fully periodic"):
        apply_strains(Atoms("H", positions=[[0, 0, 0]]), generate_strains(1e-3))


def test_structure_evaluation_unwraps_cell_dependent_polarization_quantum():
    reference = periodic_reference()
    structures = build_strained_structures(reference, amplitude=1e-3)
    strains = np.asarray([atoms.info["strain"] for atoms in structures])
    _, improper, proper, polarizations = synthetic_response(strains)

    # Move selected calculations to other Berry-phase branches. The quantum
    # changes with each strained cell, so each jump must use that cell.
    for n, (atoms, polarization) in enumerate(zip(structures, polarizations)):
        volume = atoms.get_volume()
        if n in (2, 7, 10):
            polarization = polarization + atoms.cell.array[n % 3] / volume
        atoms.info["REF_polarization"] = polarization

    result = evaluate_piezoelectric_from_structures(structures, reference)

    np.testing.assert_allclose(result.improper.data, improper, atol=1e-10)
    np.testing.assert_allclose(result.proper.data, proper, atol=1e-10)

    for atoms in structures:
        atoms.info["REF_polarization"] += atoms.cell.array[0] / atoms.get_volume()
    shifted_result = evaluate_piezoelectric_from_structures(structures, reference)

    assert not np.allclose(shifted_result.improper.data, result.improper.data)
    np.testing.assert_allclose(shifted_result.proper.data, result.proper.data, atol=1e-7)

    for atoms in structures:
        atoms.info["REF_polarization"] *= E_PER_ANGSTROM2_TO_C_PER_M2
    result_si = evaluate_piezoelectric_from_structures(
        structures,
        reference,
        polarization_quantum_scale=E_PER_ANGSTROM2_TO_C_PER_M2,
    )

    np.testing.assert_allclose(
        result_si.improper.data,
        shifted_result.improper.data * E_PER_ANGSTROM2_TO_C_PER_M2,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        result_si.proper.data,
        shifted_result.proper.data * E_PER_ANGSTROM2_TO_C_PER_M2,
        atol=1e-9,
    )
