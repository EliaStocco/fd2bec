import numpy as np

from fd2bec.acoustic_sum_rule import write_acoustic_sum_rule_diagnostics
from fd2bec.cli.dPdR.dPdR2bec import (
    apply_acoustic_sum_rule,
    prepare_args,
    print_dipole_point_residuals,
)
from fd2bec.displacements import displacement_directions, displacement_magnitudes
from fd2bec.fit_diagnostics import (
    frame_residual_diagnostics,
    retained_equation_mask,
    write_dipole_fit_residual_report,
)
from fd2bec.linear_system import LinearSystem


def test_apply_acoustic_sum_rule_reports_sum_and_zeroes_corrected_tensors():
    bec = np.asarray(
        [
            np.eye(3),
            2.0 * np.eye(3),
        ]
    )

    asr_sum, corrected = apply_acoustic_sum_rule(bec)

    np.testing.assert_allclose(asr_sum, 3.0 * np.eye(3))
    np.testing.assert_allclose(corrected.sum(axis=0), np.zeros((3, 3)))
    np.testing.assert_allclose(corrected, np.asarray([-0.5 * np.eye(3), 0.5 * np.eye(3)]))


def test_asr_diagnostics_include_species_and_atom_resolved_contributions(tmp_path):
    path = tmp_path / "asr-diagnostics.txt"
    write_acoustic_sum_rule_diagnostics(
        path,
        np.asarray([np.eye(3), 2.0 * np.eye(3)]),
        ["Ba", "Ti"],
        np.asarray([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        tolerance=1e-5,
    )

    text = path.read_text(encoding="utf-8")
    assert "# mean residual norm: 1.500000000000e+00" in text
    assert "# total residual norm: 3.000000000000e+00" in text
    assert "# mean_I Z*_I (the quantity compared with the tolerance)" in text
    assert "# Species-resolved sums" in text
    assert "Ba 1  1.000000000000e+00" in text
    assert "Ti 1  2.000000000000e+00" in text
    assert "0 Ba  0.000000000000e+00" in text
    assert "1 Ti  5.000000000000e-01" in text


def test_dipole_point_residuals_mark_outliers_and_show_displacements(capsys):
    system = LinearSystem(
        np.zeros((12, 1)),
        np.asarray([1.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.9, 0.0, 0.0, 10.0, 0.0, 0.0]),
    )
    system.solve()

    diagnostics = frame_residual_diagnostics(system.residual_vector, group_size=3)
    np.testing.assert_allclose(diagnostics.squared_residual_contributions.sum(), 1.0)
    np.testing.assert_array_equal(diagnostics.outliers, [False, False, False, True])
    displacements = np.asarray(
        [
            [[0.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0]],
            [[-0.2, 0.0, 0.0]],
            [[0.3, 0.0, 0.0]],
        ]
    )
    directions = displacement_directions(displacements)
    displacement_norms, maximum_atomic_displacements = displacement_magnitudes(displacements)
    print_dipole_point_residuals(
        diagnostics, directions, displacement_norms, maximum_atomic_displacements
    )
    output = capsys.readouterr().out
    assert "reference" in output
    assert "displacement" in output
    assert "disp. norm" in output
    assert "max atom disp." in output
    assert "residual norm" in output
    assert "1.000000e+01" in output
    assert "*" in output


def test_dipole_fit_report_writes_png_and_markdown(tmp_path):
    diagnostics = frame_residual_diagnostics(
        np.asarray([1.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.9, 0.0, 0.0, 10.0, 0.0, 0.0]),
        group_size=3,
    )
    image, markdown = write_dipole_fit_residual_report(
        tmp_path,
        diagnostics,
        np.asarray(["0", "+", "-", "+"], dtype=object),
        np.asarray([0.0, 0.1, 0.2, 0.3]),
        np.asarray([0.0, 0.1, 0.2, 0.3]),
        n_equations=12,
        n_unknowns=4,
        rank=4,
        condition_number=2.0,
        relative_residual=0.1,
    )

    assert image.is_file()
    text = markdown.read_text(encoding="utf-8")
    assert "dipole-fit-residuals.png" in text
    assert "Max. atomic displacement" in text
    assert "| 3 | displacement | +" in text
    assert "| * |" in text


def test_retained_equation_mask_removes_all_cartesian_components_of_outlier_frames():
    mask = retained_equation_mask(np.asarray([False, True, False]), group_size=3)

    np.testing.assert_array_equal(
        mask,
        [True, True, True, False, False, False, True, True, True],
    )


def test_outlier_refit_is_disabled_by_default_and_can_be_forced():
    parser = prepare_args("test")

    assert not parser.parse_args(["-i", "dataset.extxyz"]).retry
    assert not parser.parse_args(["-i", "dataset.extxyz"]).refit_outliers
    assert parser.parse_args(["-i", "dataset.extxyz", "--refit-outliers"]).refit_outliers
