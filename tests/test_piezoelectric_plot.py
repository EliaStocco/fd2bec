import numpy as np
import pytest

from fd2bec.cli.general import plot_piezoelectric
from fd2bec.piezoelectric_plot import (
    directional_piezoelectric_response,
    directional_surface,
    induced_polarization_response,
    longitudinal_piezoelectric_response,
    plot_piezoelectric_response_comparison,
    plot_piezoelectric_surface,
    read_piezoelectric_voigt,
)


def test_longitudinal_response_contracts_all_three_tensor_indices():
    piezoelectric = np.zeros((3, 6))
    piezoelectric[0, 0] = 2.5
    directions = np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    response = longitudinal_piezoelectric_response(piezoelectric, directions)

    np.testing.assert_allclose(response, [2.5, -2.5, 0.0])


def test_directional_surface_uses_absolute_radius_and_preserves_response_sign():
    piezoelectric = np.zeros((3, 6))
    piezoelectric[0, 0] = 1.0

    x, y, z, response = directional_surface(piezoelectric, resolution=9)

    np.testing.assert_allclose(np.sqrt(x**2 + y**2 + z**2), np.abs(response))
    assert response.min() < 0.0 < response.max()


def test_shear_response_contributes_to_longitudinal_and_transverse_surfaces():
    piezoelectric = np.zeros((3, 6))
    piezoelectric[0, 5] = 1.0
    direction = np.asarray([[2.0**-0.5, 2.0**-0.5, 0.0]])

    polarization = induced_polarization_response(piezoelectric, direction)
    longitudinal = longitudinal_piezoelectric_response(piezoelectric, direction)
    transverse = directional_piezoelectric_response(piezoelectric, direction, "transverse")

    np.testing.assert_allclose(polarization, [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(longitudinal, [2.0**-0.5])
    np.testing.assert_allclose(transverse, [2.0**-0.5])


def test_reader_accepts_flat_and_standard_voigt_input(tmp_path):
    flat = tmp_path / "flat.txt"
    matrix = tmp_path / "matrix.txt"
    values = np.arange(18.0)
    np.savetxt(flat, values)
    np.savetxt(matrix, values.reshape((3, 6)))

    np.testing.assert_array_equal(read_piezoelectric_voigt(flat), values.reshape((3, 6)))
    np.testing.assert_array_equal(read_piezoelectric_voigt(matrix), values.reshape((3, 6)))
    with pytest.raises(ValueError, match="18 values"):
        np.savetxt(tmp_path / "bad.txt", np.zeros(17))
        read_piezoelectric_voigt(tmp_path / "bad.txt")


def test_plot_and_cli_write_an_image(tmp_path, capsys):
    input_path = tmp_path / "proper-piezoelectric.txt"
    output_path = tmp_path / "surface.png"
    np.savetxt(input_path, np.eye(3, 6))

    response = plot_piezoelectric_surface(np.eye(3, 6), output_path, resolution=9, unit="C/m^2")
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert response.shape == (9, 19)

    cli_output = tmp_path / "cli-surface.png"
    args = plot_piezoelectric.prepare_args(plot_piezoelectric.description).parse_args(
        ["-i", str(input_path), "-o", str(cli_output), "--resolution", "9"]
    )
    plot_piezoelectric.main.__wrapped__(args)
    assert cli_output.exists()
    assert "Longitudinal response range:" in capsys.readouterr().out


def test_shared_plot_scale_must_contain_the_response(tmp_path):
    piezoelectric = np.zeros((3, 6))
    piezoelectric[0, 0] = 2.0

    with pytest.raises(ValueError, match="smaller than the response maximum"):
        plot_piezoelectric_surface(piezoelectric, tmp_path / "surface.png", resolution=9, max_abs=1.0)

    output = tmp_path / "shared-scale.png"
    plot_piezoelectric_surface(piezoelectric, output, resolution=9, max_abs=3.0)
    assert output.exists()


def test_response_comparison_writes_all_directional_surfaces(tmp_path):
    piezoelectric = np.eye(3, 6)
    output = tmp_path / "responses.png"

    responses = plot_piezoelectric_response_comparison(piezoelectric, output, resolution=9)

    assert output.exists()
    assert set(responses) == {"longitudinal", "magnitude", "transverse"}
    assert all(response.shape == (9, 19) for response in responses.values())
