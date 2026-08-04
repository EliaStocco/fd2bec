import numpy as np

from fd2bec.cli.dPdS.build_dataset4dPdS_aims import extract_aims_polarization


def test_extract_final_aims_cartesian_polarization(tmp_path):
    output = tmp_path / "aims.n=0.out"
    output.write_text(
        "Cartesian Polarization  1.0 2.0 3.0\n"
        "unrelated output\n"
        "Cartesian Polarization -4.5e-2 6.7E+01 +8.9\n",
        encoding="utf-8",
    )

    polarization = extract_aims_polarization(output)

    np.testing.assert_allclose(polarization, [-0.045, 67.0, 8.9])
