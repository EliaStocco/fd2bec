from pathlib import Path

from fd2bec.cli.aims import get_basis_functions_fhi_aims as create_fhi_aims


class FakeAtoms:
    def get_chemical_symbols(self):
        return ["O", "H", "H", "O"]


def test_create_species_file_uses_sorted_unique_species_and_default_layout(monkeypatch, tmp_path):
    aims_folder = tmp_path / "fhi-aims"
    species_folder = aims_folder / "species_defaults" / "defaults_2020" / "light"
    species_folder.mkdir(parents=True)
    (species_folder / "01_H_default").write_text("species H\n", encoding="utf-8")
    (species_folder / "02_O_default").write_text("species O\n", encoding="utf-8")
    monkeypatch.setattr(create_fhi_aims, "read", lambda *args, **kwargs: FakeAtoms())

    output = create_fhi_aims.create_species_file(
        "geometry.in", aims_folder=str(aims_folder), output=str(tmp_path / "species.in")
    )

    assert output == tmp_path / "species.in"
    assert output.read_text(encoding="utf-8") == "species H\nspecies O\n"


def test_create_species_file_uses_environment_variable(monkeypatch, tmp_path):
    aims_folder = tmp_path / "build"
    species_folder = tmp_path / "species_defaults" / "defaults_2020" / "light"
    species_folder.mkdir(parents=True)
    (species_folder / "01_C_default").write_text("species C\n", encoding="utf-8")
    monkeypatch.setenv("TEST_AIMS_PATH", str(aims_folder))
    monkeypatch.setattr(
        create_fhi_aims, "read", lambda *args, **kwargs: type("Atoms", (), {"get_chemical_symbols": lambda self: ["C"]})()
    )

    output = create_fhi_aims.create_species_file(
        "geometry.in", variable="TEST_AIMS_PATH", output=str(tmp_path / "species.in")
    )

    assert Path(output).read_text(encoding="utf-8") == "species C\n"
