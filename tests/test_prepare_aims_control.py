from pathlib import Path

import pytest

import fd2bec.cli.aims.prepare_aims as prepare_aims


def test_prepare_args_accepts_explicit_k_grid():
    args = prepare_aims.prepare_args("test").parse_args(
        [
            "-i",
            "reference.extxyz",
            "--k-grid",
            "3",
            "4",
            "5",
            "--k-grid-polarization",
            "6",
            "7",
            "8",
        ]
    )

    assert args.k_grid == [3, 4, 5]
    assert args.k_grid_polarization == [6, 7, 8]


def test_ensure_basis_functions_appends_species_only_when_missing(monkeypatch, tmp_path):
    control = tmp_path / "control.in"
    control.write_text("xc pbe\n", encoding="utf-8")
    species = tmp_path / "species.light.in"
    species.write_text("species H\n", encoding="utf-8")

    def fake_create_species_file(**kwargs):
        assert kwargs["output"] == str(species)
        return species

    monkeypatch.setattr(prepare_aims, "create_species_file", fake_create_species_file)

    generated = prepare_aims.ensure_basis_functions(control, "reference.extxyz")

    assert generated == species
    assert control.read_text(encoding="utf-8") == "xc pbe\nspecies H\n"
    assert prepare_aims.ensure_basis_functions(control, "reference.extxyz") is None


def test_update_control_file_replaces_k_grid_density_and_polarization(monkeypatch, tmp_path):
    control = tmp_path / "control.in"
    control.write_text(
        """# settings
k_grid_density 5
output polarization 1 99 99 99
output polarization 2 99 99 99
species H
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(prepare_aims, "suggest_kgrid", lambda *_: (4, 6, 8))

    k_grid, matches = prepare_aims.update_control_file(control, "reference.extxyz", 5.0)

    assert k_grid == (4, 6, 8)
    assert not matches
    assert control.read_text(encoding="utf-8") == (
        "# settings\n"
        "k_grid 4 6 8\n"
        "output polarization 1 40 6 8\n"
        "output polarization 2 4 60 8\n"
        "output polarization 3 4 6 80\n"
        "species H\n"
    )


def test_polarization_kgrids_are_denser_than_scf_grid():
    scf_grid = (4, 6, 8)

    polarization = prepare_aims.polarization_kgrids(scf_grid, (2, 10, 7))

    assert polarization == (
        (1, 5, 6, 8),
        (2, 4, 10, 8),
        (3, 4, 6, 9),
    )
    scf_points = 4 * 6 * 8
    assert all(mesh[1] * mesh[2] * mesh[3] > scf_points for mesh in polarization)


def test_update_control_file_uses_polarization_density(monkeypatch, tmp_path):
    control = tmp_path / "control.in"
    control.write_text("species H\n", encoding="utf-8")
    def fake_suggest_kgrid(_, density):
        return (4, 6, 8) if density == 5.0 else (2, 10, 7)

    monkeypatch.setattr(prepare_aims, "suggest_kgrid", fake_suggest_kgrid)

    prepare_aims.update_control_file(control, "reference.extxyz", 5.0, 10.0)

    assert "output polarization 1 5 6 8\n" in control.read_text(encoding="utf-8")
    assert "output polarization 2 4 10 8\n" in control.read_text(encoding="utf-8")
    assert "output polarization 3 4 6 9\n" in control.read_text(encoding="utf-8")


def test_update_control_file_uses_explicit_k_grid(monkeypatch, tmp_path):
    control = tmp_path / "control.in"
    control.write_text("k_grid 1 1 1\nspecies H\n", encoding="utf-8")

    def fake_suggest_kgrid(_, density):
        assert density == 10.0
        return (6, 6, 6)

    monkeypatch.setattr(prepare_aims, "suggest_kgrid", fake_suggest_kgrid)

    k_grid, _ = prepare_aims.update_control_file(
        control,
        "reference.extxyz",
        5.0,
        k_grid=(2, 3, 4),
    )

    assert k_grid == (2, 3, 4)
    assert control.read_text(encoding="utf-8") == (
        "k_grid 2 3 4\n"
        "output polarization 1 6 3 4\n"
        "output polarization 2 2 6 4\n"
        "output polarization 3 2 3 6\n"
        "species H\n"
    )


def test_update_control_file_uses_explicit_polarization_k_grid(monkeypatch, tmp_path):
    control = tmp_path / "control.in"
    control.write_text("species H\n", encoding="utf-8")

    def fail_if_called(*_):
        raise AssertionError("k-grid density should not be used for explicit grids")

    monkeypatch.setattr(prepare_aims, "suggest_kgrid", fail_if_called)

    prepare_aims.update_control_file(
        control,
        "reference.extxyz",
        5.0,
        k_grid=(2, 3, 4),
        k_grid_polarization=(7, 8, 9),
    )

    assert control.read_text(encoding="utf-8") == (
        "k_grid 2 3 4\n"
        "output polarization 1 7 3 4\n"
        "output polarization 2 2 8 4\n"
        "output polarization 3 2 3 9\n"
        "species H\n"
    )


def test_explicit_polarization_k_grid_must_exceed_scf_grid(tmp_path):
    control = tmp_path / "control.in"
    control.write_text("species H\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must exceed"):
        prepare_aims.update_control_file(
            control,
            "reference.extxyz",
            5.0,
            k_grid=(2, 3, 4),
            k_grid_polarization=(2, 8, 9),
        )


def test_write_control_templates_always_writes_general_and_csc_controls(tmp_path):
    control = tmp_path / "control.in"
    control.write_text("species H\n", encoding="utf-8")

    general_control, first_control, other_control = prepare_aims.write_control_templates(control)

    assert Path(general_control).name == "control.general.in"
    assert Path(general_control).read_text(encoding="utf-8") == "species H\n"
    assert Path(first_control).name == "control.first.in"
    assert Path(first_control).read_text(encoding="utf-8") == (
        "elsi_restart write scf_converged\n"
        "species H\n"
    )
    assert Path(other_control).name == "control.other.in"
    assert Path(other_control).read_text(encoding="utf-8") == (
        "elsi_restart read\n"
        "species H\n"
    )


def test_write_control_templates_removes_existing_restart_from_general_control(tmp_path):
    control = tmp_path / "control.in"
    control.write_text(
        "species H\nelsi_restart read\nxc pbe\n",
        encoding="utf-8",
    )

    general_control, first_control, other_control = prepare_aims.write_control_templates(control)

    assert "elsi_restart" not in general_control.read_text(encoding="utf-8")
    assert "elsi_restart write scf_converged" in first_control.read_text(encoding="utf-8")
    assert "elsi_restart read" in other_control.read_text(encoding="utf-8")


def test_ensure_basis_functions_does_not_duplicate_existing_species(monkeypatch, tmp_path):
    control = tmp_path / "control.in"
    control.write_text("species H\n", encoding="utf-8")
    called = False

    def fail_if_called(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("species generation should not be called")

    monkeypatch.setattr(prepare_aims, "create_species_file", fail_if_called)

    assert prepare_aims.ensure_basis_functions(control, "reference.extxyz") is None
    assert not called
