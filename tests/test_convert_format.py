from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read as ase_read

from fd2bec import SYMPREC, io
from fd2bec.cli.structures import convert_format


def test_output_format_is_optional_when_the_filename_has_an_extension():
    parser = convert_format.prepare_args(convert_format.description)

    args = parser.parse_args(["-i", "structure.extxyz", "-o", "structure.cif"])

    assert args.format is None


def test_parser_accepts_structure_transformations():
    parser = convert_format.prepare_args(convert_format.description)

    args = parser.parse_args(
        [
            "-i",
            "structure.extxyz",
            "-o",
            "structure.cif",
            "--standardize",
            "primitive",
            "--rotate-cell",
        ]
    )

    assert args.standardize == "primitive"
    assert args.rotate_cell


def test_standardize_option_transforms_before_writing(monkeypatch, tmp_path):
    parser = convert_format.prepare_args(convert_format.description)
    args = parser.parse_args(
        [
            "-i",
            "input.extxyz",
            "-o",
            str(tmp_path / "primitive.cif"),
            "--standardize",
            "primitive",
        ]
    )
    input_atoms = Atoms("Na", cell=np.eye(3), scaled_positions=[[0, 0, 0]], pbc=True)
    output_atoms = Atoms("Na", cell=np.eye(3) * 2, scaled_positions=[[0, 0, 0]], pbc=True)
    calls = []
    monkeypatch.setattr(
        convert_format,
        "read_input_structures",
        lambda *args, **kwargs: input_atoms,
    )
    monkeypatch.setattr(
        convert_format,
        "standardize_structure",
        lambda atoms, **kwargs: calls.append((atoms, kwargs)) or output_atoms,
    )
    monkeypatch.setattr(
        convert_format,
        "write_structure",
        lambda output, atoms, output_format, **kwargs: calls.append(
            (output, atoms, output_format, kwargs)
        ),
    )

    convert_format.main.__wrapped__(args)

    assert calls == [
        (input_atoms, {"setting": "primitive", "symprec": SYMPREC}),
        (
            tmp_path / "primitive.cif",
            output_atoms,
            "cif",
            {"symprec": SYMPREC, "conventional": False, "primitive": True},
        ),
    ]


def test_write_structure_routes_cif_to_the_symmetry_aware_writer(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        io,
        "write_symmetry_cif",
        lambda output, atoms, **kwargs: calls.append((output, atoms, kwargs)),
    )

    output = tmp_path / "structure.cif"
    io.write_structure(output, "atoms", "cif", symprec=1e-4, conventional=True)

    assert calls == [(output, "atoms", {"symprec": 1e-4, "conventional": True, "primitive": False})]


def test_symmetry_cif_requires_a_periodic_structure(tmp_path):
    with pytest.raises(ValueError, match="fully periodic"):
        io.write_symmetry_cif(
            tmp_path / "molecule.cif", Atoms("H"), symprec=1e-4, conventional=False
        )


def test_write_structure_reuses_the_espresso_geometry_writer(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        io,
        "write_espresso_geometry",
        lambda output, atoms: calls.append((output, atoms)),
    )

    output = tmp_path / "geometry.in"
    io.write_structure(output, "atoms", "espresso-in", symprec=1e-4, conventional=False)

    assert calls == [(output, "atoms")]


def test_write_structure_uses_ase_for_other_formats(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        io,
        "write",
        lambda output, atoms, **kwargs: calls.append((output, atoms, kwargs)),
    )

    output = Path(tmp_path / "structure.xyz")
    io.write_structure(output, "atoms", "xyz", symprec=1e-4, conventional=False)

    assert calls == [(output, "atoms", {"format": "xyz"})]


def test_cif_extension_selects_the_symmetry_aware_writer():
    assert convert_format.inferred_output_format(Path("structure.cif")) == "cif"
    assert convert_format.inferred_output_format(Path("structure.CIF")) == "cif"
    assert convert_format.inferred_output_format(Path("structure.extxyz")) is None


def test_primitive_symmetry_cif_keeps_a_primitive_cell(tmp_path):
    atoms = Atoms("Na", cell=[3.0, 3.0, 3.0], scaled_positions=[[0.0, 0.0, 0.0]], pbc=True)
    output = tmp_path / "primitive.cif"

    io.write_primitive_symmetry_cif(output, atoms, symprec=1e-4)

    text = output.read_text(encoding="utf-8")
    assert "Input-cell symmetry CIF" in text
    assert "_symmetry_equiv_pos_as_xyz" in text
    assert "_space_group_IT_number" in text
    assert len(ase_read(output)) == len(atoms)


def test_input_symmetry_cif_preserves_a_supercell_and_expands_its_sites(tmp_path):
    primitive = Atoms(
        "BaTiO3",
        cell=np.eye(3) * 4.0,
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        pbc=True,
    )
    atoms = primitive.repeat((2, 1, 1))
    output = tmp_path / "input-cell.cif"

    io.write_symmetry_cif(output, atoms, symprec=1e-4, conventional=False)

    restored = ase_read(output)
    text = output.read_text(encoding="utf-8")
    assert "Input-cell symmetry CIF" in text
    assert "_space_group_name_H-M_alt" in text
    assert "_symmetry_equiv_pos_as_xyz" in text
    assert len(restored) == len(atoms)
    np.testing.assert_allclose(restored.cell.array, atoms.cell.array)
