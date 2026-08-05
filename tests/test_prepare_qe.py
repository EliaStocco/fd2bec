from argparse import Namespace

import pytest
from ase import Atoms
from ase.io import write

from fd2bec.cli.qe.prepare_qe import (
    automatic_k_grid,
    nscf_template,
    preparation_commands,
    prepare_qe_files,
    write_run_script,
)

SCF = """&control
 calculation = 'scf'
 ! FD2BEC
/
&system
 ibrav = 0
/
K_POINTS automatic
 4 6 8 0 0 0
"""


def test_nscf_template_adds_directional_berry_settings():
    generated = nscf_template(SCF, gdir=2, nppstr=60)

    assert "calculation = 'nscf'" in generated
    assert "lberry = .true." in generated
    assert "gdir = 2" in generated
    assert "nppstr = 60" in generated
    assert generated.count("! FD2BEC") == 1


def test_prepare_qe_files_writes_geometry_and_four_templates(tmp_path):
    atoms = Atoms("H", scaled_positions=[[0.1, 0.2, 0.3]], cell=[2, 3, 4], pbc=True)
    structures = tmp_path / "structures.extxyz"
    template = tmp_path / "scf.in"
    write(structures, [atoms, atoms], format="extxyz")
    template.write_text(SCF, encoding="utf-8")

    number, k_grid = prepare_qe_files(structures, template, tmp_path / "prepared")

    assert number == 2
    assert k_grid == (4, 6, 8)
    for gdir, nppstr in enumerate((40, 60, 80), start=1):
        content = (tmp_path / f"prepared/templates/nscf.g={gdir}.in").read_text()
        assert f"nppstr = {nppstr}" in content


def test_automatic_k_grid_rejects_nonautomatic_card():
    with pytest.raises(ValueError, match="K_POINTS automatic"):
        automatic_k_grid("K_POINTS gamma\n")


def test_scf_template_requires_exactly_one_marker():
    with pytest.raises(ValueError, match="exactly one"):
        nscf_template(SCF.replace("! FD2BEC", ""), 1, 40)


def test_preparation_commands_generate_piezo_and_export_qe_geometries():
    args = Namespace(
        input="reference.extxyz",
        what="piezo",
        amplitude=0.002,
        no_symmetry=False,
        number=7,
        seed=12,
    )

    generate, export = preparation_commands(
        args, "displaced.extxyz", "displacements.txt", "geometries"
    )

    assert "fd2bec.cli.displacements.generate_displacements" in generate
    assert generate[generate.index("-w") + 1] == "piezo"
    assert generate[-4:] == ["--number", "7", "--seed", "12"]
    assert "fd2bec.cli.displacements.extxyz2folder" in export
    assert export[-4:] == ["-f", "espresso-in", "-o", "geometries"]


def test_run_script_uses_relocatable_paths(tmp_path):
    output = tmp_path / "prepared"
    script = output / "sourceme.sh"

    write_run_script(output, script, last_index=4)
    content = script.read_text()

    assert str(tmp_path) not in content
    assert 'FD2BEC_QE_ROOT="${FD2BEC_QE_SCRIPT_DIR}/."' in content
    assert "FD2BEC_QE_LAST_INDEX=4" in content
