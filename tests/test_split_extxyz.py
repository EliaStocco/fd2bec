from pathlib import Path
from zipfile import ZipFile

import numpy as np
from ase import Atoms
from ase.io import read

from fd2bec.cli.dataset.split_extxyz import (
    write_dataset,
    write_dataset_archive,
    write_run_script,
)


def test_write_dataset_creates_one_extxyz_folder_per_structure(tmp_path):
    structures = [
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        Atoms("He", positions=[[1.0, 2.0, 3.0]]),
    ]

    filenames = write_dataset(structures, tmp_path / "dataset")

    assert [filename.relative_to(tmp_path / "dataset") for filename in filenames] == [
        Path("structure-0/start.extxyz"),
        Path("structure-1/start.extxyz"),
    ]
    assert read(filenames[0], index=0).get_chemical_symbols() == ["H"]
    assert np.allclose(read(filenames[1], index=0).positions, [[1.0, 2.0, 3.0]])


def test_write_dataset_creates_editable_runner(tmp_path):
    output = tmp_path / "dataset"
    write_dataset([Atoms("H")], output)

    script = tmp_path / "run_all.sh"
    write_run_script(output, script)
    assert script.is_file()
    assert script.stat().st_mode & 0o111
    text = script.read_text(encoding="utf-8")
    assert "output_dir=dataset" in text
    assert 'for folder in "$output_dir"/structure-*/' in text
    assert "cd \"$folder\"" in text
    assert "prepare_aims -i start.extxyz --k-density 5.0" in text


def test_write_dataset_archive_streams_structures_into_zip(tmp_path):
    structures = (Atoms(symbol) for symbol in ("H", "He"))
    archive = tmp_path / "dataset.zip"

    count = write_dataset_archive(structures, archive)

    assert count == 2
    with ZipFile(archive) as zipped:
        assert zipped.namelist() == [
            "structure-0/start.extxyz",
            "structure-1/start.extxyz",
        ]
        assert zipped.read("structure-0/start.extxyz").startswith(b"1\n")
