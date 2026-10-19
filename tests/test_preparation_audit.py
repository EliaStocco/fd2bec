from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from fd2bec.preparation_audit import audit_path, find_preparation_paths


EXTXYZ = """1
Properties=species:S:1:pos:R:3 pbc=\"T T T\" Lattice=\"2 0 0 0 2 0 0 0 2\"\nH 0 0 0\n"""


def _complete_bundle(path):
    (path / "geometries").mkdir(parents=True)
    (path / "start.extxyz").write_text(EXTXYZ, encoding="utf-8")
    (path / "displaced-structures.extxyz").write_text(EXTXYZ + EXTXYZ, encoding="utf-8")
    (path / "displacements.txt").write_text("0 0 0\n0.001 0 0\n", encoding="utf-8")
    (path / "geometries" / "geometry.n=0.in").write_text("atom 0 0 0 H\n", encoding="utf-8")
    (path / "geometries" / "geometry.n=1.in").write_text("atom 0.001 0 0 H\n", encoding="utf-8")
    (path / "fd2bec-log.txt").write_text("@ Job done :)\n", encoding="utf-8")


def test_audit_complete_preparation_folder(tmp_path):
    bundle = tmp_path / "structure-0"
    _complete_bundle(bundle)

    audit = audit_path(bundle, deep=True)

    assert audit.passed
    assert "frames: 2; displacement rows: 2" in audit.details


def test_audit_reports_missing_geometry(tmp_path):
    bundle = tmp_path / "structure-0"
    _complete_bundle(bundle)
    (bundle / "geometries" / "geometry.n=1.in").unlink()

    audit = audit_path(bundle, deep=True)

    assert not audit.passed
    assert "1 geometry files; expected 2" in audit.errors


def test_audit_zip_archive_and_discovery(tmp_path):
    bundle = tmp_path / "structure-0"
    _complete_bundle(bundle)
    archive = tmp_path / "structure-0.zip"
    with ZipFile(str(archive), "w", ZIP_DEFLATED) as zip_file:
        for file in bundle.rglob("*"):
            if file.is_file():
                zip_file.write(str(file), str(file.relative_to(tmp_path)))

    audit = audit_path(archive, deep=True)

    assert audit.passed
    assert find_preparation_paths([tmp_path]) == [bundle, archive]


def test_audit_reports_rank_deficient_cached_bec_selection(tmp_path):
    bundle = tmp_path / "structure-0"
    _complete_bundle(bundle)
    cache = bundle / ".fd2bec"
    cache.mkdir()
    (cache / "displacements.info").write_text(
        "what = bec\nno_symmetry = false\n", encoding="utf-8"
    )
    (cache / "displacements.txt").write_text("0 0 0\n0.001 0 0\n", encoding="utf-8")
    (cache / "born-charge-component-modes.txt").write_text(
        "# fd2bec sparse matrix\n# shape: 2 9\n# zero_tolerance: 1e-10\n"
        "# row column value\n0 0 1\n1 4 1\n",
        encoding="utf-8",
    )

    audit = audit_path(bundle, check_rank=True)

    assert not audit.passed
    assert "BEC projected rank: 1/2; fit rank: 4/5" in audit.details
    assert any("underdetermined" in error for error in audit.errors)
