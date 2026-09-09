import warnings
from pathlib import Path

import fd2bec.cli as cli_module
from fd2bec.cli import cli, read_input_structures


def test_read_input_structures_normalizes_path(monkeypatch):
    calls = []
    expected = [object()]

    def fake_read(path, **kwargs):
        calls.append((path, kwargs))
        return expected

    monkeypatch.setattr(cli_module, "_read_structure", fake_read)

    result = read_input_structures("dataset.extxyz", index=":", input_format="extxyz", rename=True)

    assert result is expected
    assert calls == [(Path("dataset.extxyz"), {"index": ":", "rename": True, "format": "extxyz"})]


def test_read_input_structures_reads_a_single_structure(monkeypatch):
    expected = object()
    monkeypatch.setattr(cli_module, "_read_structure", lambda *args, **kwargs: expected)

    result = read_input_structures("structure.cif")

    assert result is expected


def test_regular_cli_does_not_warn():
    @cli()
    def regular_command(args):
        return None

    with warnings.catch_warnings(record=True) as caught:
        regular_command()

    assert not caught


def test_python_environment_detects_conda(monkeypatch):
    monkeypatch.setenv("CONDA_PREFIX", "/tmp/conda/test-environment")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "test-environment")

    assert (
        cli_module.python_environment() == "conda: test-environment (/tmp/conda/test-environment)"
    )
