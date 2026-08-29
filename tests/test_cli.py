import warnings
from pathlib import Path

import fd2bec.cli as cli_module
from fd2bec.cli import cli, read_input_structures


def test_read_input_structures_normalizes_path_and_reports_progress(monkeypatch, capsys):
    calls = []
    expected = [object()]

    def fake_read(path, **kwargs):
        calls.append((path, kwargs))
        return expected

    monkeypatch.setattr(cli_module, "_read_structure", fake_read)

    result = read_input_structures("dataset.extxyz", index=":", input_format="extxyz", rename=True)

    assert result is expected
    assert calls == [(Path("dataset.extxyz"), {"index": ":", "rename": True, "format": "extxyz"})]
    assert capsys.readouterr().out == "Reading input structures from dataset.extxyz ... done\n"


def test_read_input_structures_uses_a_singular_default_label(monkeypatch, capsys):
    expected = object()
    monkeypatch.setattr(cli_module, "_read_structure", lambda *args, **kwargs: expected)

    result = read_input_structures("structure.cif")

    assert result is expected
    assert capsys.readouterr().out == "Reading input structure from structure.cif ... done\n"


def test_regular_cli_does_not_warn():
    @cli()
    def regular_command(args):
        return None

    with warnings.catch_warnings(record=True) as caught:
        regular_command()

    assert not caught


def test_cli_prints_execution_context(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    metadata_directories = []
    monkeypatch.setattr(
        cli_module,
        "git_metadata",
        lambda directory: (
            metadata_directories.append(directory) or "main",
            "abc123 Test commit",
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "python_environment",
        lambda: "conda: test-environment (/tmp/conda)",
    )

    @cli()
    def command(args):
        return None

    command()

    output = capsys.readouterr().out
    assert f"@ Working directory: {tmp_path}" in output
    assert "@ Git branch: main" in output
    assert "@ Last commit: abc123 Test commit" in output
    assert metadata_directories == [cli_module.PACKAGE_DIRECTORY]
    assert "@ Started: " in output
    assert f"@ Python: {cli_module.sys.version.split()[0]}" in output
    assert "@ Python environment: conda: test-environment (/tmp/conda)" in output


def test_python_environment_detects_conda(monkeypatch):
    monkeypatch.setenv("CONDA_PREFIX", "/tmp/conda/test-environment")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "test-environment")

    assert (
        cli_module.python_environment() == "conda: test-environment (/tmp/conda/test-environment)"
    )
