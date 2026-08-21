import warnings

import fd2bec.cli as cli_module
from fd2bec.cli import cli


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
        cli_module.python_environment()
        == "conda: test-environment (/tmp/conda/test-environment)"
    )


def test_cli_prints_description_after_execution_context(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_module,
        "git_metadata",
        lambda directory: ("main", "abc123 Test commit"),
    )
    monkeypatch.setattr(cli_module, "python_environment", lambda: "system Python")

    @cli(description="Prepare calculations for FHI-aims.")
    def command(args):
        return None

    command()

    output = capsys.readouterr().out
    assert output.index("@ Running:") < output.index("@ Let's start!")
    assert output.index("@ Let's start!") < output.index("@ Description:")
    assert output.index("@ Description:") < output.index(
        "\tPrepare calculations for FHI-aims."
    )
