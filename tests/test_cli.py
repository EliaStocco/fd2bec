import sys
import warnings

import pytest

from fd2bec.cli import cli


def test_deprecated_cli_warns_before_running(monkeypatch):
    called = False

    @cli(deprecated=True)
    def deprecated_command(args):
        nonlocal called
        called = True

    monkeypatch.setattr(sys, "argv", ["old-command"])

    with pytest.warns(FutureWarning, match="'old-command' command is deprecated"):
        deprecated_command()

    assert called


def test_regular_cli_does_not_warn():
    @cli()
    def regular_command(args):
        return None

    with warnings.catch_warnings(record=True) as caught:
        regular_command()

    assert not caught
