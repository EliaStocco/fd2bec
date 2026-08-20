import warnings

from fd2bec.cli import cli


def test_regular_cli_does_not_warn():
    @cli()
    def regular_command(args):
        return None

    with warnings.catch_warnings(record=True) as caught:
        regular_command()

    assert not caught
