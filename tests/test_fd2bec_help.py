import subprocess
import sys
from pathlib import Path


HELP_SCRIPT = Path(__file__).resolve().parents[1] / "fd2bec" / "cli" / "fd2bec_help.py"


def run_help(*arguments: str) -> str:
    result = subprocess.run(
        [sys.executable, str(HELP_SCRIPT), "--no-color", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_lists_scripts_grouped_by_cli_folder():
    output = run_help()

    assert "\taims:" in output
    assert "\tgeneral:" in output
    assert "\t - prepare_aims.py" in output
    assert "\t - solve_linear_system.py" in output
    assert "tools.py" not in output


def test_filters_folders_and_prints_descriptions():
    output = run_help("--descriptions", "--folders", "general")

    assert "\tgeneral:" in output
    assert "Overview of the mathematical problem to solve." in output
    assert "\taims:" not in output


def test_can_show_only_folder_names():
    output = run_help("--show-folders")

    assert "\taims:" in output
    assert "prepare_aims.py" not in output
