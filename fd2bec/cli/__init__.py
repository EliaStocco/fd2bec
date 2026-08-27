import argparse
import os
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Union

KEYWORDS = {
    "forces": "REF_forces",
    "efield": "REF_efield",
    "dipole": "REF_dipole",
    "polarization": "REF_polarization",
    "displacements": "displacements",
    "strain": "strain",
}

PACKAGE_DIRECTORY = Path(__file__).resolve().parent


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for an argparse option."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def count_with_percentage(count: int, total: int) -> str:
    """Format a count relative to a non-zero total."""
    return f"{count} out of {total} ({100 * count / total:.1f}%)"


def extract_n(file_path: Path):
    m = re.search(r"n=(\d+)", file_path.name)
    return int(m.group(1)) if m else float("inf")


# ---------------------------------------#
def str2bool(v: Union[bool, str]):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def size_type(s: str, dtype=float, N=None):
    s = s.replace("[", "").replace("]", "")
    if "," in s:
        s = s.replace(",", " ")
    s = s.split()
    if N is not None and len(s) != N:
        raise ValueError(f"You should provide {N} values")
    values = []
    for k in s:
        if k.lower() == "none":
            values.append(None)
        else:
            values.append(dtype(k))
    return values  # return list, not np.array, so None stays


def flist(s):
    return size_type(s, float)  # float list


def ilist(s):
    return size_type(s, int)  # integer list


def slist(s):
    return size_type(s, str)  # string list


def print_input_arguments(args: argparse.Namespace):
    """Print every parsed CLI argument, including parser defaults."""
    print("\t" + "-" * 40)
    print("\tInput arguments:")
    for name, value in vars(args).items():
        print(f"\t {name:>20s}: {value}")
    print("\t" + "-" * 40)
    print()


def git_metadata(directory: Path) -> tuple[str, str]:
    """Return the current Git branch and latest commit."""
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not branch:
            branch = "detached HEAD"
        commit = subprocess.run(
            ["git", "log", "-1", "--format=%H %s"],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return branch, commit or "no commits"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable", "unavailable"


def python_environment() -> str:
    """Describe the active Python environment using standard markers."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        name = os.environ.get("CONDA_DEFAULT_ENV", Path(conda_prefix).name)
        return f"conda: {name} ({conda_prefix})"

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        return f"virtualenv: {Path(virtual_env).name} ({virtual_env})"

    pyenv_version = os.environ.get("PYENV_VERSION")
    if pyenv_version:
        return f"pyenv: {pyenv_version} ({sys.prefix})"

    for location in (Path(sys.executable), Path(sys.prefix)):
        if ".pyenv" in location.parts:
            index = location.parts.index(".pyenv")
            if (
                location.parts[index + 1 : index + 2] == ("versions",)
                and len(location.parts) > index + 2
            ):
                return f"pyenv: {location.parts[index + 2]} ({sys.prefix})"

    if sys.prefix != sys.base_prefix:
        return f"virtual environment ({sys.prefix})"
    return f"system Python ({sys.prefix})"


def cli(prepare_parser=None, description=None):
    """
    Minimal decorator for CLI scripts.

    Features:
    - argparse integration
    - optional description
    - timing
    - optional error logging
    """

    def decorator(main_func):

        @wraps(main_func)
        def wrapper():
            start = time.time()
            started_at = datetime.now().astimezone().strftime("%A, %d %B %Y at %H:%M:%S %Z")

            # --- build parser ---
            if prepare_parser is not None:
                parser = prepare_parser(description)
                args = parser.parse_args()
            else:
                args = argparse.Namespace()

            # # --- header ---
            print()
            print("@-------------------------------------------")
            print("@ Running: ", end="")
            print(f"{' '.join(sys.argv)}")
            print(f"@ Started: {started_at}")
            working_directory = Path.cwd()
            branch, commit = git_metadata(PACKAGE_DIRECTORY)
            print(f"@ Working directory: {working_directory}")
            print(f"@ Git branch: {branch}")
            print(f"@ Last commit: {commit}")
            print(f"@ Python: {sys.version.split()[0]} ({sys.executable})")
            print(f"@ Python environment: {python_environment()}")

            # --- run main ---
            print("@ Let's start!")
            if description:
                print("\n\tDescription:")
                print(textwrap.indent(description, "\t"))
            else:
                print()
            print()
            print_input_arguments(args)
            with RedirectStdout():
                result = main_func(args)
            print("\n@ Job done :)")

            # --- footer ---
            elapsed = time.time() - start
            print(f"@ Finished in {elapsed:.2f}s")
            print("-------------------------------------------@")
            print()

            return result

        return wrapper

    return decorator


class PrefixStdout:
    def __init__(self, prefix="\t"):
        self.prefix = prefix
        self._old = None
        self._at_line_start = True

    def write(self, text):
        for part in text.splitlines(True):
            if self._at_line_start:
                sys.__stdout__.write(self.prefix)
            sys.__stdout__.write(part)
            self._at_line_start = part.endswith("\n")

    def flush(self):
        sys.__stdout__.flush()


class RedirectStdout:
    def __enter__(self):
        self.old = sys.stdout
        sys.stdout = PrefixStdout()
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self.old
