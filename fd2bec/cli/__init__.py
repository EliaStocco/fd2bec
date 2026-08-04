import argparse
import re
import sys
import time
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

            # --- build parser ---
            if prepare_parser is not None:
                parser = prepare_parser(description)
                args = parser.parse_args()
            else:
                args = argparse.Namespace()

            # # --- header ---
            print()
            print("@-------------------------------------------")
            if description:
                print("@ Description: ")
                print(description)

            print("@ Running: ", end="")
            print(f"{' '.join(sys.argv)}")

            # --- run main ---
            print("@ Let's start!\n")
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
