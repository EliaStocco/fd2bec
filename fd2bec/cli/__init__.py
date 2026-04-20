import argparse
import sys
import time
from functools import wraps
from typing import Union


# ---------------------------------------#
def str2bool(v: Union[bool, str]):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def size_type(s: str, dtype=float, N=None):
    s = s.replace("[", "").replace("]", "")
    if "," in s:
        s = s.replace(",", " ")
    s = s.split()
    if N is not None and len(s) != N:
        raise ValueError("You should provide {:d} values".format(N))
    else:
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
            if description:
                print(description)

            print(f"Running: {' '.join(sys.argv)}\n")

            # --- run main ---

            result = main_func(args)

            # --- footer ---
            elapsed = time.time() - start
            print(f"\nFinished in {elapsed:.2f}s")

            return result

        return wrapper

    return decorator
