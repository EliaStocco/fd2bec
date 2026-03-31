import sys
import time
import argparse
from functools import wraps
from fd2bec import DEBUG

#---------------------------------------#
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
    
flist = lambda s:size_type(s,float) # float list
ilist = lambda s:size_type(s,int)   # integer list
slist = lambda s:size_type(s,str)   # string list

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