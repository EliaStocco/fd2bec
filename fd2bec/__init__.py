import inspect
from functools import wraps
from typing import Literal

import spglib

spglib.error.OLD_ERROR_HANDLING = False

Basis = Literal["cartesian", "fractional"]


def validate_types(func):
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        for name, value in bound.arguments.items():
            expected = func.__annotations__.get(name, None)
            if expected is None:
                continue

            # handle typing.Literal etc. simply
            if isinstance(expected, type):
                if not isinstance(value, expected):
                    raise TypeError(
                        f"{name} must be {expected.__name__}, got {type(value).__name__}"
                    )

        return func(*args, **kwargs)

    return wrapper


BEC_NORM_THRESHOLD = 20.0  # choose an appropriate value
SYMPREC = 1e-4
ATOL = 1e-5
DEBUG = True
float_format = "% 24.12e"
