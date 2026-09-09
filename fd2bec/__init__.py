from typing import Literal

import spglib

spglib.error.OLD_ERROR_HANDLING = False

Basis = Literal["cartesian", "fractional"]


BEC_NORM_THRESHOLD = 20.0  # choose an appropriate value
SYMPREC = 1e-4
ATOL = 1e-5
DEBUG = True
float_format = "% 24.12e"
