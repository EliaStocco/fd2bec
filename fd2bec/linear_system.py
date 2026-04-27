from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearSystem:
    A: np.ndarray
    b: np.ndarray

    def __post_init__(self):
        if self.A.ndim != 2:
            raise ValueError(f"A must be a 2D array, but got shape {self.A.shape}")
        if self.b.ndim != 1:
            raise ValueError(f"b must be a 1D array, but got shape {self.b.shape}")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError(
                f"Number of rows in A ({self.A.shape[0]}) "
                + f"must match length of b ({self.b.shape[0]})"
            )
