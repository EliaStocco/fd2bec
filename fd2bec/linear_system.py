from dataclasses import dataclass, field
from typing import List

import numpy as np

from fd2bec.mathematics import block_diag


@dataclass
class LinearSystem:
    A: np.ndarray
    b: np.ndarray

    x: np.ndarray = field(init=False, default=None)

    # diagnostics
    rank: int = field(init=False, default=None)
    singular_values: np.ndarray = field(init=False, default=None)
    condition_number: float = field(init=False, default=None)

    residual_vector: np.ndarray = field(init=False, default=None)
    residual_norm: np.ndarray = field(init=False, default=None)
    relative_residual: np.ndarray = field(init=False, default=None)
    rms_residual: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        if self.A.ndim != 2:
            raise ValueError(f"A must be 2D, got {self.A.shape}")

        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError(f"A rows ({self.A.shape[0]}) must match b length ({self.b.shape[0]})")

        # normalize b to 2D
        if self.b.ndim == 1:
            self.b = self.b[:, None]  # shape (n, 1)

    @property
    def n_parallel(self):
        return self.b.shape[1]

    @property
    def n_rows(self):
        return self.A.shape[0]

    @property
    def n_cols(self):
        return self.A.shape[1]

    @property
    def solved(self):
        return self.x is not None

    def solve(self, method="lstsq"):
        """Solve Ax=b and compute diagnostics."""

        self.rank = np.linalg.matrix_rank(self.A)

        if method == "pseudo-inverse":
            self.x = np.linalg.pinv(self.A) @ self.b

        elif method == "lstsq":
            self.x, _, rank_lstsq, self.singular_values = np.linalg.lstsq(
                self.A, self.b, rcond=None
            )

            assert self.rank == rank_lstsq, (
                f"Rank mismatch: matrix_rank={self.rank}, lstsq_rank={rank_lstsq}"
            )

        else:
            raise ValueError(f"Unknown method '{method}'")

        # SVD diagnostics
        if self.singular_values is None:
            self.singular_values = np.linalg.svd(self.A, compute_uv=False)

        self.condition_number = (
            np.inf
            if self.singular_values[-1] == 0
            else self.singular_values[0] / self.singular_values[-1]
        )

        # residual diagnostics
        self.residual_vector = self.A @ self.x - self.b

        self.residual_norm = np.linalg.norm(
            self.residual_vector,
            axis=0,
        )

        b_norm = np.linalg.norm(self.b, axis=0)

        self.relative_residual = np.divide(
            self.residual_norm,
            b_norm,
            out=np.full_like(self.residual_norm, np.nan, dtype=float),
            where=b_norm > 0,
        )

        self.rms_residual = np.sqrt(np.mean(self.residual_vector**2, axis=0))

        return self

    def _quality(self, rel_err):
        return "GOOD" if rel_err < 1e-2 else "BAD"

    def summary(self):
        if self.x is None:
            print("Linear system not solved.")
            return

        err = self.residual_norm[0]
        rel = self.relative_residual[0]

        quality = self._quality(rel)

        print()
        print("-" * 40)
        print("Linear Fit Summary")
        print("-" * 40)
        print(f"Error          : {err:.3e}")
        print(f"Relative Error : {rel:.3e}")
        print(f"Quality        : {quality}")
        print("-" * 40)
        print()


@dataclass
class StackedLinearSystem(LinearSystem):
    A: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    x: np.ndarray = field(init=False)
    list_n_rows: List[int] = field(init=False)
    list_n_cols: List[int] = field(init=False)
    systems: List[LinearSystem]

    def __post_init__(self):
        n_parallel = self.systems[0].n_parallel
        all_parallel = [sys.n_parallel for sys in self.systems]
        assert all([p == n_parallel for p in all_parallel]), (
            "Different number of parallel solutions."
        )
        self.list_n_rows = [sys.n_rows for sys in self.systems]
        self.list_n_cols = [sys.n_cols for sys in self.systems]
        self.A = block_diag([sys.A for sys in self.systems])
        self.b = np.vstack([sys.b for sys in self.systems])
        super().__post_init__()

    def __len__(self):
        return len(self.systems)

    def solve(self, *argv, **kwargs):
        super().solve(*argv, **kwargs)
        offset = 0
        for n, r in enumerate(self.list_n_cols):
            self.systems[n].x = self.x[offset : offset + r, :]
            offset += r
