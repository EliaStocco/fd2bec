"""Helpers for analysing forces and stresses from symmetry-constrained relaxations."""

from dataclasses import dataclass

import numpy as np

from fd2bec.atomic import AtomicStructure
from fd2bec.tensor import Tensor


@dataclass(frozen=True)
class TensorProjectionReport:
    """A tensor split into symmetry-preserving and symmetry-breaking parts."""

    original: Tensor
    symmetrized: Tensor
    symmetry_breaking: Tensor

    @property
    def maximum_original_component(self) -> float:
        """Largest absolute component before projection."""
        return float(np.max(np.abs(self.original.data)))

    @property
    def maximum_symmetrized_component(self) -> float:
        """Largest absolute component in the symmetry-preserving part."""
        return float(np.max(np.abs(self.symmetrized.data)))

    @property
    def maximum_symmetry_breaking_component(self) -> float:
        """Largest absolute component removed by the symmetry projection."""
        return float(np.max(np.abs(self.symmetry_breaking.data)))


def project_relaxation_tensor(structure: AtomicStructure, tensor: Tensor) -> TensorProjectionReport:
    """Split a force-like tensor into symmetry-preserving and residual parts.

    The group-average projection is the component that can drive a relaxation
    while preserving ``structure``'s detected symmetry.  The residual is the
    component in symmetry-lowering directions.
    """
    symmetrized = structure.symmetrize(tensor)
    residual = tensor.data - symmetrized.data
    if np.issubdtype(residual.dtype, np.inexact):
        real_dtype = np.empty((), dtype=residual.dtype).real.dtype
        roundoff = (
            32
            * np.finfo(real_dtype).eps
            * np.maximum(1.0, np.maximum(np.abs(tensor.data), np.abs(symmetrized.data)))
        )
        residual[np.abs(residual) <= roundoff] = 0.0
    symmetry_breaking = tensor.copy_with(data=residual)
    return TensorProjectionReport(
        original=tensor,
        symmetrized=symmetrized,
        symmetry_breaking=symmetry_breaking,
    )
