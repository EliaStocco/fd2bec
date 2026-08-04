from typing import Union

import numpy as np
from ase.cell import Cell

from ._tensor_base import SpecialDict, Tensor


class Vector(Tensor):
    def __init__(self, **kwargs):
        kwargs = SpecialDict(kwargs)
        kwargs["axes"] = [False]
        super().__init__(**kwargs)


class AtomicVector(Vector):
    def __init__(self, **kwargs):
        kwargs = SpecialDict(kwargs)
        kwargs["is_atomic"] = True
        super().__init__(**kwargs)


class GlobalVector(Vector):
    def __init__(self, **kwargs):
        kwargs = SpecialDict(kwargs)
        kwargs["is_atomic"] = False
        super().__init__(**kwargs)


class Position(AtomicVector):
    def __init__(self, **kwargs):
        kwargs = SpecialDict(kwargs)
        kwargs["is_affine"] = True
        super().__init__(**kwargs)


class Dipole(GlobalVector):
    def __init__(self, **kwargs):
        kwargs = SpecialDict(kwargs)
        kwargs["is_affine"] = True
        super().__init__(**kwargs)


class Displacement(AtomicVector):
    pass


class LatticeVectors(GlobalVector):
    """
    Lattice basis vectors represented as a rank-2 tensor (3x3).

    This class represents the crystal lattice matrix A whose rows/columns
    encode the primitive lattice vectors in Cartesian coordinates.

    Convention
    ----------
    data[..., i, j] corresponds to the j-th Cartesian component of the
    i-th lattice vector.

    In matrix form:
        A = [ a1 ]
            [ a2 ]
            [ a3 ]

    where a1, a2, a3 are lattice vectors in Cartesian space.

    Parameters
    ----------
    data : Union[Cell, np.ndarray]
        Lattice representation. Either:
        - ASE Cell object
        - numpy array with shape (..., 3, 3)

    Notes
    -----
    - The last two axes must be (3, 3), representing a full lattice matrix.
    - This object is treated as a contravariant vector-valued basis object
      under rotations and a linear map under coordinate transformations.
    - Compatible with fractional transformations via:
          r_cart = A · r_frac
          r_frac = A⁻¹ · r_cart
    """

    def __init__(self, data: Union[Cell, np.ndarray], **kwargs):

        if isinstance(data, LatticeVectors):
            kwargs = SpecialDict(kwargs)
            kwargs["data"] = data.data

        elif isinstance(data, np.ndarray):
            kwargs["data"] = data
        elif isinstance(data, Cell):
            kwargs["data"] = data.array
        else:
            raise ValueError("Only LatticeVectors, np.ndarray and ase.cell.Cell supported.")
        super().__init__(**kwargs)

    # @cached_property
    # def inv(self):
    #     return np.linalg.inv(self.data)


class Forces(Tensor):
    def __init__(self, **kwargs):
        kwargs = SpecialDict(kwargs)
        kwargs["axes"] = [True]
        kwargs["is_atomic"] = True
        super().__init__(**kwargs)


class Stress(Tensor):
    def __init__(self, **kwargs):
        kwargs = SpecialDict(kwargs)
        kwargs["axes"] = [True, True]
        kwargs["is_atomic"] = False
        super().__init__(**kwargs)


class BornCharges(Tensor):
    def __init__(self, **kwargs):
        kwargs = SpecialDict(kwargs)
        kwargs["axes"] = [False, True]
        kwargs["is_atomic"] = True
        super().__init__(**kwargs)


class Rotation(Tensor):
    def __init__(self, **kwargs):
        kwargs = SpecialDict(kwargs)
        kwargs["axes"] = [False, True]
        kwargs["is_atomic"] = False
        super().__init__(**kwargs)


class Translation(GlobalVector):
    pass
