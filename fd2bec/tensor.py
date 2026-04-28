import numpy as np
from typing import Union, List, Tuple
from ase.cell import Cell
from functools import cached_property
from dataclasses import dataclass, field

def pq_to_axes(p: int, q: int) -> list[bool]:
    """
    Convert rank-(p,q) tensor notation into axis covariance list.

    In this convention:
        p = number of contravariant (upper) indices  -> False
        q = number of covariant (lower) indices      -> True

    The output encodes axis variance in order:
        [contravariant..., covariant...]

    Parameters
    ----------
    p : int
        Number of contravariant (upper) indices.
    q : int
        Number of covariant (lower) indices.

    Returns
    -------
    list[bool]
        Axis variance list:
        False -> contravariant (upper index)
        True  -> covariant (lower index)

    Examples
    --------
    Vector (contravariant)  : pq_to_axes(1, 0) → [False]
    Covector / force        : pq_to_axes(0, 1) → [True ]
    Born effective charge   : pq_to_axes(1, 1) → [False, True]
    Stress-like tensor      : pq_to_axes(0, 2) → [True , True]
    Rank-2 mixed tensor     : pq_to_axes(2, 1) → [False, False, True]
    Rank-4 elasticity tensor: pq_to_axes(2, 2) → [False, False, True, True]
    """
    if p < 0 or q < 0:
        raise ValueError("p and q must be non-negative integers")

    return [False] * p + [True] * q

@dataclass
class Tensor:
    """
    General tensor object with explicit index covariance tracking.

    Each axis has a variance flag:
        False -> contravariant (upper index, vector-like)
        True  -> covariant (lower index, covector-like)

    Coordinate systems:
        - cartesian
        - fractional (lattice coordinates)
    """
    
    data: np.ndarray
    axes: List[bool]
    # cell: Union[Cell,np.ndarray] = field(default=None)
    cell: np.ndarray
    shape: Tuple[int] = field(init=False)
    basis: str = field(default="cartesian")
    
    
    
    def __post_init__(self):
        self.shape = self.data.shape
        # if self.cell is not None:
        #     self.cell = LatticeVectors(self.cell)

    def _apply_axis(self, arr: np.ndarray, axis: int, M: np.ndarray) -> np.ndarray:
        """
        Apply a linear map to a single tensor axis.

        Parameters
        ----------
        arr : np.ndarray
            Tensor data.

        axis : int
            Axis to transform.

        M : (3,3) ndarray
            Transformation matrix.

        Returns
        -------
        np.ndarray
            Transformed tensor.
        """
        arr = np.moveaxis(arr, axis, -1)
        arr = np.einsum("...j,ij->...i", arr, M, optimize=True)
        return np.moveaxis(arr, -1, axis)

    def __transform(self, matrices: list[np.ndarray]) -> "Tensor":
        """
        Apply per-axis linear transformations.
        Assumes matrices are already correctly prepared.
        """
        result = self.data
        rank = len(self.axes)

        for i in range(rank):
            axis = -(i + 1)
            result = self._apply_axis(result, axis, matrices[i])

        return Tensor(
            data=result,
            axes=self.axes,
            cell=self.cell,
            basis=self.basis
        )
    
    def transform(self, A, Ainv=None, to="fractional"):
        """
        Change lattice basis (Cartesian <-> fractional).
        """
        if self.basis == to:
            return self

        if Ainv is None:
            Ainv = np.linalg.inv(A)

        mats = []

        if self.basis == "cartesian" and to == "fractional":
            for cov in self.axes[::-1]:
                if cov:
                    mats.append(A)      # covariant
                else:
                    mats.append(Ainv.T)     # contravariant

        elif self.basis == "fractional" and to == "cartesian":
            for cov in self.axes[::-1]:
                if cov:
                    mats.append(Ainv)
                else:
                    mats.append(A.T)

        else:
            raise ValueError("Unsupported transformation")

        return self.__transform(mats)._replace_basis(to)
        
    def rotate(self, Q: np.ndarray):
        Q = np.asarray(Q)

        if Q.shape != (3, 3):
            raise ValueError("Rotation must be 3x3")

        mats = []

        for cov in self.axes[::-1]:
            mats.append(Q)

        return self.__transform(mats)
        
    def to(self,basis:str):
        return self.transform(self.cell, None, basis)
    
    def _replace_basis(self, basis: str):
        """
        Return a shallow copy of the tensor with updated basis.
        """
        return Tensor(
            data=self.data,
            axes=self.axes,
            cell=self.cell,
            basis=basis
        )

    # ------------------------------------------------------------
    # BASIC ALGEBRA
    # ------------------------------------------------------------

    def __add__(self, other:"Tensor"):
        if self.axes != other.axes:
            raise ValueError("Tensor index structures must match")
        if self.basis != other.basis:
            raise ValueError("Basis must match")

        return Tensor(data=self.data + other.data, axes=self.axes, cell=self.cell, basis=self.basis)

    def __mul__(self, scalar):
        return Tensor(data=self.data * scalar, axes=self.axes, cell=self.cell, basis=self.basis)

    __rmul__ = __mul__

    # ------------------------------------------------------------
    # TENSOR CONTRACTION
    # ------------------------------------------------------------

    def contract(self, i, j):
        """
        Contract one covariant and one contravariant index.
        """
        if self.axes[i] == self.axes[j]:
            raise ValueError("Cannot contract indices of same variance")

        data = np.tensordot(self.data, np.eye(3), axes=([i, j], [0, 1]))

        new_axes = [
            ax for k, ax in enumerate(self.axes)
            if k not in (i, j)
        ]

        return Tensor(data=data, axes=new_axes, cell=self.cell, basis=self.basis)

    # ------------------------------------------------------------
    # REPRESENTATION
    # ------------------------------------------------------------

    # def __repr__(self):
    #     return f"Tensor(\nshape={self.data.shape},\naxes={self.axes},\nbasis={self.basis},\ndata={self.data})"
    
    
class Vector(Tensor):
    def __init__(self, data, cell=None, basis="cartesian"):
        super().__init__(
            data=np.asarray(data),
            axes=[False],
            cell=cell,
            basis=basis
        )
        
class Position(Vector):
    pass

class Dipole(Vector):
    pass

class Displacement(Vector):
    pass

class LatticeVectors(Vector):
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
    def __init__(self, data:Union[Cell,np.ndarray]):
        
        if isinstance(data, LatticeVectors):
            super().__init__(
                data=data.data,
                cell=None,
                basis="cartesian"
            )
        elif isinstance(data,np.ndarray):
            if data.shape[-2:] != (3, 3):
                raise ValueError("Lattice vectors must be (...,3,3)")
            super().__init__(
                data=np.asarray(data),
                cell=None,
                basis="cartesian"
            )
        elif isinstance(data,Cell):
            super().__init__(
                data=data.array,
                cell=None,
                basis="cartesian"
            )
        else:
            raise ValueError("Only LatticeVectors, np.ndarray and ase.cell.Cell supported.")
        
    @cached_property
    def inv(self):
        return np.linalg.inv(self.data)
        
class Force(Tensor):
    def __init__(self, data, cell=None, basis="cartesian"):
        super().__init__(
            data=np.asarray(data),
            axes=[True],
            cell=cell,
            basis=basis
        )

class Stress(Tensor):
    def __init__(self, data, cell=None, basis="cartesian"):
        arr = np.asarray(data)

        if arr.shape[-2:] != (3, 3):
            raise ValueError("Stress must be (...,3,3)")

        super().__init__(
            data=arr,
            axes=[True, True],
            cell=cell,
            basis=basis
        )
        
class BornCharge(Tensor):
    def __init__(self, data, cell=None, basis="cartesian"):
        arr = np.asarray(data)

        if arr.shape[-2:] != (3, 3):
            raise ValueError("Born charge must be (...,3,3)")

        super().__init__(
            data=arr,
            axes=[False, True],
            cell=cell,
            basis=basis
        )
        
        