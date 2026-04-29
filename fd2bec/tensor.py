import numpy as np
from typing import Union, List, Tuple
from ase.cell import Cell
from functools import cached_property
from dataclasses import dataclass, field, replace

def axes_to_pq(axes: list[bool]) -> tuple[int, int]:
    """
    Convert axis covariance list into rank-(p,q) tensor notation.

    In this convention:
        False -> contravariant (upper index)
        True  -> covariant (lower index)

    Returns:
        p = number of contravariant (False) axes
        q = number of covariant (True) axes

    Parameters
    ----------
    axes : list[bool]
        Axis variance list.

    Returns
    -------
    tuple[int, int]
        (p, q) where:
        p = number of contravariant indices
        q = number of covariant indices

    Examples
    --------
    [False]                     → (1, 0)
    [True]                      → (0, 1)
    [False, True]               → (1, 1)
    [True, True]                → (0, 2)
    [False, False, True]        → (2, 1)
    [False, False, True, True]  → (2, 2)
    """
    if not all(isinstance(a, bool) for a in axes):
        raise ValueError("axes must be a list of booleans")

    p = axes.count(False)
    q = axes.count(True)

    return p, q

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
    cell: np.ndarray = field(default=None)
    is_atomic: bool = field(default=None)
    is_affine: bool = field(default=False)
    shape: Tuple[int] = field(init=False)
    basis: str = field(default="cartesian")
    
    def __post_init__(self):
        self.shape = self.data.shape
        if self.is_affine and self.rank != (1,0):
            raise ValueError("Affine tensors can only be of rank-(1,0) (positions).")
        
    @property
    def rank(self) -> Tuple[int,int]:
        return axes_to_pq(self.axes)
        
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

    def __transform(self, matrices: list[np.ndarray]) -> 'Tensor':
        """
        Apply per-axis linear transformations.
        Assumes matrices are already correctly prepared.
        """
        result = self.data
        rank = len(self.axes)

        for i in range(rank):
            axis = -(i + 1)
            result = self._apply_axis(result, axis, matrices[i])
            
        return replace(self,data=result)
            
    def __build_operator(self, matrices: list[np.ndarray]) -> np.ndarray:
        """
        Construct flattened full tensor transformation operator.

        Parameters
        ----------
        matrices : list of (3,3) ndarray
            One already-prepared transform matrix per tensor axis.

        Returns
        -------
        np.ndarray
            Flattened operator of shape (3^rank, 3^rank)

        Notes
        -----
        If:
            T'_{flat} = R_tensor @ T_flat

        then:
            R_tensor = M_rank ⊗ ... ⊗ M_2 ⊗ M_1

        Ordering matches NumPy row-major flattening and the internal
        last-axis-first transformation convention.
        """
        if len(matrices) != len(self.axes):
            raise ValueError("Number of matrices must match tensor rank")

        R_tensor = None

        # reverse because last axis is transformed first
        for M in matrices[::-1]:
            R_tensor = M if R_tensor is None else np.kron(R_tensor, M)

        return R_tensor
        
    def __apply_matrices(
        self,
        mats: list[np.ndarray],
        *,
        method: str = "recursive",
        basis: str | None = None,
    ) -> 'Tensor':
        """
        Apply already-prepared per-axis matrices to tensor.

        Parameters
        ----------
        mats : list of (3,3) ndarray
            One transform matrix per tensor axis
            (already accounting for covariance).

        method : {"recursive", "flat"}
            Application strategy.

        basis : str, optional
            If provided, replaces tensor basis in returned object.

        Returns
        -------
        Tensor
            Transformed tensor.
        """
        rank = len(self.axes)

        if len(mats) != rank:
            raise ValueError(
                f"Expected {rank} matrices, got {len(mats)}"
            )

        # ------------------------------------------------------------
        # Recursive per-axis transform
        # ------------------------------------------------------------
        if method == "recursive":
            result = self.__transform(mats)

        # ------------------------------------------------------------
        # Full flattened operator
        # ------------------------------------------------------------
        elif method == "flat":
            tensor_shape = self.data.shape[-rank:]

            if tensor_shape != (3,) * rank:
                raise ValueError(
                    f"Last {rank} axes must each have size 3, got {tensor_shape}"
                )

            batch_shape = self.data.shape[:-rank]

            # Full tensor-product operator
            R_tensor = self.__build_operator(mats)

            # Flatten tensor axes
            arr_flat = self.data.reshape(*batch_shape, -1)

            # Apply operator
            transformed:np.ndarray = np.einsum(
                "...j,ij->...i",
                arr_flat,
                R_tensor,
                optimize=True,
            )

            # Restore original tensor shape
            new_data = transformed.reshape(*batch_shape, *tensor_shape)

            result = replace(self,
                data=new_data
            )

        else:
            raise ValueError(
                f"method must be 'recursive' or 'flat', got '{method}'"
            )

        return replace(result,basis=basis)

    def rotate(self, R: np.ndarray, method: str = "recursive") -> 'Tensor':
        """
        Rotate tensor in current basis.

        For orthogonal rotations, all tensor axes transform with R
        regardless of covariance.
        """
        R = np.asarray(R)

        if R.shape != (3, 3):
            raise ValueError("Rotation must be (3,3)")

        mats = [R] * len(self.axes)

        return self.__apply_matrices(
            mats,
            method=method,
            basis=self.basis,
        )

    def transform(self, A, Ainv=None, to="fractional", method: str = "recursive") -> 'Tensor' :
        """
        Change tensor basis between Cartesian and fractional coordinates.
        """
        if self.basis == to:
            return self

        if Ainv is None:
            Ainv = np.linalg.inv(A)

        mats = []

        # ------------------------------------------------------------
        # Cartesian -> Fractional
        # ------------------------------------------------------------
        if self.basis == "cartesian" and to == "fractional":
            for cov in self.axes[::-1]:
                mats.append(A if cov else Ainv.T)

        # ------------------------------------------------------------
        # Fractional -> Cartesian
        # ------------------------------------------------------------
        elif self.basis == "fractional" and to == "cartesian":
            for cov in self.axes[::-1]:
                mats.append(Ainv if cov else A.T)

        else:
            raise ValueError(
                f"Unsupported transformation: {self.basis} -> {to}"
            )

        return self.__apply_matrices(
            mats,
            method=method,
            basis=to,
        )
             
    def to(self,basis:str,**kwargs):
        return self.transform(self.cell, None, basis,**kwargs)
    
    def rotation_operator(self, R: np.ndarray) -> np.ndarray:
        """
        Construct the full flattened rotation operator for this tensor.

        This builds the linear operator acting on the vectorized tensor such that:

            vec(T') = R_tensor @ vec(T)

        where R is a 3×3 rotation matrix and R_tensor is the corresponding
        Kronecker-product operator acting on all tensor indices.

        Parameters
        ----------
        R : (3,3) ndarray
            Orthogonal rotation matrix (det = ±1, typically det = 1).

        Returns
        -------
        np.ndarray
            Flattened rotation operator of shape (3^rank, 3^rank).

        Notes
        -----
        For orthogonal rotations, covariant and contravariant indices transform
        identically under row-vector convention, so each tensor axis uses R.

        This is equivalent to:
            R_tensor = R ⊗ R ⊗ ... ⊗ R   (rank times)

        This operator is independent of the tensor's covariance structure
        because rotations preserve the Euclidean metric.

        Examples
        --------
        Vector (rank-1):
            R_tensor = R

        Matrix (rank-2):
            R_tensor = R ⊗ R

        Stress or Born-type rank-2 tensors:
            R_tensor = R ⊗ R
        """
        if R.shape != (3, 3):
            raise ValueError("Rotation must be (3,3)")

        mats = [R] * len(self.axes)

        return self.__build_operator(mats)
        

    # ------------------------------------------------------------
    # BASIC ALGEBRA
    # ------------------------------------------------------------

    # def __add__(self, other:'Tensor'):
    #     if self.axes != other.axes:
    #         raise ValueError("Tensor index structures must match")
    #     if self.basis != other.basis:
    #         raise ValueError("Basis must match")

    #     return Tensor(data=self.data + other.data, axes=self.axes, cell=self.cell, basis=self.basis)

    def __mul__(self, scalar):
        return replace(self,data=self.data * scalar)

    __rmul__ = __mul__

    def flatten(self, full:bool=False)->np.ndarray:
        if full and self.is_atomic:
            shape = self.shape[:-len(self.axes)-1] + (-1,)
            return self.data.reshape(shape)
        if self.data.ndim == 1:
            return self.data
        else:
            shape = self.shape[:-len(self.axes)] + (-1,)
            return self.data.reshape(shape)
        
    @cached_property
    def template(self)->np.ndarray:
        n = len(self.axes)
        if self.is_atomic:
            n += 1
        return np.zeros(self.shape[-n:])
            
            
        
    def contract(self,R: np.ndarray)->'Tensor':
        arr = self.flatten()
        arr = contract(R,arr)
        return replace(self,data=arr)
    
    def __array__(self, dtype=None):
        if dtype:
            return self.data.astype(dtype)
        return self.data


def contract(R: np.ndarray,x:np.ndarray)->np.ndarray:
    return np.einsum("ij,...j->...i",R,x)
           
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
    pass


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
    def __init__(self, data:Union[Cell,np.ndarray], **kwargs):
        
        if isinstance(data, LatticeVectors):
            kwargs = SpecialDict(kwargs)
            kwargs["data"] = data.data
            
        elif isinstance(data,np.ndarray):
            kwargs["data"] = data
        elif isinstance(data,Cell):
            kwargs["data"] = data.array
        else:
            raise ValueError("Only LatticeVectors, np.ndarray and ase.cell.Cell supported.")
        super().__init__(**kwargs)
        
    # @cached_property
    # def inv(self):
    #     return np.linalg.inv(self.data)
    
class Force(Tensor):
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
        
class BornCharge(Tensor):
    def __init__(self, **kwargs):
        kwargs = SpecialDict(kwargs)
        kwargs["axes"] = [False, True]
        kwargs["is_atomic"] = True
        super().__init__(
            **kwargs
        )
        
class SpecialDict(dict):
    """
    Dictionary that enforces consistency of existing keys.

    - New keys are always allowed
    - Existing keys must not be reassigned to a different value
    """

    def __setitem__(self, key, value):
        if key in self:
            if self[key] != value:
                raise ValueError(
                    f"Key '{key}' already exists with a different value:\n"
                    f"  existing: {self[key]}\n"
                    f"  new:      {value}"
                )
        super().__setitem__(key, value)
    
        
        