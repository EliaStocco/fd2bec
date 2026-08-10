"""Public tensor definitions and compatibility constructors."""

from typing import Dict, Type, Union

import numpy as np
from ase.cell import Cell

from ._tensor_base import Tensor
from ._tensor_definition import (
    BORN_CHARGES,
    DEFINITIONS,
    DIPOLE,
    ELASTIC_STIFFNESS,
    ENERGY,
    FORCE_CONSTANTS,
    FORCES,
    IMPROPER_PIEZOELECTRIC,
    PIEZOELECTRIC,
    PIEZOELECTRIC_DERIVATIVE,
    POSITIONS,
    STRAIN,
    STRESS,
    STRESS_DERIVATIVE,
    VOLUME,
    build_registry,
    derivative,
    deserialize_definition,
    divide_by,
    evaluate_scalar,
    multiply_by,
    serialize_definition,
    validate_definition,
)


class _DefinedTensor(Tensor):
    def __init__(self, definition=None, **kwargs):
        super().__init__(definition=definition or self.tensor_definition, **kwargs)


class Vector(_DefinedTensor):
    tensor_definition = {
        "name": "vector",
        "axes": [{"name": "vector", "type": "cartesian", "variance": "contravariant"}],
    }


class AtomicVector(_DefinedTensor):
    tensor_definition = {
        "name": "atomic_vector",
        "axes": [
            {"name": "atom", "type": "atomic"},
            {"name": "vector", "type": "cartesian", "variance": "contravariant"},
        ],
    }


class GlobalVector(Vector):
    pass


class Position(AtomicVector):
    tensor_definition = POSITIONS


class Dipole(GlobalVector):
    tensor_definition = DIPOLE


class Displacement(AtomicVector):
    tensor_definition = {
        "name": "displacement",
        "axes": [
            {"name": "atom", "type": "atomic"},
            {"name": "displacement", "type": "cartesian", "variance": "contravariant"},
        ],
    }


class LatticeVectors(GlobalVector):
    """A 3x3 lattice matrix represented by two Cartesian dimensions."""

    tensor_definition = {
        "name": "lattice_vectors",
        "axes": [
            {"name": "lattice", "type": "cartesian", "variance": "contravariant"},
            {"name": "component", "type": "cartesian", "variance": "contravariant"},
        ],
    }

    def __init__(self, data: Union[Cell, np.ndarray] = None, **kwargs):
        if isinstance(data, LatticeVectors):
            data = data.data
        elif isinstance(data, Cell):
            data = data.array
        elif data is not None and not isinstance(data, np.ndarray):
            raise ValueError(
                "Only LatticeVectors, numpy arrays and ase Cell objects are supported."
            )
        super().__init__(data=data, **kwargs)


class Forces(_DefinedTensor):
    tensor_definition = FORCES


class Stress(_DefinedTensor):
    tensor_definition = STRESS


class ElasticStiffnessConstant(_DefinedTensor):
    tensor_definition = ELASTIC_STIFFNESS


class ImproperPiezoelectricTensor(_DefinedTensor):
    tensor_definition = IMPROPER_PIEZOELECTRIC


class ProperPiezoelectricTensor(_DefinedTensor):
    tensor_definition = PIEZOELECTRIC


class BornCharges(_DefinedTensor):
    tensor_definition = BORN_CHARGES


class ForceConstants(_DefinedTensor):
    tensor_definition = FORCE_CONSTANTS


class Rotation(_DefinedTensor):
    tensor_definition = {
        "name": "rotation",
        "axes": [
            {"name": "output", "type": "cartesian", "variance": "contravariant"},
            {"name": "input", "type": "cartesian", "variance": "covariant"},
        ],
    }


class Translation(GlobalVector):
    tensor_definition = {
        "name": "translation",
        "axes": [{"name": "translation", "type": "cartesian", "variance": "contravariant"}],
    }


class Energy(_DefinedTensor):
    tensor_definition = ENERGY


class Volume(_DefinedTensor):
    tensor_definition = VOLUME


class Strain(_DefinedTensor):
    tensor_definition = STRAIN


class StressDerivative(_DefinedTensor):
    tensor_definition = STRESS_DERIVATIVE


class PiezoelectricDerivative(_DefinedTensor):
    tensor_definition = PIEZOELECTRIC_DERIVATIVE


MAPPING: Dict[str, Type[Tensor]] = {
    "dipole": Dipole,
    "stress": Stress,
    "elastic": ElasticStiffnessConstant,
    "piezo": ProperPiezoelectricTensor,
    "forces": Forces,
    "bec": BornCharges,
    "force-constants": ForceConstants,
    "force_constants": ForceConstants,
    "hessian": ForceConstants,
}

__all__ = [
    "Tensor",
    "Vector",
    "AtomicVector",
    "GlobalVector",
    "Position",
    "Dipole",
    "Displacement",
    "LatticeVectors",
    "Forces",
    "Stress",
    "ElasticStiffnessConstant",
    "ImproperPiezoelectricTensor",
    "ProperPiezoelectricTensor",
    "BornCharges",
    "ForceConstants",
    "Rotation",
    "Translation",
    "Energy",
    "Volume",
    "Strain",
    "StressDerivative",
    "PiezoelectricDerivative",
    "MAPPING",
    "DEFINITIONS",
    "ENERGY",
    "POSITIONS",
    "DIPOLE",
    "STRAIN",
    "VOLUME",
    "FORCES",
    "STRESS_DERIVATIVE",
    "STRESS",
    "BORN_CHARGES",
    "PIEZOELECTRIC_DERIVATIVE",
    "IMPROPER_PIEZOELECTRIC",
    "PIEZOELECTRIC",
    "ELASTIC_STIFFNESS",
    "FORCE_CONSTANTS",
    "derivative",
    "multiply_by",
    "divide_by",
    "evaluate_scalar",
    "validate_definition",
    "serialize_definition",
    "deserialize_definition",
    "build_registry",
]
