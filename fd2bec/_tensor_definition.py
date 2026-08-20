"""JSON-compatible tensor definitions and small definition operations.

Definitions deliberately contain no NumPy objects or runtime classes.  They
describe the mathematical quantity only; data, basis, and cell belong to a
``Tensor`` instance.
"""

from copy import deepcopy
from typing import Any, Dict, Iterable

import numpy as np

DUAL_VARIANCE = {
    "contravariant": "covariant",
    "covariant": "contravariant",
}
_AXIS_TYPES = {"atomic", "cartesian"}
_ROLES = {"input", "output", "value"}
_VARIANCES = set(DUAL_VARIANCE)


def validate_definition(definition: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and return a deep copy of a tensor definition."""
    if not isinstance(definition, dict):
        raise TypeError("A tensor definition must be a dictionary.")
    if not isinstance(definition.get("name"), str) or not definition["name"]:
        raise ValueError("A tensor definition requires a non-empty string 'name'.")
    axes = definition.get("axes")
    if not isinstance(axes, list):
        raise ValueError("A tensor definition requires an 'axes' list.")

    seen_cartesian = False
    for index, axis in enumerate(axes):
        if not isinstance(axis, dict):
            raise TypeError(f"Axis {index} must be a dictionary.")
        if not isinstance(axis.get("name"), str) or not axis["name"]:
            raise ValueError(f"Axis {index} requires a non-empty string 'name'.")
        axis_type = axis.get("type")
        if axis_type not in _AXIS_TYPES:
            raise ValueError(
                f"Axis {index} has invalid type {axis_type!r}; expected 'atomic' or 'cartesian'."
            )
        if axis_type == "atomic":
            if "variance" in axis:
                raise ValueError(f"Atomic axis {index} cannot have a variance.")
            if "affine" in axis:
                raise ValueError(f"Atomic axis {index} cannot be affine.")
        else:
            seen_cartesian = True
            if axis.get("variance") not in _VARIANCES:
                raise ValueError(
                    f"Cartesian axis {index} requires variance 'covariant' or 'contravariant'."
                )
        if "role" in axis and axis["role"] not in _ROLES:
            raise ValueError(
                f"Axis {index} has invalid role {axis['role']!r}; "
                "expected 'input', 'output', or 'value'."
            )
        if "affine" in axis and not isinstance(axis["affine"], bool):
            raise TypeError(f"Axis {index} field 'affine' must be a boolean.")
        if seen_cartesian and axis_type == "atomic":
            raise ValueError("Atomic axes must precede Cartesian axes in storage order.")

    symmetric_axes = definition.get("symmetric_axes", [])
    if not isinstance(symmetric_axes, list):
        raise TypeError("'symmetric_axes' must be a list of axis-index pairs.")
    used_symmetric_axes = set()
    for pair in symmetric_axes:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(index, int) and not isinstance(index, bool) for index in pair)
        ):
            raise TypeError("Each symmetric axis pair must contain exactly two integer indices.")
        left, right = pair
        if left == right or not (0 <= left < len(axes)) or not (0 <= right < len(axes)):
            raise ValueError(f"Invalid symmetric axis pair {pair!r} for {len(axes)} axes.")
        if left in used_symmetric_axes or right in used_symmetric_axes:
            raise ValueError("An axis cannot belong to more than one symmetric axis pair.")
        left_axis, right_axis = axes[left], axes[right]
        if left_axis["type"] != "cartesian" or right_axis["type"] != "cartesian":
            raise ValueError("Only Cartesian axes can form a symmetric axis pair.")
        if left_axis["variance"] != right_axis["variance"]:
            raise ValueError("Symmetric Cartesian axes must have the same variance.")
        if left_axis.get("role") != right_axis.get("role"):
            raise ValueError("Symmetric axes must have the same tensor role.")
        if left_axis.get("affine", False) or right_axis.get("affine", False):
            raise ValueError("Affine axes cannot form a symmetric axis pair.")
        used_symmetric_axes.update(pair)

    result = deepcopy(definition)
    # This is intentionally a plain JSON check.  It catches accidental NumPy
    # scalars/arrays in definitions at the boundary where they are introduced.
    import json

    try:
        json.dumps(result)
    except (TypeError, ValueError) as exc:
        raise TypeError("Tensor definitions must contain JSON-compatible values.") from exc
    return result


def definition_json(definition: Dict[str, Any]) -> Dict[str, Any]:
    """Return a validated, detached JSON-compatible definition."""
    return validate_definition(definition)


def serialize_definition(definition: Dict[str, Any]) -> Dict[str, Any]:
    """Named alias for definition-level serialization."""
    return definition_json(definition)


def deserialize_definition(value: Any) -> Dict[str, Any]:
    """Validate a definition loaded from a dictionary or JSON string."""
    if isinstance(value, str):
        import json

        value = json.loads(value)
    return validate_definition(value)


def build_registry(definitions: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Validate definitions and reject duplicate intrinsic names."""
    registry = {}
    for definition in definitions:
        validated = validate_definition(definition)
        name = validated["name"]
        if name in registry:
            raise ValueError(f"Duplicate tensor definition name {name!r}.")
        registry[name] = validated
    return registry


def _expression(definition: Dict[str, Any]) -> Any:
    """Represent a definition as a readable formula operand."""
    if "formula" in definition:
        return deepcopy(definition["formula"])
    return definition["name"]


def derivative(
    numerator: Dict[str, Any],
    denominator: Dict[str, Any],
    *,
    name: str,
    factor: float = 1.0,
) -> Dict[str, Any]:
    """Build metadata for a derivative, without evaluating it."""
    numerator = validate_definition(numerator)
    denominator = validate_definition(denominator)
    tagged_axes = []
    for index, axis in enumerate(numerator["axes"]):
        new_axis = deepcopy(axis)
        new_axis["role"] = "output"
        new_axis.pop("affine", None)
        tagged_axes.append(("numerator", index, new_axis))
    for index, axis in enumerate(denominator["axes"]):
        new_axis = deepcopy(axis)
        new_axis["role"] = "input"
        new_axis.pop("affine", None)
        if new_axis["type"] == "cartesian":
            new_axis["variance"] = DUAL_VARIANCE[new_axis["variance"]]
        tagged_axes.append(("denominator", index, new_axis))

    # Keep the storage convention explicit: atomic axes first, then Cartesian
    # axes, with numerator axes before denominator axes in each group.
    tagged_axes = [item for item in tagged_axes if item[2]["type"] == "atomic"] + [
        item for item in tagged_axes if item[2]["type"] == "cartesian"
    ]
    axes = [axis for _, _, axis in tagged_axes]
    index_map = {
        (origin, original_index): new_index
        for new_index, (origin, original_index, _) in enumerate(tagged_axes)
    }
    symmetric_axes = []
    for origin, definition in (("numerator", numerator), ("denominator", denominator)):
        for left, right in definition.get("symmetric_axes", []):
            symmetric_axes.append([index_map[(origin, left)], index_map[(origin, right)]])
    formula = {
        "operation": "derivative",
        "numerator": _expression(numerator),
        "denominator": _expression(denominator),
        "factor": factor,
    }
    result = {"name": name, "axes": axes, "formula": formula}
    if symmetric_axes:
        result["symmetric_axes"] = symmetric_axes
    return validate_definition(result)


def _require_scalar(definition: Dict[str, Any]) -> Dict[str, Any]:
    definition = validate_definition(definition)
    if definition["axes"]:
        raise ValueError("Scalar multiplication and division require a rank-zero definition.")
    return definition


def multiply_by(
    definition: Dict[str, Any], scalar: Dict[str, Any], *, name: str, power: int = 1
) -> Dict[str, Any]:
    """Build metadata for multiplying a definition by a scalar quantity."""
    definition = validate_definition(definition)
    scalar = _require_scalar(scalar)
    if not isinstance(power, int):
        raise TypeError("The scalar power must be an integer.")
    result = {
        "name": name,
        "axes": deepcopy(definition["axes"]),
        "formula": {
            "operation": "multiply",
            "numerator": _expression(definition),
            "scalar": _expression(scalar),
            "power": power,
        },
    }
    if definition.get("symmetric_axes"):
        result["symmetric_axes"] = deepcopy(definition["symmetric_axes"])
    return validate_definition(result)


def divide_by(definition: Dict[str, Any], scalar: Dict[str, Any], *, name: str) -> Dict[str, Any]:
    """Build metadata for dividing a definition by a scalar quantity."""
    definition = validate_definition(definition)
    scalar = _require_scalar(scalar)
    result = {
        "name": name,
        "axes": deepcopy(definition["axes"]),
        "formula": {
            "operation": "divide",
            "numerator": _expression(definition),
            "denominator": _expression(scalar),
        },
    }
    if definition.get("symmetric_axes"):
        result["symmetric_axes"] = deepcopy(definition["symmetric_axes"])
    return validate_definition(result)


def evaluate_scalar(definition: Dict[str, Any], cell: Any) -> float:
    """Evaluate the currently supported cell-derived scalar quantity."""
    definition = validate_definition(definition)
    if definition["axes"]:
        raise ValueError("Only scalar definitions can be evaluated as scalar factors.")
    if definition.get("source") == "cell" or definition["name"] == "volume":
        return float(abs(np.linalg.det(np.asarray(cell))))
    raise ValueError(f"No scalar evaluator is registered for {definition['name']!r}.")


ENERGY = {"name": "energy", "axes": []}
POSITIONS = {
    "name": "positions",
    "axes": [
        {"name": "atom", "type": "atomic"},
        {
            "name": "position",
            "type": "cartesian",
            "variance": "contravariant",
            "affine": True,
        },
    ],
}
DIPOLE = {
    "name": "dipole",
    "axes": [
        {
            "name": "dipole",
            "type": "cartesian",
            "variance": "contravariant",
            "affine": True,
        }
    ],
}
STRAIN = {
    "name": "strain",
    "axes": [
        {"name": "strain_i", "type": "cartesian", "variance": "contravariant"},
        {"name": "strain_j", "type": "cartesian", "variance": "contravariant"},
    ],
    "symmetric_axes": [[0, 1]],
}
VOLUME = {"name": "volume", "axes": [], "source": "cell"}

FORCES = derivative(ENERGY, POSITIONS, name="forces", factor=-1.0)
STRESS_DERIVATIVE = derivative(ENERGY, STRAIN, name="energy_derivative_strain")
STRESS = divide_by(STRESS_DERIVATIVE, VOLUME, name="stress")
BORN_CHARGES = derivative(DIPOLE, POSITIONS, name="born_charges")
PIEZOELECTRIC_DERIVATIVE = derivative(DIPOLE, STRAIN, name="dipole_derivative_strain")
IMPROPER_PIEZOELECTRIC = PIEZOELECTRIC_DERIVATIVE
PIEZOELECTRIC = divide_by(PIEZOELECTRIC_DERIVATIVE, VOLUME, name="piezoelectric")
ELASTIC_STIFFNESS = derivative(STRESS, STRAIN, name="elastic")
FORCE_CONSTANTS = derivative(FORCES, POSITIONS, name="force_constants")


DEFINITIONS = build_registry(
    (
        ENERGY,
        POSITIONS,
        DIPOLE,
        STRAIN,
        VOLUME,
        FORCES,
        STRESS_DERIVATIVE,
        STRESS,
        BORN_CHARGES,
        PIEZOELECTRIC_DERIVATIVE,
        PIEZOELECTRIC,
        ELASTIC_STIFFNESS,
        FORCE_CONSTANTS,
    )
)
# Public aliases used by command-line interfaces.
DEFINITIONS.update({"bec": BORN_CHARGES, "piezo": PIEZOELECTRIC})
