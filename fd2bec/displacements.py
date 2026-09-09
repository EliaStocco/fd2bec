"""Finite-displacement construction and symmetry-reduction utilities."""

from typing import Tuple
from warnings import warn

import numpy as np
from ase import Atoms

from fd2bec.atomic import AtomicStructure
from fd2bec.tensor import MAPPING, Tensor

CELL_COMPONENTS = (
    (0, 0),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (2, 2),
)


def target_tensor(name: str, natoms: int) -> Tensor:
    """Construct a tensor template for a finite-displacement target name."""
    try:
        tensor_class = MAPPING[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported tensor target {name!r}.") from exc
    return tensor_class.template(natoms)


def tensor_perturbation_shape(tensor: Tensor) -> Tuple[int, ...]:
    """Return the explicit shape of a tensor's input dimensions."""
    if tensor.data is None:
        raise ValueError("A tensor template with explicit data is required.")
    return tensor.input_shape


def tensor_has_atomic_input(tensor: Tensor) -> bool:
    """Whether the selected perturbation space contains an atomic dimension."""
    return any(tensor.axes[index]["type"] == "atomic" for index in tensor.input_axes)


def _signed_directions(directions: np.ndarray) -> np.ndarray:
    """Return reference, positive, and negative versions without duplicates."""
    directions = np.asarray(directions, dtype=float).reshape((-1, directions.shape[-1]))
    signed = np.concatenate([np.zeros((1, directions.shape[1])), directions, -directions], axis=0)
    _, first = np.unique(signed, axis=0, return_index=True)
    return signed[np.sort(first)]


def _rank_increasing_generators(candidates: np.ndarray, design_blocks: np.ndarray) -> np.ndarray:
    """Keep candidates whose response block adds an independent parameter direction."""
    candidates = np.asarray(candidates, dtype=float)
    design_blocks = np.asarray(design_blocks, dtype=float)
    if design_blocks.ndim != 3 or len(design_blocks) != len(candidates):
        raise ValueError(
            "Response design blocks must have shape "
            "(number_of_candidates, response_size, number_of_modes)."
        )

    covered = np.empty((0, design_blocks.shape[-1]))
    selected = []
    rank = 0
    for candidate, block in zip(candidates, design_blocks):
        block = block.reshape((-1, design_blocks.shape[-1]))
        trial = np.vstack((covered, block))
        trial_rank = np.linalg.matrix_rank(trial, tol=1e-10)
        if trial_rank > rank:
            selected.append(candidate)
            covered = trial
            rank = trial_rank
    return np.asarray(selected, dtype=float).reshape((-1, candidates.shape[1]))


def all_cartesian_displacements(number_of_components: int) -> np.ndarray:
    """Return the reference and positive/negative Cartesian basis directions."""
    if number_of_components <= 0:
        raise ValueError("The number of displacement components must be positive.")
    return _signed_directions(np.eye(number_of_components))


def atomic_structure2all_displacements(unit_cell: AtomicStructure, amplitude: float) -> np.ndarray:
    """Return all signed Cartesian atomic displacements and the reference."""
    if amplitude <= 0:
        raise ValueError("The displacement amplitude must be positive.")
    return all_cartesian_displacements(3 * len(unit_cell)) * amplitude


def cell_components2displacements(components: np.ndarray) -> np.ndarray:
    """Expand six components into flattened lower-triangular 3x3 matrices."""
    components = np.asarray(components, dtype=float)
    if components.ndim != 2 or components.shape[1] != len(CELL_COMPONENTS):
        raise ValueError("Cell components must have shape (N, 6).")

    displacements = np.zeros((len(components), 3, 3))
    for column, (row, component) in enumerate(CELL_COMPONENTS):
        displacements[:, row, component] = components[:, column]
    return displacements.reshape((len(components), 9))


def all_cell_displacements() -> np.ndarray:
    """Return the reference and positive/negative basis for six cell components."""
    return _signed_directions(cell_components2displacements(np.eye(len(CELL_COMPONENTS))))


def random_cartesian_displacements(
    number: int, number_of_components: int, atomic: bool, seed: int = None
) -> np.ndarray:
    """Generate normally distributed atomic or lower-triangular cell displacements."""
    if number <= 0:
        raise ValueError("The number of random displacements must be positive.")

    generator = np.random.default_rng(seed)
    if atomic:
        return generator.normal(size=(number, number_of_components))
    if number_of_components != 9:
        raise ValueError("Cell displacements must have nine flattened components.")
    components = generator.normal(size=(number, len(CELL_COMPONENTS)))
    return cell_components2displacements(components)


def _physical_input_candidates(unit_cell: AtomicStructure, tensor: Tensor):
    """Return physical perturbations and their coordinates in input space."""
    input_shape = tensor_perturbation_shape(tensor)
    number_of_components = int(np.prod(input_shape))
    if tensor_has_atomic_input(tensor):
        candidates = np.eye(number_of_components)
        return candidates, candidates
    if input_shape == (3, 3):
        if not hasattr(unit_cell, "cell"):
            candidates = np.eye(number_of_components)
            return candidates, candidates
        candidates = cell_components2displacements(np.eye(len(CELL_COMPONENTS)))
        inverse_cell = np.linalg.inv(np.asarray(unit_cell.cell))
        inputs = []
        for candidate in candidates:
            gradient = (inverse_cell @ candidate.reshape(3, 3)).T
            inputs.append((0.5 * (gradient + gradient.T)).reshape(-1))
        return candidates, np.asarray(inputs)
    candidates = np.eye(number_of_components)
    return candidates, candidates


def symmetry_inequivalent_displacements(
    unit_cell: AtomicStructure, tensor: Tensor, component_modes: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return signed perturbations sufficient to fit a symmetry-reduced tensor."""
    has_structure_representation = hasattr(unit_cell, "get_tensor_symmetry_operations")
    candidates, input_candidates = _physical_input_candidates(unit_cell, tensor)
    if component_modes is None:
        _, _, component_modes = unit_cell.get_symmetry_modes(tensor=tensor)
    if len(component_modes) == 0:
        warn("The provided tensor has no symmetry-allowed components.")
        empty = np.empty((0, input_candidates.shape[1]))
        return _signed_directions(empty), _signed_directions(
            candidates if has_structure_representation else empty
        )

    modes = component_modes.reshape((-1, *tensor.data.shape))
    input_axes = [index + 1 for index in tensor.input_axes]
    modes = np.moveaxis(modes, input_axes, range(-len(input_axes), 0))
    input_size = int(np.prod(tensor_perturbation_shape(tensor)))
    response_size = modes.size // (len(modes) * input_size)
    mode_matrix = modes.reshape((len(modes), response_size, input_size))
    response_design = np.einsum("moi,ki->kom", mode_matrix, input_candidates)
    selected = _rank_increasing_generators(candidates, response_design)
    all_candidates = candidates if has_structure_representation else selected
    return _signed_directions(selected), _signed_directions(all_candidates)


def displacements2structures(atoms: Atoms, displacements: np.ndarray, atomic: bool) -> list[Atoms]:
    """Apply flattened atomic or cell displacements to copies of ``atoms``."""
    displacements = np.asarray(displacements, dtype=float)
    if displacements.ndim != 2:
        raise ValueError("Displacements must be a two-dimensional array.")

    expected = 3 * len(atoms) if atomic else 9
    if displacements.shape[1] != expected:
        raise ValueError(
            f"Expected {expected} displacement components, got {displacements.shape[1]}."
        )
    if not atomic and not np.all(atoms.get_pbc()):
        raise ValueError("Cell displacements require a fully periodic structure.")

    structures = []
    reference_cell = atoms.cell.array
    reference_handedness = np.sign(np.linalg.det(reference_cell))
    for displacement in displacements:
        displaced = atoms.copy()
        if atomic:
            atomic_displacement = displacement.reshape((len(atoms), 3))
            displaced.set_positions(atoms.get_positions() + atomic_displacement)
            displaced.set_array("displacements", atomic_displacement.copy())
        else:
            cell_displacement = displacement.reshape((3, 3))
            displaced_cell = reference_cell + cell_displacement
            if np.sign(np.linalg.det(displaced_cell)) != reference_handedness:
                raise ValueError("A cell displacement produced a singular or inverted cell.")
            displaced.set_cell(displaced_cell, scale_atoms=True)
            displaced.info["cell_displacement"] = cell_displacement.copy()
        structures.append(displaced)

    return structures
