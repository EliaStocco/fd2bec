"""Finite-strain evaluation of proper and improper piezoelectric tensors."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from ase import Atoms

from fd2bec.tensor import ImproperPiezoelectricTensor, ProperPiezoelectricTensor

# Voigt order and engineering-shear convention:
# (xx, yy, zz, yz, xz, xy) = (εxx, εyy, εzz, 2εyz, 2εxz, 2εxy)
VOIGT_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 0),
    (1, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (0, 1),
)

# One e/Å² expressed in C/m². The elementary charge is exact in SI.
E_PER_ANGSTROM2_TO_C_PER_M2 = 16.02176634


def strain_to_voigt(strain: np.ndarray, atol: float = 1e-10) -> np.ndarray:
    """Convert symmetric strain tensors to engineering-strain Voigt vectors."""
    strain = np.asarray(strain, dtype=float)
    if strain.shape[-2:] != (3, 3):
        raise ValueError(f"Strain must end in shape (3, 3), got {strain.shape}.")
    if not np.allclose(strain, np.swapaxes(strain, -1, -2), atol=atol):
        raise ValueError("Only symmetric strain tensors are supported.")

    voigt = np.empty((*strain.shape[:-2], 6), dtype=float)
    for column, (j, k) in enumerate(VOIGT_PAIRS):
        factor = 1.0 if j == k else 2.0
        voigt[..., column] = factor * strain[..., j, k]
    return voigt


def voigt_to_strain(voigt: np.ndarray) -> np.ndarray:
    """Convert engineering-strain Voigt vectors to symmetric 3x3 tensors."""
    voigt = np.asarray(voigt, dtype=float)
    if voigt.shape[-1:] != (6,):
        raise ValueError(f"Voigt strain must end in shape (6,), got {voigt.shape}.")

    strain = np.zeros((*voigt.shape[:-1], 3, 3), dtype=float)
    for column, (j, k) in enumerate(VOIGT_PAIRS):
        value = voigt[..., column] if j == k else 0.5 * voigt[..., column]
        strain[..., j, k] = value
        strain[..., k, j] = value
    return strain


def piezoelectric_to_voigt(tensor: np.ndarray, atol: float = 1e-10) -> np.ndarray:
    """Convert a piezoelectric tensor symmetric in its last indices to 3x6 form."""
    tensor = np.asarray(tensor, dtype=float)
    if tensor.shape[-3:] != (3, 3, 3):
        raise ValueError(f"Piezoelectric tensor must end in shape (3, 3, 3), got {tensor.shape}.")
    if not np.allclose(tensor, np.swapaxes(tensor, -1, -2), atol=atol):
        raise ValueError("Piezoelectric tensor must be symmetric in its last two indices.")

    voigt = np.empty((*tensor.shape[:-3], 3, 6), dtype=float)
    for column, (j, k) in enumerate(VOIGT_PAIRS):
        voigt[..., :, column] = tensor[..., :, j, k]
    return voigt


def voigt_to_piezoelectric(voigt: np.ndarray) -> np.ndarray:
    """Convert a 3x6 piezoelectric matrix to a symmetric rank-3 tensor."""
    voigt = np.asarray(voigt, dtype=float)
    if voigt.shape[-2:] != (3, 6):
        raise ValueError(f"Piezoelectric Voigt data must end in shape (3, 6), got {voigt.shape}.")

    tensor = np.zeros((*voigt.shape[:-2], 3, 3, 3), dtype=float)
    for column, (j, k) in enumerate(VOIGT_PAIRS):
        tensor[..., :, j, k] = voigt[..., :, column]
        tensor[..., :, k, j] = voigt[..., :, column]
    return tensor


def generate_strains(amplitude: float) -> np.ndarray:
    """Return one reference and positive/negative versions of all six strain modes."""
    if amplitude <= 0:
        raise ValueError("The strain amplitude must be positive.")

    strains = [np.zeros((3, 3))]
    for column in range(6):
        mode = np.zeros(6)
        mode[column] = amplitude
        strain = voigt_to_strain(mode)
        strains.extend((strain, -strain))
    return np.asarray(strains)


def apply_strains(reference: Atoms, strains: np.ndarray) -> List[Atoms]:
    """Apply homogeneous strains while keeping fractional atomic positions fixed."""
    if not np.all(reference.get_pbc()):
        raise ValueError("Piezoelectric strains require a fully periodic structure.")

    strains = np.asarray(strains, dtype=float)
    strain_to_voigt(strains)  # shape and symmetry validation
    reference_cell = reference.cell.array

    structures = []
    for strain in strains:
        deformation = np.eye(3) + strain
        if np.linalg.det(deformation) <= 0:
            raise ValueError("A strain produced a non-positive cell volume.")

        atoms = reference.copy()
        # ASE stores lattice vectors as rows. For column-vector convention
        # r' = F r, the row-vector cell therefore transforms as C' = C F^T.
        atoms.set_cell(reference_cell @ deformation.T, scale_atoms=True)
        atoms.info["strain"] = strain.copy()
        structures.append(atoms)
    return structures


def build_strained_structures(reference: Atoms, amplitude: float) -> List[Atoms]:
    """Build the shared central-difference structure set for both piezo tensors."""
    return apply_strains(reference, generate_strains(amplitude))


def strains_from_cells(reference_cell: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Recover infinitesimal symmetric strains from deformed ASE cell matrices."""
    reference_cell = np.asarray(reference_cell, dtype=float)
    cells = np.asarray(cells, dtype=float)
    if reference_cell.shape != (3, 3):
        raise ValueError(f"Reference cell must have shape (3, 3), got {reference_cell.shape}.")
    if cells.shape[-2:] != (3, 3):
        raise ValueError(f"Cells must end in shape (3, 3), got {cells.shape}.")

    deformation = np.einsum("ij,...jk->...ik", np.linalg.inv(reference_cell), cells)
    deformation = np.swapaxes(deformation, -1, -2)
    displacement_gradient = deformation - np.eye(3)
    return 0.5 * (displacement_gradient + np.swapaxes(displacement_gradient, -1, -2))


def unwrap_polarizations(
    polarizations: np.ndarray,
    cells: np.ndarray,
    reference_index: int = 0,
    quantum_scale: float = 1.0,
) -> np.ndarray:
    """Move strained-cell polarizations onto the branch nearest the reference.

    Polarization is converted to reduced coordinates using the polarization
    quanta ``a_i / volume`` of each strained cell. ``quantum_scale`` is the
    value of one e/Å² in the input polarization unit (1 for e/Å² and
    16.02176634 for C/m²). Integer jumps are removed in reduced coordinates
    before converting back to Cartesian polarization in the original unit.
    """
    polarizations = np.asarray(polarizations, dtype=float)
    cells = np.asarray(cells, dtype=float)
    if polarizations.ndim != 2 or polarizations.shape[1] != 3:
        raise ValueError(f"Polarizations must have shape (N, 3), got {polarizations.shape}.")
    if cells.shape != (len(polarizations), 3, 3):
        raise ValueError(f"Cells must have shape ({len(polarizations)}, 3, 3), got {cells.shape}.")
    if not 0 <= reference_index < len(polarizations):
        raise IndexError("reference_index is outside the polarization array.")
    if quantum_scale <= 0:
        raise ValueError("quantum_scale must be positive.")

    volumes = np.abs(np.linalg.det(cells))
    if np.any(volumes == 0):
        raise ValueError("Polarization cells must have non-zero volume.")

    reduced = np.empty_like(polarizations)
    for n, (polarization, cell, volume) in enumerate(zip(polarizations, cells, volumes)):
        reduced[n] = polarization / quantum_scale * volume @ np.linalg.inv(cell)

    reference = reduced[reference_index]
    reduced -= np.rint(reduced - reference)

    unwrapped = np.empty_like(polarizations)
    for n, (polarization, cell, volume) in enumerate(zip(reduced, cells, volumes)):
        unwrapped[n] = polarization @ cell / volume * quantum_scale
    return unwrapped


def proper_piezoelectric_tensor(
    improper: np.ndarray, reference_polarization: np.ndarray
) -> np.ndarray:
    """Apply the proper-piezoelectric geometric correction for symmetric strain."""
    improper = np.asarray(improper, dtype=float)
    polarization = np.asarray(reference_polarization, dtype=float)
    if improper.shape != (3, 3, 3):
        raise ValueError(f"Improper tensor must have shape (3, 3, 3), got {improper.shape}.")
    if polarization.shape != (3,):
        raise ValueError(f"Reference polarization must have shape (3,), got {polarization.shape}.")

    delta = np.eye(3)
    correction = np.zeros_like(improper)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                correction[i, j, k] = delta[j, k] * polarization[i] - 0.5 * (
                    delta[i, j] * polarization[k] + delta[i, k] * polarization[j]
                )
    return improper + correction


@dataclass(frozen=True)
class PiezoelectricResult:
    """Both piezoelectric tensors obtained from one polarization/strain fit."""

    reference_polarization: np.ndarray
    improper: ImproperPiezoelectricTensor
    proper: ProperPiezoelectricTensor
    rank: int
    residual_rms: float

    @property
    def improper_voigt(self) -> np.ndarray:
        return piezoelectric_to_voigt(self.improper.data)

    @property
    def proper_voigt(self) -> np.ndarray:
        return piezoelectric_to_voigt(self.proper.data)


def evaluate_piezoelectric_tensors(
    polarizations: np.ndarray,
    strains: np.ndarray,
    *,
    cell: np.ndarray = None,
    basis: str = "cartesian",
) -> PiezoelectricResult:
    """Fit improper and proper tensors from the same strained configurations.

    Polarizations must already be placed on a common Berry-phase branch. The
    returned tensors have the same polarization units as the input because
    strain is dimensionless.
    """
    polarizations = np.asarray(polarizations, dtype=float)
    strains = np.asarray(strains, dtype=float)
    if polarizations.ndim != 2 or polarizations.shape[1] != 3:
        raise ValueError(f"Polarizations must have shape (N, 3), got {polarizations.shape}.")
    if strains.shape != (len(polarizations), 3, 3):
        raise ValueError(
            f"Strains must have shape ({len(polarizations)}, 3, 3), got {strains.shape}."
        )
    if not np.all(np.isfinite(polarizations)) or not np.all(np.isfinite(strains)):
        raise ValueError("Polarizations and strains must be finite.")

    strain_voigt = strain_to_voigt(strains)
    design = np.column_stack((np.ones(len(strain_voigt)), strain_voigt))
    rank = int(np.linalg.matrix_rank(design))
    if rank < design.shape[1]:
        raise ValueError(
            "Strained configurations do not span the six symmetric strain modes "
            f"(design rank {rank}, expected {design.shape[1]})."
        )

    coefficients, _, fitted_rank, _ = np.linalg.lstsq(design, polarizations, rcond=None)
    if fitted_rank != rank:
        raise RuntimeError(f"Inconsistent least-squares ranks: {fitted_rank} != {rank}.")

    reference_polarization = coefficients[0]
    improper_data = voigt_to_piezoelectric(coefficients[1:].T)
    proper_data = proper_piezoelectric_tensor(improper_data, reference_polarization)
    residual = design @ coefficients - polarizations
    residual_rms = float(np.sqrt(np.mean(residual**2)))

    tensor_kwargs = {"cell": cell, "basis": basis}
    return PiezoelectricResult(
        reference_polarization=reference_polarization,
        improper=ImproperPiezoelectricTensor(data=improper_data, **tensor_kwargs),
        proper=ProperPiezoelectricTensor(data=proper_data, **tensor_kwargs),
        rank=rank,
        residual_rms=residual_rms,
    )


def evaluate_piezoelectric_from_structures(
    structures: Sequence[Atoms],
    reference: Atoms,
    polarization_key: str = "REF_polarization",
    unwrap: bool = True,
    polarization_quantum_scale: float = 1.0,
) -> PiezoelectricResult:
    """Evaluate both tensors from one sequence of polarized strained structures."""
    if not structures:
        raise ValueError("At least one strained structure is required.")
    if not np.all(reference.get_pbc()):
        raise ValueError("The reference structure must be fully periodic.")

    cells = np.asarray([atoms.cell.array for atoms in structures])
    polarizations = np.asarray([atoms.info[polarization_key] for atoms in structures])
    strains = strains_from_cells(reference.cell.array, cells)
    if unwrap:
        reference_index = int(np.argmin(np.linalg.norm(strains.reshape(len(strains), -1), axis=1)))
        polarizations = unwrap_polarizations(
            polarizations,
            cells,
            reference_index,
            quantum_scale=polarization_quantum_scale,
        )

    return evaluate_piezoelectric_tensors(
        polarizations,
        strains,
        cell=reference.cell.array,
        basis="cartesian",
    )
