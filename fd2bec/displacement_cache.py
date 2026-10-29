"""Human-readable persistent cache for finite-displacement workflows."""

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from ase.io import read as ase_read
from ase.io import write as ase_write

from fd2bec import ATOL

CACHE_VERSION = "1"
STRUCTURES_FILENAME = "structures.extxyz"
DISPLACEMENTS_FILENAME = "displacements.txt"
DISPLACEMENT_REFERENCE_FILENAME = "displacements-reference.extxyz"
DISPLACEMENT_METADATA_FILENAME = "displacements.info"
BORN_CHARGE_MODE_FILENAME = "born-charge-component-modes.txt"
BORN_CHARGE_REFERENCE_FILENAME = "born-charge-reference.extxyz"
BORN_CHARGE_MODE_METADATA_FILENAME = "born-charge-component-modes.info"
MODE_CACHE_ZERO_TOLERANCE = ATOL


def displacement_cache_metadata(
    what: str,
    amplitude: float,
    symprec: float,
    no_symmetry: bool,
    number: Optional[int],
    seed: Optional[int],
) -> Dict[str, str]:
    """Return the readable settings that determine a displacement dataset."""
    return {
        "cache_version": CACHE_VERSION,
        "what": str(what),
        "amplitude_angstrom": repr(float(amplitude)),
        "symprec": repr(float(symprec)),
        "no_symmetry": str(bool(no_symmetry)).lower(),
        "number": "" if number is None else str(number),
        "seed": "" if seed is None else str(seed),
    }


def born_charge_mode_cache_metadata(symprec: float) -> Dict[str, str]:
    """Return the readable settings that determine BEC component modes."""
    return {
        "cache_version": CACHE_VERSION,
        "symprec": repr(float(symprec)),
    }


def restore_displacement_cache(
    cache_dir: str,
    metadata: Dict[str, str],
    reference_file: str,
    structures_output: str,
    displacements_output: Optional[str] = None,
) -> bool:
    """Restore cached displacements when the reference and settings match."""
    cache_path = Path(cache_dir)
    structures_source = cache_path / STRUCTURES_FILENAME
    displacements_source = cache_path / DISPLACEMENTS_FILENAME
    reference_source = cache_path / DISPLACEMENT_REFERENCE_FILENAME
    metadata_source = cache_path / DISPLACEMENT_METADATA_FILENAME
    if not all(
        path.is_file()
        for path in (structures_source, displacements_source, reference_source, metadata_source)
    ):
        return False
    if _read_metadata(metadata_source) != metadata or not _same_structure_file(
        reference_file, reference_source
    ):
        return False

    _copy_file(structures_source, Path(structures_output))
    if displacements_output is not None:
        _copy_file(displacements_source, Path(displacements_output))
    return True


def save_displacement_cache(
    cache_dir: str,
    metadata: Dict[str, str],
    reference_file: str,
    structures_output: str,
    displacements: np.ndarray,
    displacement_format: str,
) -> Path:
    """Save a displacement dataset directly in ``cache_dir``."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    _copy_file(Path(reference_file), cache_path / DISPLACEMENT_REFERENCE_FILENAME)
    _copy_file(Path(structures_output), cache_path / STRUCTURES_FILENAME)
    np.savetxt(str(cache_path / DISPLACEMENTS_FILENAME), displacements, fmt=displacement_format)
    _write_metadata(cache_path / DISPLACEMENT_METADATA_FILENAME, metadata)
    return cache_path


def restore_born_charge_mode_cache(
    cache_dir: str, metadata: Dict[str, str], atoms: Any
) -> Optional[np.ndarray]:
    """Load BEC component modes when their reference and settings match."""
    cache_path = Path(cache_dir)
    data_path = cache_path / BORN_CHARGE_MODE_FILENAME
    reference_path = cache_path / BORN_CHARGE_REFERENCE_FILENAME
    metadata_path = cache_path / BORN_CHARGE_MODE_METADATA_FILENAME
    if not all(path.is_file() for path in (data_path, reference_path, metadata_path)):
        return None
    if _read_metadata(metadata_path) != metadata or not _same_atoms(
        atoms, ase_read(reference_path)
    ):
        return None
    try:
        return _load_component_modes(data_path)
    except (OSError, ValueError):
        return None


def save_born_charge_mode_cache(
    cache_dir: str, metadata: Dict[str, str], atoms: Any, component_modes: np.ndarray
) -> Path:
    """Save BEC component modes as a plain text matrix in ``cache_dir``."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    component_modes = np.asarray(component_modes)
    if component_modes.ndim != 2:
        raise ValueError("BEC component modes must be a two-dimensional matrix.")
    ase_write(str(cache_path / BORN_CHARGE_REFERENCE_FILENAME), atoms, format="extxyz")
    _write_sparse_matrix(cache_path / BORN_CHARGE_MODE_FILENAME, component_modes)
    _write_metadata(cache_path / BORN_CHARGE_MODE_METADATA_FILENAME, metadata)
    return cache_path


def _copy_file(source: Path, destination: Path) -> None:
    """Copy a file without failing when source and destination coincide."""
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(source), str(destination))


def _write_metadata(path: Path, metadata: Dict[str, str]) -> None:
    path.write_text(
        "# fd2bec cache settings\n"
        + "".join("{} = {}\n".format(key, value) for key, value in sorted(metadata.items())),
        encoding="utf-8",
    )


def _read_metadata(path: Path) -> Dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            return {}
        values[key.strip()] = value.strip()
    return values


def _matrix_shape(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# shape:"):
            values = line.split(":", 1)[1].split()
            if len(values) == 2:
                return tuple(int(value) for value in values)
    raise ValueError("Cached component-mode matrix does not declare its shape.")


def _load_component_modes(path: Path) -> np.ndarray:
    """Load either the current sparse cache format or a legacy dense matrix."""
    shape = _matrix_shape(path)
    values = np.atleast_2d(np.loadtxt(str(path), comments="#"))
    if not _is_sparse_matrix(path):
        return values.reshape(shape)
    if values.shape[1] != 3:
        raise ValueError("Cached component-mode matrix has an invalid column count.")

    rows = values[:, 0].astype(int)
    columns = values[:, 1].astype(int)
    if not (np.allclose(values[:, 0], rows) and np.allclose(values[:, 1], columns)):
        raise ValueError("Sparse component-mode indices must be integers.")
    if (
        np.any(rows < 0)
        or np.any(rows >= shape[0])
        or np.any(columns < 0)
        or np.any(columns >= shape[1])
    ):
        raise ValueError("Sparse component-mode index is outside the declared shape.")
    matrix = np.zeros(shape)
    matrix[rows, columns] = values[:, 2]
    return matrix


def _is_sparse_matrix(path: Path) -> bool:
    """Whether ``path`` uses the coordinate-table cache format."""
    with path.open("r", encoding="utf-8") as stream:
        return stream.readline().strip() == "# fd2bec sparse matrix"


def _write_sparse_matrix(path: Path, matrix: np.ndarray) -> None:
    """Write significant matrix entries as readable row, column, value triples."""
    rows, columns = np.nonzero(np.abs(matrix) > MODE_CACHE_ZERO_TOLERANCE)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# fd2bec sparse matrix\n")
        stream.write("# shape: {} {}\n".format(*matrix.shape))
        stream.write("# zero_tolerance: {:.1e}\n".format(MODE_CACHE_ZERO_TOLERANCE))
        stream.write("# row column value\n")
        for row, column in zip(rows, columns):
            stream.write("{} {} {:.12e}\n".format(row, column, matrix[row, column]))


def _same_structure_file(first: str, second: Path) -> bool:
    return _same_atoms(ase_read(first), ase_read(str(second)))


def _same_atoms(first: Any, second: Any) -> bool:
    return (
        first.get_chemical_symbols() == second.get_chemical_symbols()
        and np.array_equal(first.get_pbc(), second.get_pbc())
        and np.allclose(first.get_positions(), second.get_positions(), rtol=0.0, atol=1e-10)
        and np.allclose(first.get_cell(), second.get_cell(), rtol=0.0, atol=1e-10)
    )
