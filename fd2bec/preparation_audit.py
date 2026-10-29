"""Read-only validation of finite-displacement preparation bundles."""

from dataclasses import dataclass, field
from io import StringIO, TextIOWrapper
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from zipfile import ZipFile

import numpy as np
from ase.io import read as ase_read

from fd2bec.displacements import mode_response_design_blocks
from fd2bec.tensor import BornCharges

REQUIRED_FILES = ("start.extxyz", "displacements.txt", "displaced-structures.extxyz")
COMPLETION_LOGS = ("fd2bec-log.txt", "prepare.log", "summary.txt")
COMPLETION_MARKER = "@ Job done :)"


@dataclass
class PreparationAudit:
    """Result of checking one prepared calculation bundle."""

    path: Path
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether the bundle has no required-data errors."""
        return not self.errors


class _DirectoryContents:
    """Uniform access to a preparation directory."""

    def __init__(self, root: Path):
        self.root = root

    def exists(self, name: str) -> bool:
        return (self.root / name).is_file()

    def names(self) -> Iterable[str]:
        for path in self.root.rglob("*"):
            if path.is_file():
                yield str(path.relative_to(self.root))

    def text(self, name: str) -> str:
        return (self.root / name).read_text(encoding="utf-8")

    def open_text(self, name: str):
        return (self.root / name).open("r", encoding="utf-8")


class _ArchiveContents:
    """Uniform access to the single preparation directory in a ZIP archive."""

    def __init__(self, archive: Path):
        self.archive = archive
        self.zip_file = ZipFile(str(archive))
        names = self.zip_file.namelist()
        starts = [name for name in names if name.endswith("/start.extxyz")]
        if len(starts) != 1:
            self.prefix = None
            self.error = "archive must contain exactly one preparation folder with start.extxyz"
        else:
            self.prefix = starts[0][: -len("start.extxyz")]
            self.error = None

    def close(self) -> None:
        self.zip_file.close()

    def _name(self, name: str) -> str:
        if self.prefix is None:
            raise ValueError(self.error)
        return self.prefix + name

    def exists(self, name: str) -> bool:
        return self.prefix is not None and self._name(name) in self.zip_file.namelist()

    def names(self) -> Iterable[str]:
        if self.prefix is None:
            return []
        return [
            name[len(self.prefix) :]
            for name in self.zip_file.namelist()
            if name.startswith(self.prefix)
        ]

    def text(self, name: str) -> str:
        return self.zip_file.read(self._name(name)).decode("utf-8")

    def open_text(self, name: str):
        return TextIOWrapper(self.zip_file.open(self._name(name)), encoding="utf-8")


def _count_extxyz_frames(stream) -> int:
    """Count and validate frames in an extxyz stream without loading it all."""
    frames = 0
    while True:
        line = stream.readline()
        while line and not line.strip():
            line = stream.readline()
        if not line:
            return frames
        try:
            number_of_atoms = int(line.strip())
        except ValueError as error:
            raise ValueError("expected an atom count") from error
        if number_of_atoms < 1:
            raise ValueError("atom count must be positive")
        for _ in range(number_of_atoms + 1):
            if not stream.readline():
                raise ValueError("truncated frame")
        frames += 1


def _count_nonempty_rows(stream) -> int:
    return sum(bool(line.strip()) for line in stream)


def _parse_metadata(text: str) -> Dict[str, str]:
    metadata = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            return {}
        metadata[key.strip()] = value.strip()
    return metadata


def _geometry_indices(names: Iterable[str]) -> Tuple[List[int], List[str]]:
    indices = []
    invalid = []
    prefix = "geometries/geometry.n="
    for name in names:
        if not name.startswith(prefix) or not name.endswith(".in"):
            continue
        index = name[len(prefix) : -len(".in")]
        if index.isdigit():
            indices.append(int(index))
        else:
            invalid.append(name)
    return sorted(indices), invalid


def _load_array(contents, name: str) -> np.ndarray:
    with contents.open_text(name) as stream:
        return np.atleast_2d(np.loadtxt(stream, comments="#"))


def _load_component_modes(contents, name: str) -> np.ndarray:
    text = contents.text(name)
    shape = None
    sparse = text.startswith("# fd2bec sparse matrix")
    for line in text.splitlines():
        if line.startswith("# shape:"):
            values = line.split(":", 1)[1].split()
            if len(values) == 2:
                shape = tuple(int(value) for value in values)
            break
    values = np.atleast_2d(np.loadtxt(StringIO(text), comments="#"))
    if not sparse:
        return values if shape is None else values.reshape(shape)
    if shape is None or values.shape[1] != 3:
        raise ValueError("invalid sparse component-mode cache")
    rows = values[:, 0].astype(int)
    columns = values[:, 1].astype(int)
    if (
        not np.allclose(values[:, 0], rows)
        or not np.allclose(values[:, 1], columns)
        or np.any(rows < 0)
        or np.any(rows >= shape[0])
        or np.any(columns < 0)
        or np.any(columns >= shape[1])
    ):
        raise ValueError("invalid sparse component-mode indices")
    modes = np.zeros(shape)
    modes[rows, columns] = values[:, 2]
    return modes


def _audit_bec_rank(contents, audit: PreparationAudit) -> None:
    """Check the current BEC fit rank from an archived preparation cache."""
    metadata_name = ".fd2bec/displacements.info"
    if not contents.exists(metadata_name):
        audit.warnings.append("no displacement cache metadata; cannot audit BEC selection rank")
        return
    metadata = _parse_metadata(contents.text(metadata_name))
    if metadata.get("what") != "bec":
        return
    if metadata.get("no_symmetry") == "true":
        audit.details.append(
            "BEC selection rank: not applicable (full Cartesian displacement mode)"
        )
        return

    required = (
        ".fd2bec/displacements.txt",
        ".fd2bec/born-charge-component-modes.txt",
        "start.extxyz",
    )
    missing = [name for name in required if not contents.exists(name)]
    if missing:
        audit.warnings.append(
            "cannot audit symmetry-reduced BEC rank; missing " + ", ".join(missing)
        )
        return
    try:
        displacements = _load_array(contents, ".fd2bec/displacements.txt")
        component_modes = _load_component_modes(contents, ".fd2bec/born-charge-component-modes.txt")
        atoms = ase_read(StringIO(contents.text("start.extxyz")), format="extxyz")
        tensor = BornCharges(data=np.zeros((len(atoms), 3, 3)), cell=atoms.cell)
        design = mode_response_design_blocks(displacements, tensor, component_modes)
        projected = design.reshape((-1, design.shape[-1]))
        offsets = np.tile(-np.eye(3), len(displacements)).T
        fit_matrix = np.hstack((projected, offsets))
    except (OSError, ValueError, TypeError) as error:
        audit.warnings.append("could not audit symmetry-reduced BEC rank: {}".format(error))
        return

    projected_rank = np.linalg.matrix_rank(projected)
    fit_rank = np.linalg.matrix_rank(fit_matrix)
    modes = component_modes.shape[0]
    unknowns = modes + 3
    audit.details.append(
        "BEC projected rank: {}/{}; fit rank: {}/{}".format(
            projected_rank, modes, fit_rank, unknowns
        )
    )
    if projected_rank < modes or fit_rank < unknowns:
        audit.errors.append(
            "symmetry-reduced BEC fit is underdetermined "
            "(projected rank {}/{}, fit rank {}/{})".format(
                projected_rank, modes, fit_rank, unknowns
            )
        )


def audit_preparation(
    contents, path: Path, deep: bool = False, check_rank: bool = False
) -> PreparationAudit:
    """Audit the files required to post-process one finite-displacement bundle."""
    audit = PreparationAudit(path=path)
    if getattr(contents, "error", None):
        audit.errors.append(contents.error)
        return audit

    names = set(contents.names())
    for name in REQUIRED_FILES:
        if not contents.exists(name):
            audit.errors.append("missing {}".format(name))
    if not any(contents.exists(name) for name in COMPLETION_LOGS):
        audit.errors.append("missing completion log")
    else:
        complete = any(
            contents.exists(name) and COMPLETION_MARKER in contents.text(name)
            for name in COMPLETION_LOGS
        )
        if not complete:
            audit.errors.append("no successful completion marker in preparation log")

    indices, invalid = _geometry_indices(names)
    if not indices:
        audit.errors.append("missing geometries/geometry.n=0.in")
    else:
        expected = list(range(indices[-1] + 1))
        if indices != expected:
            audit.errors.append("geometry indices are not consecutive from 0")
        audit.details.append("geometry files: {}".format(len(indices)))
    if invalid:
        audit.errors.append("invalid geometry filename(s): " + ", ".join(sorted(invalid)))

    if deep and all(contents.exists(name) for name in REQUIRED_FILES):
        try:
            with contents.open_text("displaced-structures.extxyz") as stream:
                frames = _count_extxyz_frames(stream)
            with contents.open_text("displacements.txt") as stream:
                displacement_rows = _count_nonempty_rows(stream)
        except (OSError, UnicodeError, ValueError) as error:
            audit.errors.append("cannot read displacement data: {}".format(error))
        else:
            audit.details.append(
                "frames: {}; displacement rows: {}".format(frames, displacement_rows)
            )
            if displacement_rows != frames:
                audit.errors.append(
                    "{} displacement rows; expected {}".format(displacement_rows, frames)
                )
            if len(indices) != frames:
                audit.errors.append("{} geometry files; expected {}".format(len(indices), frames))

    if check_rank:
        _audit_bec_rank(contents, audit)
    return audit


def audit_path(path: Path, deep: bool = False, check_rank: bool = False) -> PreparationAudit:
    """Audit either an unpacked preparation directory or a ZIP archive."""
    if path.suffix == ".zip":
        contents = _ArchiveContents(path)
        try:
            return audit_preparation(contents, path, deep=deep, check_rank=check_rank)
        finally:
            contents.close()
    return audit_preparation(_DirectoryContents(path), path, deep=deep, check_rank=check_rank)


def find_preparation_paths(paths: Sequence[Path]) -> List[Path]:
    """Discover unpacked and archived preparation bundles below ``paths``."""
    found = set()
    for path in paths:
        if path.is_file():
            if path.suffix == ".zip":
                found.add(path)
            continue
        if (path / "start.extxyz").is_file():
            found.add(path)
        found.update(start.parent for start in path.rglob("start.extxyz"))
        found.update(path.rglob("structure-*.zip"))
    return sorted(found)
