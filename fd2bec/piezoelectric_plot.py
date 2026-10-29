"""Directional visualisation of piezoelectric tensors.

For uniaxial strain along a unit vector ``n``, the induced polarization per
unit strain is ``P_i(n) = e_ijk n_j n_k``.  The module plots its signed
longitudinal part, its total magnitude, or the magnitude transverse to ``n``.
Surfaces use absolute response as their radius; a signed radial coordinate
would duplicate and fold the longitudinal surface.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from fd2bec.piezoelectric import voigt_to_piezoelectric

RESPONSE_COMPONENTS = ("longitudinal", "magnitude", "transverse")
LONGITUDINAL_COLORMAP = "coolwarm"
MAGNITUDE_COLORMAP = "viridis"


def read_piezoelectric_voigt(filename: Union[str, Path]) -> np.ndarray:
    """Read a finite 3-by-6 piezoelectric matrix from a text or CSV file.

    A flat sequence of 18 values is accepted in addition to the standard
    three-row format written by :mod:`fd2bec.cli.dPdS.dPdS2piezo`.
    """
    path = Path(filename)
    delimiter = "," if path.suffix.lower() == ".csv" else None
    values = np.loadtxt(path, delimiter=delimiter, ndmin=1)
    if values.size != 18:
        raise ValueError(
            f"Piezoelectric input must contain 18 values (a 3 x 6 matrix), got {values.size}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Piezoelectric input contains non-finite values.")
    return np.asarray(values, dtype=float).reshape((3, 6))


def spherical_directions(resolution: int) -> np.ndarray:
    """Return a regular full-sphere direction mesh of shape ``(N, 2N+1, 3)``."""
    if resolution < 3:
        raise ValueError("The sphere resolution must be at least 3.")
    polar = np.linspace(0.0, np.pi, resolution)
    azimuth = np.linspace(0.0, 2.0 * np.pi, 2 * resolution + 1)
    polar, azimuth = np.meshgrid(polar, azimuth, indexing="ij")
    return np.stack(
        (
            np.sin(polar) * np.cos(azimuth),
            np.sin(polar) * np.sin(azimuth),
            np.cos(polar),
        ),
        axis=-1,
    )


def induced_polarization_response(
    piezoelectric_voigt: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    """Return the induced polarization per unit uniaxial strain.

    The input uses fd2bec's Voigt order ``xx, yy, zz, yz, xz, xy`` and its
    engineering-shear convention.  Contracting the expanded Cartesian tensor
    with ``n_j n_k`` gives ``P_i(n) = e_ijk n_j n_k``.
    """
    piezoelectric_voigt = np.asarray(piezoelectric_voigt, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if piezoelectric_voigt.shape != (3, 6):
        raise ValueError(
            "Piezoelectric data must have shape (3, 6) in Voigt order xx, yy, zz, yz, xz, xy."
        )
    if directions.shape[-1:] != (3,):
        raise ValueError("Directions must end in three Cartesian components.")
    norms = np.linalg.norm(directions, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("Directions must be normalized.")
    tensor = voigt_to_piezoelectric(piezoelectric_voigt)
    return np.einsum("ijk,...j,...k->...i", tensor, directions, directions)


def longitudinal_piezoelectric_response(
    piezoelectric_voigt: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    """Evaluate the signed longitudinal response ``n_i e_ijk n_j n_k``."""
    polarization = induced_polarization_response(piezoelectric_voigt, directions)
    return np.einsum("...i,...i->...", directions, polarization)


def directional_piezoelectric_response(
    piezoelectric_voigt: np.ndarray, directions: np.ndarray, component: str = "longitudinal"
) -> np.ndarray:
    """Evaluate one scalar directional piezoelectric response component.

    ``longitudinal`` is the signed polarization parallel to the strain
    direction, ``magnitude`` is the total induced-polarization magnitude, and
    ``transverse`` is the magnitude perpendicular to the strain direction.
    """
    if component not in RESPONSE_COMPONENTS:
        choices = ", ".join(RESPONSE_COMPONENTS)
        raise ValueError(f"Unknown response component {component!r}; choose from {choices}.")
    polarization = induced_polarization_response(piezoelectric_voigt, directions)
    longitudinal = np.einsum("...i,...i->...", directions, polarization)
    if component == "longitudinal":
        return longitudinal
    if component == "magnitude":
        return np.linalg.norm(polarization, axis=-1)
    transverse = polarization - longitudinal[..., np.newaxis] * directions
    return np.linalg.norm(transverse, axis=-1)


def directional_surface(
    piezoelectric_voigt: np.ndarray, resolution: int = 91, component: str = "longitudinal"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Cartesian surface coordinates and directional response values."""
    directions = spherical_directions(resolution)
    response = directional_piezoelectric_response(piezoelectric_voigt, directions, component)
    coordinates = np.abs(response)[..., np.newaxis] * directions
    return coordinates[..., 0], coordinates[..., 1], coordinates[..., 2], response


def _selected_maximum(responses, max_abs: float) -> float:
    """Select and validate a shared magnitude limit for one or more responses."""
    response_maximum = max(float(np.max(np.abs(response))) for response in responses)
    if max_abs is None:
        return response_maximum
    if max_abs <= 0.0:
        raise ValueError("max_abs must be positive when specified.")
    if max_abs < response_maximum:
        raise ValueError(
            f"max_abs ({max_abs:.6e}) is smaller than the response maximum "
            f"({response_maximum:.6e})."
        )
    return max_abs


def _surface_labels(component: str, unit: str) -> Tuple[str, str]:
    """Return title and colour-bar label for a response component."""
    labels = {
        "longitudinal": (
            r"Longitudinal response $n_i e_{ijk} n_j n_k$",
            f"signed longitudinal coefficient [{unit}]",
        ),
        "magnitude": (
            r"Total induced response $\left|e_{ijk} n_j n_k\right|$",
            f"induced-polarization magnitude [{unit}]",
        ),
        "transverse": (
            r"Transverse response $\left|P - \hat n(\hat n \cdot P)\right|$",
            f"transverse induced-polarization magnitude [{unit}]",
        ),
    }
    return labels[component]


def _draw_directional_surface(axis, surface, component, maximum, unit, plt, colors):
    """Draw one directional-response surface and return its colour mapping."""
    x, y, z, response = surface
    display_maximum = maximum if maximum > 0.0 else np.finfo(float).eps
    if component == "longitudinal":
        norm = colors.Normalize(vmin=-display_maximum, vmax=display_maximum)
        colormap = plt.get_cmap(LONGITUDINAL_COLORMAP)
    else:
        norm = colors.Normalize(vmin=0.0, vmax=display_maximum)
        colormap = plt.get_cmap(MAGNITUDE_COLORMAP)
    axis.plot_surface(
        x,
        y,
        z,
        facecolors=colormap(norm(response)),
        rcount=x.shape[0],
        ccount=x.shape[1],
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    limits = maximum if maximum > 0.0 else 1.0
    axis.set(xlim=(-limits, limits), ylim=(-limits, limits), zlim=(-limits, limits))
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    title, label = _surface_labels(component, unit)
    axis.set_title(title)
    if float(np.max(np.abs(response))) == 0.0:
        axis.text2D(
            0.5,
            0.52,
            "zero response",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )
    return plt.cm.ScalarMappable(norm=norm, cmap=colormap), label


def plot_piezoelectric_surface(
    piezoelectric_voigt: np.ndarray,
    output: Union[str, Path],
    *,
    resolution: int = 91,
    unit: str = "e/Angstrom^2",
    max_abs: float = None,
    component: str = "longitudinal",
) -> np.ndarray:
    """Save one colour-coded 3D directional piezoelectric response surface.

    The returned response array makes the numerical directional values
    available to callers without requiring them to reconstruct the mesh.
    """
    surface = directional_surface(piezoelectric_voigt, resolution=resolution, component=component)
    response = surface[3]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fd2bec-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    maximum = _selected_maximum((response,), max_abs)

    figure = plt.figure(figsize=(7, 6), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    mapper, label = _draw_directional_surface(axis, surface, component, maximum, unit, plt, colors)
    figure.colorbar(mapper, ax=axis, shrink=0.72, pad=0.08).set_label(label)
    figure.savefig(output, dpi=300)
    plt.close(figure)
    return response


def plot_piezoelectric_response_comparison(
    piezoelectric_voigt: np.ndarray,
    output: Union[str, Path],
    *,
    resolution: int = 91,
    unit: str = "e/Angstrom^2",
    max_abs: float = None,
) -> Dict[str, np.ndarray]:
    """Save longitudinal, total, and transverse response surfaces side by side."""
    surfaces = {
        component: directional_surface(
            piezoelectric_voigt, resolution=resolution, component=component
        )
        for component in RESPONSE_COMPONENTS
    }
    maximum = _selected_maximum([surface[3] for surface in surfaces.values()], max_abs)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fd2bec-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    figure, axes = plt.subplots(
        1, 3, figsize=(18, 6), subplot_kw={"projection": "3d"}, constrained_layout=True
    )
    for axis, component in zip(axes, RESPONSE_COMPONENTS):
        mapper, label = _draw_directional_surface(
            axis, surfaces[component], component, maximum, unit, plt, colors
        )
        figure.colorbar(mapper, ax=axis, shrink=0.68, pad=0.08).set_label(label)
    figure.savefig(output, dpi=300)
    plt.close(figure)
    return {component: surface[3] for component, surface in surfaces.items()}
