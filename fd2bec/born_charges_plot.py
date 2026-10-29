"""Directional visualisation of individual Born effective charge tensors.

For an atom displaced along a unit vector ``n``, its Born effective charge
maps that displacement to ``P_i(n) = Zstar_ij n_j``.  This module plots the
signed longitudinal part, the total magnitude, or the magnitude transverse to
the displacement direction.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from fd2bec.piezoelectric_plot import (
    LONGITUDINAL_COLORMAP,
    MAGNITUDE_COLORMAP,
    RESPONSE_COMPONENTS,
    spherical_directions,
)
from fd2bec.tools import tensor_data_from_atoms


def born_charges_from_atoms(atoms, keyword: str) -> np.ndarray:
    """Extract finite 3-by-3 Born effective charge tensors from ASE data."""
    values, _ = tensor_data_from_atoms(atoms, keyword, "bec")
    values = np.asarray(values, dtype=float)
    if values.ndim == 2 and values.shape[1:] == (9,):
        values = values.reshape((-1, 3, 3))
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError(
            f"Born-charge keyword {keyword!r} must have shape (atoms, 3, 3) or "
            f"(atoms, 9), got {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Born-charge keyword {keyword!r} contains non-finite values.")
    return values


def born_charge_from_atoms(atoms, keyword: str, atom_index: int) -> np.ndarray:
    """Extract one finite 3-by-3 Born effective charge tensor from ASE data."""
    if atom_index < 0:
        raise IndexError("The atom index must be non-negative.")
    values = born_charges_from_atoms(atoms, keyword)
    if atom_index >= len(values):
        raise IndexError(
            f"The atom index {atom_index} is outside the {len(values)} Born-charge tensors."
        )
    return values[atom_index]


def induced_polarization_response(born_charge: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Return ``Zstar_ij n_j`` for normalized displacement directions ``n``."""
    born_charge = np.asarray(born_charge, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if born_charge.shape != (3, 3):
        raise ValueError(
            f"A Born effective charge must have shape (3, 3), got {born_charge.shape}."
        )
    if directions.shape[-1:] != (3,):
        raise ValueError("Directions must end in three Cartesian components.")
    norms = np.linalg.norm(directions, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("Directions must be normalized.")
    return np.einsum("ij,...j->...i", born_charge, directions)


def directional_born_charge_response(
    born_charge: np.ndarray, directions: np.ndarray, component: str = "longitudinal"
) -> np.ndarray:
    """Evaluate longitudinal, total, or transverse directional BEC responses."""
    if component not in RESPONSE_COMPONENTS:
        choices = ", ".join(RESPONSE_COMPONENTS)
        raise ValueError(f"Unknown response component {component!r}; choose from {choices}.")
    polarization = induced_polarization_response(born_charge, directions)
    longitudinal = np.einsum("...i,...i->...", directions, polarization)
    if component == "longitudinal":
        return longitudinal
    if component == "magnitude":
        return np.linalg.norm(polarization, axis=-1)
    transverse = polarization - longitudinal[..., np.newaxis] * directions
    return np.linalg.norm(transverse, axis=-1)


def directional_surface(
    born_charge: np.ndarray, resolution: int = 91, component: str = "longitudinal"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Cartesian surface coordinates and directional BEC response values."""
    directions = spherical_directions(resolution)
    response = directional_born_charge_response(born_charge, directions, component)
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
            r"Longitudinal effective charge $n_i Z^*_{ij} n_j$",
            f"signed longitudinal effective charge [{unit}]",
        ),
        "magnitude": (
            r"Total effective charge $\left|Z^*_{ij} n_j\right|$",
            f"effective-charge magnitude [{unit}]",
        ),
        "transverse": (
            r"Transverse effective charge $\left|P - \hat n(\hat n \cdot P)\right|$",
            f"transverse effective-charge magnitude [{unit}]",
        ),
    }
    return labels[component]


def _draw_directional_surface(axis, surface, component, maximum, unit, plt, colors):
    """Draw one directional BEC surface and return its colour mapping."""
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


def plot_born_charge_surface(
    born_charge: np.ndarray,
    output: Union[str, Path],
    *,
    resolution: int = 91,
    unit: str = "e",
    max_abs: float = None,
    component: str = "longitudinal",
) -> np.ndarray:
    """Save one colour-coded 3D directional Born-effective-charge surface."""
    surface = directional_surface(born_charge, resolution=resolution, component=component)
    response = surface[3]
    maximum = _selected_maximum((response,), max_abs)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fd2bec-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    figure = plt.figure(figsize=(7, 6), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    mapper, label = _draw_directional_surface(axis, surface, component, maximum, unit, plt, colors)
    figure.colorbar(mapper, ax=axis, shrink=0.72, pad=0.08).set_label(label)
    figure.savefig(output, dpi=300)
    plt.close(figure)
    return response


def plot_born_charge_response_comparison(
    born_charge: np.ndarray,
    output: Union[str, Path],
    *,
    resolution: int = 91,
    unit: str = "e",
    max_abs: float = None,
) -> Dict[str, np.ndarray]:
    """Save longitudinal, total, and transverse BEC surfaces side by side."""
    surfaces = {
        component: directional_surface(born_charge, resolution=resolution, component=component)
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


def plot_born_charge_structure_comparison(
    born_charges: np.ndarray,
    output: Union[str, Path],
    *,
    labels=None,
    resolution: int = 91,
    unit: str = "e",
    max_abs: float = None,
) -> Dict[str, np.ndarray]:
    """Save a three-column directional-response grid with one atom per row."""
    born_charges = np.asarray(born_charges, dtype=float)
    if born_charges.ndim != 3 or born_charges.shape[1:] != (3, 3):
        raise ValueError(
            f"Born effective charges must have shape (atoms, 3, 3), got {born_charges.shape}."
        )
    if not len(born_charges):
        raise ValueError("At least one Born effective charge tensor is required.")
    if labels is None:
        labels = [f"atom {index}" for index in range(len(born_charges))]
    if len(labels) != len(born_charges):
        raise ValueError("One row label is required for every Born effective charge tensor.")

    surfaces = {
        component: [
            directional_surface(tensor, resolution=resolution, component=component)
            for tensor in born_charges
        ]
        for component in RESPONSE_COMPONENTS
    }
    maximum = _selected_maximum(
        [surface[3] for component_surfaces in surfaces.values() for surface in component_surfaces],
        max_abs,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fd2bec-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    figure, axes = plt.subplots(
        len(born_charges),
        len(RESPONSE_COMPONENTS),
        figsize=(18, 5 * len(born_charges)),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
        squeeze=False,
    )
    mappers = {}
    colorbar_labels = {}
    for row, label in enumerate(labels):
        for column, component in enumerate(RESPONSE_COMPONENTS):
            axis = axes[row, column]
            mapper, colorbar_label = _draw_directional_surface(
                axis, surfaces[component][row], component, maximum, unit, plt, colors
            )
            if row:
                axis.set_title("")
            if column == 0:
                axis.text2D(
                    -0.12,
                    0.5,
                    str(label),
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=14,
                    fontweight="bold",
                )
            mappers[component] = mapper
            colorbar_labels[component] = colorbar_label
    for column, component in enumerate(RESPONSE_COMPONENTS):
        figure.colorbar(
            mappers[component], ax=axes[:, column].tolist(), shrink=0.88, pad=0.02
        ).set_label(colorbar_labels[component])
    figure.savefig(output, dpi=300)
    plt.close(figure)
    return {
        component: np.asarray([surface[3] for surface in component_surfaces])
        for component, component_surfaces in surfaces.items()
    }
